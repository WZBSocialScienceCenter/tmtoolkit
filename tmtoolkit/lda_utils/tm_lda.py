# -*- coding: utf-8 -*-
import logging

from lda import LDA

from .common import MultiprocModelsRunner, MultiprocModelsWorkerABC, MultiprocEvaluationRunner, \
    MultiprocEvaluationWorkerABC
from .eval_metrics import metric_griffiths_2004, metric_cao_juan_2009, metric_arun_2010

try:
    import gmpy2
    AVAILABLE_METRICS = (
        'loglikelihood',    # simply uses the last reported log likelihood
        'griffiths_2004',
        'cao_juan_2009',
        'arun_2010'
    )
except ImportError:  # if gmpy2 is not available: do not use 'griffiths_2004'
    AVAILABLE_METRICS = (
        'loglikelihood',    # simply uses the last reported log likelihood
        'cao_juan_2009',
        'arun_2010'
    )


logger = logging.getLogger('tmtoolkit')


class MultiprocModelsWorkerLDA(MultiprocModelsWorkerABC):
    package_name = 'lda'

    def fit_model(self, data, params):
        lda_instance = LDA(**params)
        lda_instance.fit(data)

        return lda_instance


class MultiprocEvaluationWorkerLDA(MultiprocEvaluationWorkerABC, MultiprocModelsWorkerLDA):
    def fit_model(self, data, params):
        lda_instance = super(MultiprocEvaluationWorkerLDA, self).fit_model(data, params)

        results = {}
        if self.return_models:
            results['model'] = lda_instance

        for metric in self.eval_metric:
            if metric == 'griffiths_2004':
                burnin = self.eval_metric_options.get('griffiths_2004_burnin', 50) // lda_instance.refresh
                if burnin >= len(lda_instance.loglikelihoods_):
                    raise ValueError('`griffiths_2004_burnin` set too high (%d). should be less than %d.'
                                     % (burnin, len(lda_instance.loglikelihoods_) * lda_instance.refresh))
                logliks = lda_instance.loglikelihoods_[burnin:]
                res = metric_griffiths_2004(logliks)
            elif metric == 'cao_juan_2009':
                res = metric_cao_juan_2009(lda_instance.topic_word_)
            elif metric == 'arun_2010':
                res = metric_arun_2010(lda_instance.topic_word_, lda_instance.doc_topic_, data.sum(axis=1))
            else:  # default: loglikelihood
                res = lda_instance.loglikelihoods_[-1]

            logger.info('> evaluation result with metric "%s": %f' % (metric, res))
            results[metric] = res

        return results


def compute_models_parallel(data, varying_parameters=None, constant_parameters=None, n_max_processes=None):
    """
    Compute several Topic Models in parallel using the "lda" package. Use a single or multiple document term matrices
    `data` and optionally a list of varying parameters `varying_parameters`. Pass parameters in `constant_parameters`
    dict to each model calculation. Use at maximum `n_max_processes` processors or use all available processors if None
    is passed.
    `data` can be either a Document-Term-Matrix (NumPy array/matrix, SciPy sparse matrix) or a dict with document ID ->
    Document-Term-Matrix mapping when calculating models for multiple corpora (named multiple documents).

    If `data` is a dict of named documents, this function will return a dict with document ID -> result list. Otherwise
    it will only return a result list. A result list always is a list containing tuples `(parameter_set, model)` where
    `parameter_set` is a dict of the used parameters.
    """
    mp_models = MultiprocModelsRunner(MultiprocModelsWorkerLDA, data, varying_parameters, constant_parameters,
                                      n_max_processes=n_max_processes)

    return mp_models.run()


def evaluate_topic_models(data, varying_parameters, constant_parameters=None, n_max_processes=None, return_models=False,
                          metric=None, **metric_kwargs):
    """
    Compute several Topic Models in parallel using the "lda" package. Calculate the models using a list of varying
    parameters `varying_parameters` on a single Document-Term-Matrix `data`. Pass parameters in `constant_parameters`
    dict to each model calculation. Use at maximum `n_max_processes` processors or use all available processors if None
    is passed.
    `data` must be a Document-Term-Matrix (NumPy array/matrix, SciPy sparse matrix).
    Will return a list of size `len(varying_parameters)` containing tuples `(parameter_set, eval_results)` where
    `parameter_set` is a dict of the used parameters and `eval_results` is a dict of metric names -> metric results.
    """
    mp_eval = MultiprocEvaluationRunner(MultiprocEvaluationWorkerLDA, AVAILABLE_METRICS, data,
                                        varying_parameters, constant_parameters,
                                        metric=metric, metric_options=metric_kwargs,
                                        n_max_processes=n_max_processes, return_models=return_models)

    return mp_eval.run()
