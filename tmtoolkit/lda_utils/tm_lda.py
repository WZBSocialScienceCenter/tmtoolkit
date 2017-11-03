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

    def fit_model_using_params(self, params):
        lda_instance = LDA(**params)
        lda_instance.fit(self.data)

        return lda_instance


class MultiprocEvaluationWorkerLDA(MultiprocEvaluationWorkerABC, MultiprocModelsWorkerLDA):
    def fit_model_using_params(self, params):
        lda_instance = super(MultiprocEvaluationWorkerLDA, self).fit_model_using_params(params)

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
                res = metric_arun_2010(lda_instance.topic_word_, lda_instance.doc_topic_, self.data.sum(axis=1))
            else:  # default: loglikelihood
                res = lda_instance.loglikelihoods_[-1]

            logger.info('> evaluation result with metric "%s": %f' % (metric, res))
            results[metric] = res

        return results


def compute_models_parallel(data, varying_parameters, constant_parameters=None, n_max_processes=None):
    mp_models = MultiprocModelsRunner(MultiprocModelsWorkerLDA, data, varying_parameters, constant_parameters,
                                      n_max_processes=n_max_processes)

    return mp_models.run()


def evaluate_topic_models(data, varying_parameters, constant_parameters=None, n_max_processes=None, return_models=False,
                          metric=None, **metric_kwargs):
    mp_eval = MultiprocEvaluationRunner(MultiprocEvaluationWorkerLDA, AVAILABLE_METRICS, data,
                                        varying_parameters, constant_parameters,
                                        metric=metric, metric_options=metric_kwargs,
                                        n_max_processes=n_max_processes, return_models=return_models)

    return mp_eval.run()
