"""
Parallel model computation and evaluation using the `lda package <https://github.com/lda-project/lda>`_.

Available evaluation metrics for this module are listed in :data:`~tmtoolkit.topicmod.tm_lda.AVAILABLE_METRICS`.
See :mod:`tmtoolkit.topicmod.evaluate` for references and implementations of those evaluation metrics.
"""

import logging
import importlib.util

import numpy as np

from ._eval_tools import split_dtm_for_cross_validation
from tmtoolkit.topicmod.parallel import MultiprocModelsRunner, MultiprocModelsWorkerABC, MultiprocEvaluationRunner, \
    MultiprocEvaluationWorkerABC
from .evaluate import metric_griffiths_2004, metric_cao_juan_2009, metric_arun_2010, metric_coherence_mimno_2011, \
    metric_coherence_gensim, metric_held_out_documents_wallach09

if importlib.util.find_spec('gmpy2'):
    metrics_using_gmpy2 = ('griffiths_2004', 'held_out_documents_wallach09')
else:  # if gmpy2 is not available: do not use 'griffiths_2004'
    metrics_using_gmpy2 = ()

if importlib.util.find_spec('gensim'):
    metrics_using_gensim = (
        'coherence_gensim_u_mass',      # same as coherence_mimno_2011
        'coherence_gensim_c_v',
        'coherence_gensim_c_uci',
        'coherence_gensim_c_npmi'
    )
else:
    metrics_using_gensim = ()


#: Available metrics for lda (``"griffiths_2004"``, ``"held_out_documents_wallach09"`` are added when package gmpy2
#: is installed, several ``"coherence_gensim_"`` metrics are added when package gensim is installed).
AVAILABLE_METRICS = (
    'loglikelihood',                # simply uses the last reported log likelihood as fallback
    'cao_juan_2009',
    'arun_2010',
    'coherence_mimno_2011',
) + metrics_using_gmpy2 + metrics_using_gensim

#: Metrics used by default.
DEFAULT_METRICS = (
    'cao_juan_2009',
    'coherence_mimno_2011'
)


logger = logging.getLogger('tmtoolkit')


#%% Specialized classes for parallel processing


class MultiprocModelsWorkerLDA(MultiprocModelsWorkerABC):
    """
    Specialized parallel model computations worker for lda.
    """

    package_name = 'lda'

    def fit_model(self, data, params):
        from lda import LDA
        lda_instance = LDA(**params)
        lda_instance.fit(data)

        return lda_instance


class MultiprocEvaluationWorkerLDA(MultiprocEvaluationWorkerABC, MultiprocModelsWorkerLDA):
    """
    Specialized parallel model evaluations worker for lda.
    """

    def fit_model(self, data, params):
        if list(self.eval_metric) != ['held_out_documents_wallach09'] or self.return_models:
            lda_instance = super(MultiprocEvaluationWorkerLDA, self).fit_model(data, params)
        else:
            lda_instance = None

        results = {}
        if self.return_models:
            results['model'] = lda_instance

        for metric in self.eval_metric:
            if metric == 'griffiths_2004':
                if 'griffiths_2004_burnin' in self.eval_metric_options:  # discard specific number of burnin iterations
                    burnin_iterations = self.eval_metric_options['griffiths_2004_burnin']
                    burnin_samples = burnin_iterations // lda_instance.refresh

                    if burnin_samples >= len(lda_instance.loglikelihoods_):
                        raise ValueError('`griffiths_2004_burnin` set too high (%d) – not enough samples to use. should be less than %d.'
                                         % (burnin_iterations, len(lda_instance.loglikelihoods_) * lda_instance.refresh))
                else:   # default: discard first 50% of the likelihood samples
                    burnin_samples = len(lda_instance.loglikelihoods_) // 2

                logliks = lda_instance.loglikelihoods_[burnin_samples:]
                if logliks:
                    res = metric_griffiths_2004(logliks)
                else:
                    raise ValueError('no log likelihood samples for calculation of `metric_griffiths_2004`')
            elif metric == 'cao_juan_2009':
                res = metric_cao_juan_2009(lda_instance.topic_word_)
            elif metric == 'arun_2010':
                res = metric_arun_2010(lda_instance.topic_word_, lda_instance.doc_topic_, data.sum(axis=1))
            elif metric == 'coherence_mimno_2011':
                default_top_n = min(20, lda_instance.topic_word_.shape[1])
                res = metric_coherence_mimno_2011(lda_instance.topic_word_, data,
                                                  top_n=self.eval_metric_options.get('coherence_mimno_2011_top_n',
                                                                                     default_top_n),
                                                  eps=self.eval_metric_options.get('coherence_mimno_2011_eps', 1e-12),
                                                  return_mean=True)
            elif metric.startswith('coherence_gensim_'):
                if 'coherence_gensim_vocab' not in self.eval_metric_options:
                    raise ValueError('corpus vocabulary must be passed as `coherence_gensim_vocab`')

                coh_measure = metric[len('coherence_gensim_'):]
                default_top_n = min(20, lda_instance.topic_word_.shape[1])
                metric_kwargs = {
                    'measure': coh_measure,
                    'topic_word_distrib': lda_instance.topic_word_,
                    'dtm': data,
                    'vocab': self.eval_metric_options['coherence_gensim_vocab'],
                    'return_mean': True,
                    'processes': 1,
                    'top_n': self.eval_metric_options.get('coherence_gensim_top_n', default_top_n),
                }

                if coh_measure != 'u_mass':
                    if 'coherence_gensim_texts' not in self.eval_metric_options:
                        raise ValueError('tokenized documents must be passed as `coherence_gensim_texts` for any other '
                                         'coherence measure than `u_mass`')
                    metric_kwargs.update({
                        'texts': self.eval_metric_options['coherence_gensim_texts']
                    })

                metric_kwargs.update(self.eval_metric_options.get('coherence_gensim_kwargs', {}))

                res = metric_coherence_gensim(**metric_kwargs)
            elif metric == 'held_out_documents_wallach09':
                n_folds = self.eval_metric_options.get('held_out_documents_wallach09_n_folds', 5)
                shuffle_docs = self.eval_metric_options.get('held_out_documents_wallach09_shuffle_docs', True)
                n_samples = self.eval_metric_options.get('held_out_documents_wallach09_n_samples', 10000)

                folds_results = []
                # TODO: parallelize this
                for fold, train, test in split_dtm_for_cross_validation(data, n_folds, shuffle_docs=shuffle_docs):
                    logger.info('> fold %d/%d of cross validation with %d held-out documents and %d training documents'
                                % (fold+1, n_folds, test.shape[0], train.shape[0]))

                    model_train = super(MultiprocEvaluationWorkerLDA, self).fit_model(train, params)
                    theta_test = model_train.transform(test)

                    folds_results.append(metric_held_out_documents_wallach09(test, theta_test, model_train.topic_word_,
                                                                             model_train.alpha, n_samples=n_samples))

                logger.debug('> cross validation results with metric "%s": %s' % (metric, str(folds_results)))
                res = np.mean(folds_results)
            elif metric == 'loglikelihood':
                res = lda_instance.loglikelihoods_[-1]
            else:
                raise ValueError('metric not available: "%s"' % metric)

            logger.info('> evaluation result with metric "%s": %f' % (metric, res))
            results[metric] = res

        return results


#%% main API functions for parallel processing


def compute_models_parallel(data, varying_parameters=None, constant_parameters=None, n_max_processes=None):
    """
    Compute several topic models in parallel using the "lda" package. Use a single or multiple document term matrices
    `data` and optionally a list of varying parameters `varying_parameters`. Pass parameters in `constant_parameters`
    dict to each model calculation. Use at maximum `n_max_processes` processors or use all available processors if None
    is passed.

    `data` can be either a Document-Term-Matrix (NumPy array/matrix, SciPy sparse matrix) or a dict with corpus ID ->
    Document-Term-Matrix mapping when calculating models for multiple corpora.

    If `data` is a dict of named matrices, this function will return a dict with document ID -> result list. Otherwise
    it will only return a result list. A result list always is a list containing tuples `(parameter_set, model)` where
    `parameter_set` is a dict of the used parameters.

    :param data: either a (sparse) 2D array/matrix or a dict mapping dataset labels to such matrices
    :param varying_parameters: list of dicts with parameters; each parameter set will be used in a separate
                               computation
    :param constant_parameters: dict with parameters that are the same for all parallel computations
    :param n_max_processes: maximum number of worker processes to spawn
    :return: if passed data is 2D array, returns a list with tuples (parameter set, results); if passed data is
             a dict of 2D arrays, returns dict with same keys as data and the respective results for each dataset
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
    `parameter_set` is a dict of the used parameters and `eval_results` is a dict of metric names -> metric results:

    .. code-block:: text

        [(parameter_set_1, {'<metric_name>': result_1, ...}),
         ...,
         (parameter_set_n, {'<metric_name>': result_n, ...})])

    .. seealso:: Results can be simplified using :func:`tmtoolkit.topicmod.evaluate.results_by_parameter`.

    :param data: a (sparse) 2D array/matrix
    :param varying_parameters: list of dicts with parameters; each parameter set will be used in a separate
                               evaluation
    :param constant_parameters: dict with parameters that are the same for all parallel computations
    :param n_max_processes: maximum number of worker processes to spawn
    :param return_models: if True, also return the computed models in the evaluation results
    :param metric: string or list of strings; if given, use only this metric(s) for evaluation; must be subset of
                   `available_metrics`
    :param metric_kwargs: dict of options for metric used metric(s)
    :return: list of evaluation results for each varying parameter set as described above
    """
    mp_eval = MultiprocEvaluationRunner(MultiprocEvaluationWorkerLDA, AVAILABLE_METRICS, data,
                                        varying_parameters, constant_parameters,
                                        metric=metric or DEFAULT_METRICS, metric_options=metric_kwargs,
                                        n_max_processes=n_max_processes, return_models=return_models)

    return mp_eval.run()
