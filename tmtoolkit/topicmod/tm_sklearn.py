"""
Parallel model computation and evaluation using the `scikit-learn package <https://scikit-learn.org/>`_.

Available evaluation metrics for this module are listed in :data:`~tmtoolkit.topicmod.tm_sklearn.AVAILABLE_METRICS`.
See :mod:`tmtoolkit.topicmod.evaluate` for references and implementations of those evaluation metrics.
"""

import logging
import importlib.util

import numpy as np
from scipy.sparse import issparse, csr_matrix

from ._eval_tools import split_dtm_for_cross_validation
from tmtoolkit.topicmod.parallel import MultiprocModelsRunner, MultiprocModelsWorkerABC, MultiprocEvaluationRunner, \
    MultiprocEvaluationWorkerABC
from .evaluate import metric_cao_juan_2009, metric_arun_2010, metric_coherence_mimno_2011, \
    metric_coherence_gensim, metric_held_out_documents_wallach09


if importlib.util.find_spec('gmpy2'):
    metrics_using_gmpy2 = ('held_out_documents_wallach09', )
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

#: Available metrics for sklearn (``"held_out_documents_wallach09"`` is added when package gmpy2
#: is installed, several ``"coherence_gensim_"`` metrics are added when package gensim is installed).
AVAILABLE_METRICS = (
    'perplexity',
    'cao_juan_2009',
    'arun_2010',
    'coherence_mimno_2011',
    'coherence_gensim_u_mass',  # same as coherence_mimno_2011
    'coherence_gensim_c_v',
    'coherence_gensim_c_uci',
    'coherence_gensim_c_npmi',
) + metrics_using_gmpy2 + metrics_using_gensim

#: Metrics used by default.
DEFAULT_METRICS = (
    'perplexity',
    'cao_juan_2009',
    'coherence_mimno_2011'
)


#%% Specialized classes for parallel processing


logger = logging.getLogger('tmtoolkit')


class MultiprocModelsWorkerSklearn(MultiprocModelsWorkerABC):
    """
    Specialized parallel model computations worker for sklearn.
    """

    package_name = 'sklearn'

    def fit_model(self, data, params, return_data=False):
        from sklearn.decomposition import LatentDirichletAllocation

        if issparse(data):
            if data.format != 'csr':
                data = data.tocsr()
        else:
            data = csr_matrix(data)

        lda_instance = LatentDirichletAllocation(**params)
        lda_instance.fit(data)

        if return_data:
            return lda_instance, data
        else:
            return lda_instance


class MultiprocEvaluationWorkerSklearn(MultiprocEvaluationWorkerABC, MultiprocModelsWorkerSklearn):
    """
    Specialized parallel model evaluations worker for sklearn.
    """

    def fit_model(self, data, params, return_data=False):
        lda_instance, data = super(MultiprocEvaluationWorkerSklearn, self).fit_model(data, params,
                                                                                     return_data=True)

        topic_word_distrib = _get_normalized_topic_word_distrib(lda_instance)

        results = {}
        if self.return_models:
            results['model'] = lda_instance

        for metric in self.eval_metric:
            if metric == 'cao_juan_2009':
                res = metric_cao_juan_2009(topic_word_distrib)
            elif metric == 'arun_2010':
                res = metric_arun_2010(topic_word_distrib, lda_instance.transform(data), data.sum(axis=1))
            elif metric == 'coherence_mimno_2011':
                default_top_n = min(20, topic_word_distrib.shape[1])
                res = metric_coherence_mimno_2011(topic_word_distrib, data,
                                                  top_n=self.eval_metric_options.get('coherence_mimno_2011_top_n', default_top_n),
                                                  eps=self.eval_metric_options.get('coherence_mimno_2011_eps', 1e-12),
                                                  return_mean=True)
            elif metric.startswith('coherence_gensim_'):
                if 'coherence_gensim_vocab' not in self.eval_metric_options:
                    raise ValueError('corpus vocabulary must be passed as `coherence_gensim_vocab`')

                coh_measure = metric[len('coherence_gensim_'):]
                default_top_n = min(20, topic_word_distrib.shape[1])
                metric_kwargs = {
                    'measure': coh_measure,
                    'topic_word_distrib': topic_word_distrib,
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

                    model_train = super(MultiprocEvaluationWorkerSklearn, self).fit_model(train, params)
                    theta_test = model_train.transform(test)

                    phi_train = _get_normalized_topic_word_distrib(lda_instance)

                    folds_results.append(metric_held_out_documents_wallach09(test, theta_test, phi_train,
                                                                             model_train.doc_topic_prior_,
                                                                             n_samples=n_samples))

                logger.debug('> cross validation results with metric "%s": %s' % (metric, str(folds_results)))
                res = np.mean(folds_results)
            elif metric == 'perplexity':
                res = lda_instance.perplexity(data)
            else:
                raise ValueError('metric not available: "%s"' % metric)

            logger.info('> evaluation result with metric "%s": %f' % (metric, res))
            results[metric] = res

        return results


#%% main API functions for parallel processing


def compute_models_parallel(data, varying_parameters=None, constant_parameters=None, n_max_processes=None):
    """
    Compute several topic models in parallel using the "sklearn" package. Use a single or multiple document term matrices
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

    mp_models = MultiprocModelsRunner(MultiprocModelsWorkerSklearn, data, varying_parameters, constant_parameters,
                                      n_max_processes=n_max_processes)

    return mp_models.run()


def evaluate_topic_models(data, varying_parameters, constant_parameters=None, n_max_processes=None, return_models=False,
                          metric=None, **metric_kwargs):
    """
    Compute several Topic Models in parallel using the "sklearn" package. Calculate the models using a list of varying
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

    mp_eval = MultiprocEvaluationRunner(MultiprocEvaluationWorkerSklearn, AVAILABLE_METRICS, data,
                                        varying_parameters, constant_parameters,
                                        metric=metric or DEFAULT_METRICS, metric_options=metric_kwargs,
                                        n_max_processes=n_max_processes, return_models=return_models)

    return mp_eval.run()


#%% Helper functions

def _get_normalized_topic_word_distrib(lda_instance):
    return lda_instance.components_ / lda_instance.components_.sum(axis=1)[:, np.newaxis]
