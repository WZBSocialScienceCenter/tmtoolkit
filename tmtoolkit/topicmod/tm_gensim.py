"""
Parallel model computation and evaluation using the `Gensim package <https://radimrehurek.com/gensim/>`_.

Available evaluation metrics for this module are listed in :data:`~tmtoolkit.topicmod.tm_gensim.AVAILABLE_METRICS`.
See :mod:`tmtoolkit.topicmod.evaluate` for references and implementations of those evaluation metrics.
"""

import logging

import numpy as np

from tmtoolkit.topicmod.parallel import MultiprocModelsRunner, MultiprocModelsWorkerABC, MultiprocEvaluationRunner, \
    MultiprocEvaluationWorkerABC
from tmtoolkit.bow.dtm import dtm_to_gensim_corpus, gensim_corpus_to_dtm
from .evaluate import metric_cao_juan_2009, metric_arun_2010, metric_coherence_mimno_2011, metric_coherence_gensim, \
    metric_deveaud_2014

#: Available metrics for Gensim.
AVAILABLE_METRICS = (
    'perplexity',
    'cao_juan_2009',
    'arun_2010',
    'deveaud_2014',
    'coherence_mimno_2011',
    'coherence_gensim_u_mass',     # same as coherence_mimno_2011
    'coherence_gensim_c_v',
    'coherence_gensim_c_uci',
    'coherence_gensim_c_npmi',
)

#: Metrics used by default.
DEFAULT_METRICS = (
    'perplexity',
    'cao_juan_2009',
    'coherence_mimno_2011',
    'coherence_gensim_c_v'
)


logger = logging.getLogger('tmtoolkit')


#%% Specialized classes for parallel processing


class MultiprocModelsWorkerGensim(MultiprocModelsWorkerABC):
    """
    Specialized parallel model computations worker for Gensim.
    """

    package_name = 'gensim'

    def fit_model(self, data, params, return_data=False):
        """
        Fit model to `data` using gensim with parameter set `params`.
        """
        from gensim.models.ldamodel import LdaModel

        dictionary = params.pop('dictionary', None)

        if hasattr(data, 'dtype') and hasattr(data, 'shape') and hasattr(data, 'transpose'):
            corpus = dtm_to_gensim_corpus(data)
            dtm = data
        else:
            if isinstance(data, tuple) and len(data) == 2:
                dictionary, corpus = data
            else:
                corpus = data
            dtm = gensim_corpus_to_dtm(corpus)

        model = LdaModel(corpus, id2word=dictionary, **params)

        if return_data:
            return model, (corpus, dtm)
        else:
            return model


class MultiprocEvaluationWorkerGensim(MultiprocEvaluationWorkerABC, MultiprocModelsWorkerGensim):
    """
    Specialized parallel model evaluations worker for Gensim.
    """

    def fit_model(self, data, params, return_data=False):
        model, (corpus, dtm) = super(MultiprocEvaluationWorkerGensim, self).fit_model(data, params, return_data=True)

        results = {}
        if self.return_models:
            results['model'] = model

        for metric in self.eval_metric:
            if metric == 'cao_juan_2009':
                res = metric_cao_juan_2009(model.state.get_lambda())
            elif metric == 'arun_2010':
                doc_topic_list = []
                for doc_topic in model.get_document_topics(corpus):
                    d = dict(doc_topic)
                    # Gensim will not output near-zero prob. topics, hence the "d.get()":
                    t = tuple(d.get(ind, 0.) for ind in range(model.num_topics))
                    doc_topic_list.append(t)

                doc_topic_distrib = np.array(doc_topic_list)
                assert doc_topic_distrib.shape == (dtm.shape[0], params['num_topics'])

                res = metric_arun_2010(model.state.get_lambda(), doc_topic_distrib, dtm.sum(axis=1))
            elif metric == 'deveaud_2014':
                topic_word = model.state.get_lambda()
                default_n = min(10, topic_word.shape[1])
                res = metric_deveaud_2014(topic_word, n=self.eval_metric_options.get('deveaud_2014_n', default_n))
            elif metric == 'coherence_mimno_2011':
                topic_word = model.state.get_lambda()
                default_top_n = min(20, topic_word.shape[1])
                res = metric_coherence_mimno_2011(topic_word, dtm,
                                                  top_n=self.eval_metric_options.get(
                                                      'coherence_mimno_2011_top_n', default_top_n),
                                                  eps=self.eval_metric_options.get('coherence_mimno_2011_eps', 1),
                                                  include_prob=self.eval_metric_options.get(
                                                      'coherence_mimno_2011_include_prob', False),
                                                  normalize=self.eval_metric_options.get(
                                                      'coherence_mimno_2011_normalize', False),
                                                  return_mean=True)
            elif metric.startswith('coherence_gensim_'):
                coh_measure = metric[len('coherence_gensim_'):]
                topic_word = model.state.get_lambda()
                default_top_n = min(20, topic_word.shape[1])
                metric_kwargs = {
                    'measure': coh_measure,
                    'gensim_model': model,
                    'gensim_corpus': corpus,
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
            elif metric == 'perplexity':
                res = _get_model_perplexity(model, corpus)
            else:
                raise ValueError('metric not available: "%s"' % metric)

            logger.info('> evaluation result with metric "%s": %f' % (metric, res))
            results[metric] = res

        return results


#%% main API functions for parallel processing


def compute_models_parallel(data, varying_parameters=None, constant_parameters=None, n_max_processes=None):
    """
    Compute several topic models in parallel using the "gensim" package. Use a single or multiple document term matrices
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
    mp_models = MultiprocModelsRunner(MultiprocModelsWorkerGensim, data, varying_parameters, constant_parameters,
                                      n_max_processes=n_max_processes)

    return mp_models.run()


def evaluate_topic_models(data, varying_parameters, constant_parameters=None, n_max_processes=None, return_models=False,
                          metric=None, **metric_kwargs):
    """
    Compute several Topic Models in parallel using the "gensim" package. Calculate the models using a list of varying
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
    mp_eval = MultiprocEvaluationRunner(MultiprocEvaluationWorkerGensim, AVAILABLE_METRICS, data,
                                        varying_parameters, constant_parameters,
                                        metric=metric or DEFAULT_METRICS, metric_options=metric_kwargs,
                                        n_max_processes=n_max_processes, return_models=return_models)

    return mp_eval.run()


#%% Helper functions


def _get_model_perplexity(model, eval_corpus):
    n_words = sum(cnt for document in eval_corpus for _, cnt in document)
    bound = model.bound(eval_corpus)
    perwordbound = bound / n_words

    return np.exp2(-perwordbound)
