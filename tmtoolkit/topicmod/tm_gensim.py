# -*- coding: utf-8 -*-
"""
Parallel model computation and evuluation with gensim.

Markus Konrad <markus.konrad@wzb.eu>
"""
from __future__ import division
import logging

import numpy as np
import gensim

from tmtoolkit.topicmod.parallel import MultiprocModelsRunner, MultiprocModelsWorkerABC, MultiprocEvaluationRunner, \
    MultiprocEvaluationWorkerABC
from tmtoolkit.topicmod._eval_tools import dtm_to_gensim_corpus
from .evaluate import metric_cao_juan_2009, metric_coherence_mimno_2011, metric_coherence_gensim


AVAILABLE_METRICS = (
    'perplexity',
    'cao_juan_2009',
    'coherence_mimno_2011',
    'coherence_gensim_u_mass',     # same as coherence_mimno_2011
    'coherence_gensim_c_v',
    'coherence_gensim_c_uci',
    'coherence_gensim_c_npmi',
)

DEFAULT_METRICS = (
    'perplexity',
    'cao_juan_2009',
    'coherence_mimno_2011',
    'coherence_gensim_c_v'
)


logger = logging.getLogger('tmtoolkit')


#%% Specialized classes for parallel processing


class MultiprocModelsWorkerGensim(MultiprocModelsWorkerABC):
    package_name = 'gensim'

    def fit_model(self, data, params, return_data=False):
        data = dtm_to_gensim_corpus(data)
        model = gensim.models.ldamodel.LdaModel(data, **params)

        if return_data:
            return model, data
        else:
            return model


class MultiprocEvaluationWorkerGensim(MultiprocEvaluationWorkerABC, MultiprocModelsWorkerGensim):
    def fit_model(self, data, params, return_data=False):
        input_dtm = data
        model, data = super(MultiprocEvaluationWorkerGensim, self).fit_model(data, params, return_data=True)

        results = {}
        if self.return_models:
            results['model'] = model

        for metric in self.eval_metric:
            # if metric == 'cross_validation': continue

            if metric == 'cao_juan_2009':
                res = metric_cao_juan_2009(model.state.get_lambda())
            # elif metric == 'arun_2010':  # TODO: fix this (get document topic distr. from gensim model)
            #     results = metric_arun_2010(train_model.state.get_lambda(), train_model[corpus_train], data.sum(axis=1))
            elif metric == 'coherence_mimno_2011':
                topic_word = model.state.get_lambda()
                default_top_n = min(20, topic_word.shape[1])
                res = metric_coherence_mimno_2011(topic_word, input_dtm,
                                                  top_n=self.eval_metric_options.get('coherence_mimno_2011_top_n', default_top_n),
                                                  eps=self.eval_metric_options.get('coherence_mimno_2011_eps', 1e-12),
                                                  return_mean=True)
            elif metric.startswith('coherence_gensim_'):
                coh_measure = metric[len('coherence_gensim_'):]
                topic_word = model.state.get_lambda()
                default_top_n = min(20, topic_word.shape[1])
                metric_kwargs = {
                    'measure': coh_measure,
                    'gensim_model': model,
                    'gensim_corpus': data,
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
            else:  # default: perplexity
                res = _get_model_perplexity(model, data)

            logger.info('> evaluation result with metric "%s": %f' % (metric, res))
            results[metric] = res

        return results


#%% main API functions for parallel processing


def compute_models_parallel(data, varying_parameters=None, constant_parameters=None, n_max_processes=None):
    """
    Compute several Topic Models in parallel using the "gensim" package. Use a single or multiple document term matrices
    `data` and optionally a list of varying parameters `varying_parameters`. Pass parameters in `constant_parameters`
    dict to each model calculation. Use at maximum `n_max_processes` processors or use all available processors if None
    is passed.
    `data` can be either a Document-Term-Matrix (NumPy array/matrix, SciPy sparse matrix) or a dict with document ID ->
    Document-Term-Matrix mapping when calculating models for multiple corpora (named multiple documents).

    If `data` is a dict of named documents, this function will return a dict with document ID -> result list. Otherwise
    it will only return a result list. A result list always is a list containing tuples `(parameter_set, model)` where
    `parameter_set` is a dict of the used parameters.
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
    `parameter_set` is a dict of the used parameters and `eval_results` is a dict of metric names -> metric results.
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
