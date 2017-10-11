# -*- coding: utf-8 -*-

# cross validation after http://ellisp.github.io/blog/2017/01/05/topic-model-cv

import logging

import numpy as np
import gensim

from .common import dtm_to_gensim_corpus
from ._evaluation_common import MultiprocEvaluation, MultiprocEvaluationWorkerABC,\
    metric_cao_juan_2009, metric_arun_2010


AVAILABLE_METRICS = (
    'perplexity',
    'cross_validation',
    'cao_juan_2009',
#    'arun_2010',
)


logger = logging.getLogger('tmtoolkit')


def get_model_perplexity(model, eval_corpus):
    n_words = sum(cnt for document in eval_corpus for _, cnt in document)
    bound = model.bound(eval_corpus)
    perwordbound = bound / n_words

    return np.exp2(-perwordbound)


class MultiprocEvaluationWorkerGensim(MultiprocEvaluationWorkerABC):
    def fit_model_using_params(self, params):
        data = self.data.tocsr()

        if self.n_folds > 1:
            logger.info('fitting LDA model from package `gensim` with %d fold validation to data of shape %s'
                        ' with parameters: %s' % (self.n_folds, data.shape, params))

            perplexity_measurments = []
            for cur_fold in range(self.n_folds):
                logger.info('> fold %d/%d' % (cur_fold+1, self.n_folds))
                dtm_train = data[self.split_folds != cur_fold, :]
                dtm_valid = data[self.split_folds == cur_fold, :]
                corpus_train = dtm_to_gensim_corpus(dtm_train)
                corpus_valid = dtm_to_gensim_corpus(dtm_valid)

                train_model = gensim.models.ldamodel.LdaModel(corpus_train, **params)

                perpl_train = get_model_perplexity(train_model, corpus_train)      # evaluate "with itself"
                perpl_valid = get_model_perplexity(train_model, corpus_valid)      # evaluate with held-out data
                perpl_both = (perpl_train, perpl_valid)

                logger.info('> done fitting model. perplexity on training data: %f /'
                            ' on validation data: %f' % perpl_both)

                perplexity_measurments.append(perpl_both)

            results = perplexity_measurments
        else:
            logger.info('fitting LDA model from package `sklearn` to data of shape %s with parameters:'
                        ' %s' % (data.shape, params))

            corpus_train = dtm_to_gensim_corpus(data)
            train_model = gensim.models.ldamodel.LdaModel(corpus_train, **params)

            if self.eval_metric == 'cao_juan_2009':
                results = metric_cao_juan_2009(train_model.state.get_lambda())
            # elif self.eval_metric == 'arun_2010':  # TODO: fix this (get document topic distr. from gensim model)
            #     results = metric_arun_2010(train_model.state.get_lambda(), train_model[corpus_train], data.sum(axis=1))
            else:  # default: perplexity
                results = get_model_perplexity(train_model, corpus_train)

            logger.info('> evaluation result with metric "%s": %f' % (self.eval_metric, results))

        self.send_results(params, results)


def evaluate_topic_models(varying_parameters, constant_parameters, data, metric=None, n_workers=None, n_folds=0):
    metric = metric or AVAILABLE_METRICS[0]

    if metric not in AVAILABLE_METRICS:
        raise ValueError('`metric` must be one of: %s' % str(AVAILABLE_METRICS))

    if metric == 'cross_validation' and n_folds <= 1:
        raise ValueError('`n_folds` must be at least 2 if `metric` is set to "cross_validation"')
    elif n_folds > 1 and metric != 'cross_validation':
        raise ValueError('`metric` must be set to "cross_validation" if `n_folds` is greater than 1')

    mp_eval = MultiprocEvaluation(MultiprocEvaluationWorkerGensim, data, varying_parameters, constant_parameters,
                                  metric=metric, n_max_processes=n_workers, n_folds=n_folds)

    return mp_eval.evaluate()
