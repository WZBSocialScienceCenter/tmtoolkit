# -*- coding: utf-8 -*-
import logging

import numpy as np
from lda import LDA


from ._evaluation_common import MultiprocEvaluation, MultiprocEvaluationWorkerABC, metric_cao_juan_2009


EVALUATE_LAST_LOGLIK = 0.05
AVAILABLE_METRICS = (
    'loglikelihood',
    'cao_juan_2009',
)

logger = logging.getLogger('tmtoolkit')


class MultiprocEvaluationWorkerLDA(MultiprocEvaluationWorkerABC):
    def fit_model_using_params(self, params):
        # if self.n_folds > 1:
        #     data = self.data.tocsr()
        #
        #     logger.info('fitting LDA model from package `lda` with %d fold validation to data of shape %s'
        #                 ' with parameters: %s' % (self.n_folds, data.shape, params))
        #
        #     perplexity_measurments = []
        #     for cur_fold in range(self.n_folds):
        #         logger.info('> fold %d/%d' % (cur_fold + 1, self.n_folds))
        #         dtm_train = data[self.split_folds != cur_fold, :]
        #         dtm_valid = data[self.split_folds == cur_fold, :]
        #
        #         lda_instance = LDA(**params)
        #         lda_instance.fit(dtm_train)
        #
        #         perpl_train = lda_instance.perplexity()           # evaluate "with itself"
        #         perpl_valid = lda_instance.perplexity(dtm_valid)  # evaluate with held-out data
        #         perpl_both = (perpl_train, perpl_valid)
        #
        #         logger.info('> done fitting model. perplexity on training data:'
        #                     ' %f / on validation data: %f' % perpl_both)
        #
        #         perplexity_measurments.append(perpl_both)
        #     results = perplexity_measurments
        # else:
        logger.info('fitting LDA model from package `lda` to data of shape %s with parameters:'
                    ' %s' % (self.data.shape, params))

        lda_instance = LDA(**params)
        lda_instance.fit(self.data)

        if self.eval_metric == 'cao_juan_2009':
            results = metric_cao_juan_2009(lda_instance.topic_word_)
        else:  # default: loglikelihood
            n_last_lls = max(int(round(EVALUATE_LAST_LOGLIK * len(lda_instance.loglikelihoods_))), 1)
            if n_last_lls > 1:
                results = np.mean(lda_instance.loglikelihoods_[-n_last_lls:])
            else:
                results = lda_instance.loglikelihoods_[-1]

        logger.info('> evaluation result with metric "%s": %f' % (self.eval_metric, results))

        self.send_results(params, results)


def evaluate_topic_models(varying_parameters, constant_parameters, data, metric=None, n_workers=None, n_folds=0):
    metric = metric or AVAILABLE_METRICS[0]

    if metric not in AVAILABLE_METRICS:
        raise ValueError('`metric` must be one of: %s' % str(AVAILABLE_METRICS))

    mp_eval = MultiprocEvaluation(MultiprocEvaluationWorkerLDA, data, varying_parameters, constant_parameters,
                                  metric=metric, n_max_processes=n_workers, n_folds=n_folds)

    return mp_eval.evaluate()
