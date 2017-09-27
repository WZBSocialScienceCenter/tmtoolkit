# -*- coding: utf-8 -*-
import logging

import numpy as np
from lda import LDA


from ._evaluation_common import MultiprocEvaluation, MultiprocEvaluationWorkerABC


EVALUATE_LAST_LOGLIK = 0.05

logger = logging.getLogger('tmtoolkit')


class MultiprocEvaluationWorkerLDA(MultiprocEvaluationWorkerABC):
    def fit_model_using_params(self, params):
        if self.n_folds > 1:
            logger.info('fitting LDA model from package `lda` with %d fold validation to data of shape %s'
                        ' with parameters: %s' % (self.n_folds, self.data.shape, params))

            perplexity_measurments = []
            for cur_fold in range(self.n_folds):
                logger.info('> fold %d/%d' % (cur_fold + 1, self.n_folds))
                dtm_train = self.data[self.split_folds != cur_fold, :]
                dtm_valid = self.data[self.split_folds == cur_fold, :]

                lda_instance = LDA(**params)
                lda_instance.fit(dtm_train)

                perpl_train = lda_instance.perplexity()           # evaluate "with itself"
                perpl_valid = lda_instance.perplexity(dtm_valid)  # evaluate with held-out data
                perpl_both = (perpl_train, perpl_valid)

                logger.info('> done fitting model. perplexity on training data:'
                            ' %f / on validation data: %f' % perpl_both)

                perplexity_measurments.append(perpl_both)
            results = perplexity_measurments
        else:
            logger.info('fitting LDA model from package `lda` to data of shape %s with parameters:'
                        ' %s' % (self.data.shape, params))

            lda_instance = LDA(**params)
            lda_instance.fit(self.data)

            n_last_lls = max(int(round(EVALUATE_LAST_LOGLIK * len(lda_instance.loglikelihoods_))), 1)

            logger.info('> done fitting model. will use mean of last %d'
                        ' log likelihood estimations for evaluation' % n_last_lls)

            if n_last_lls > 1:
                report_ll = np.mean(lda_instance.loglikelihoods_[-n_last_lls:])
            else:
                report_ll = lda_instance.loglikelihoods_[-1]

            logger.info('> log likelihood: %f' % report_ll)

            results = report_ll

        self.send_results(params, results)


def evaluate_topic_models(varying_parameters, constant_parameters, data, n_workers=None, n_folds=0):
    mp_eval = MultiprocEvaluation(MultiprocEvaluationWorkerLDA, data, varying_parameters, constant_parameters,
                                  n_max_processes=n_workers, n_folds=n_folds)

    return mp_eval.evaluate()
