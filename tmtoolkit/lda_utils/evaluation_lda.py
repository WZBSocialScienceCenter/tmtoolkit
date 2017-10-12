# -*- coding: utf-8 -*-
import logging

from lda import LDA


from ._evaluation_common import MultiprocEvaluation, MultiprocEvaluationWorkerABC,\
    metric_griffiths_2004, metric_cao_juan_2009, metric_arun_2010


AVAILABLE_METRICS = (
    'loglikelihood',    # simply uses the last reported log likelihood
    'griffiths_2004',
    'cao_juan_2009',
    'arun_2010'
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

        results = {}
        for metric in self.eval_metric:
            metric_opt = self.eval_metric_options.get(metric, {})

            if metric == 'griffiths_2004':
                burnin = metric_opt.get('burnin', 50) // lda_instance.refresh
                if burnin >= len(lda_instance.loglikelihoods_):
                    raise ValueError('`griffiths_2004_burnin` set too high. should be less than %d'
                                     % (lda_instance.loglikelihoods_ * lda_instance.refresh))
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

        self.send_results(params, results)


def evaluate_topic_models(varying_parameters, constant_parameters, data, metric=None, n_workers=None, n_folds=0,
                          **metric_kwargs):
    mp_eval = MultiprocEvaluation(MultiprocEvaluationWorkerLDA, AVAILABLE_METRICS, data,
                                  varying_parameters, constant_parameters,
                                  metric=metric, metric_options=metric_kwargs,
                                  n_max_processes=n_workers, n_folds=n_folds)

    return mp_eval.evaluate()
