# -*- coding: utf-8 -*-

# cross validation after http://ellisp.github.io/blog/2017/01/05/topic-model-cv

import logging

import numpy as np
from scipy.sparse.coo import coo_matrix
import gensim

from .common import dtm_to_gensim_corpus
from ._evaluation_common import merge_params, prepare_shared_data, start_multiproc_eval


logger = logging.getLogger('tmtoolkit')

shared_full_data = None
shared_sparse_data = None
shared_sparse_rows = None
shared_sparse_cols = None
shared_n_folds = None
shared_split_folds = None


def evaluate_topic_models(varying_parameters, constant_parameters, data, n_workers=None, n_folds=1):
    merged_params = merge_params(varying_parameters, constant_parameters)
    shared_full_data_base, shared_sparse_data_base,\
        shared_sparse_rows_base, shared_sparse_cols_base = prepare_shared_data(data)

    if n_folds > 1:
        split_folds = np.random.randint(0, n_folds, data.shape[0])
    else:
        split_folds = None

    initializer_args = (shared_full_data_base, shared_sparse_data_base, shared_sparse_rows_base,
                        shared_sparse_cols_base, data.shape[0], data.shape[1], n_folds, split_folds)
    eval_results = start_multiproc_eval(n_workers, _init_shared_data, initializer_args,
                                        _fit_model_using_params, merged_params)

    return eval_results


def get_model_perplexity(model, eval_corpus):
    n_words = sum(cnt for document in eval_corpus for _, cnt in document)
    bound = model.bound(eval_corpus)
    perwordbound = bound / n_words

    return np.exp2(-perwordbound)


def _init_shared_data(shared_full_data_base, shared_sparse_data_base, shared_sparse_rows_base,
                      shared_sparse_cols_base, n_rows, n_cols, n_folds, split_folds):
    global shared_full_data, shared_sparse_data, shared_sparse_rows, shared_sparse_cols,\
        shared_n_folds, shared_split_folds
    if shared_full_data_base is not None:
        shared_full_data = np.ctypeslib.as_array(shared_full_data_base.get_obj()).reshape(n_rows, n_cols)

        shared_sparse_data = None
        shared_sparse_rows = None
        shared_sparse_cols = None
    else:
        assert shared_sparse_data_base is not None
        shared_sparse_data = np.ctypeslib.as_array(shared_sparse_data_base.get_obj())
        assert shared_sparse_rows_base is not None
        shared_sparse_rows = np.ctypeslib.as_array(shared_sparse_rows_base.get_obj())
        assert shared_sparse_cols_base is not None
        shared_sparse_cols = np.ctypeslib.as_array(shared_sparse_cols_base.get_obj())

        shared_full_data = None
    shared_n_folds = n_folds
    shared_split_folds = split_folds


def _fit_model_using_params(params):
    if shared_full_data is not None:
        full_dtm = shared_full_data
    else:
        full_dtm = coo_matrix((shared_sparse_data, (shared_sparse_rows, shared_sparse_cols))).tocsr()

    if shared_n_folds is not None and shared_n_folds > 1:
        logger.info('fitting LDA model with %d fold validation to data of shape %s with parameters: %s'
                    % (shared_n_folds, full_dtm.shape, params))

        perplexity_measurments = []
        for cur_fold in range(shared_n_folds):
            logger.info('> fold %d/%d' % (cur_fold+1, shared_n_folds))
            dtm_train = full_dtm[shared_split_folds != cur_fold, :]
            dtm_valid = full_dtm[shared_split_folds == cur_fold, :]
            corpus_train = dtm_to_gensim_corpus(dtm_train)
            corpus_valid = dtm_to_gensim_corpus(dtm_valid)

            train_model = gensim.models.ldamodel.LdaModel(corpus_train, **params)

            perpl_train = get_model_perplexity(train_model, corpus_train)      # evaluate "with itself"
            perpl_valid = get_model_perplexity(train_model, corpus_valid)      # evaluate with held-out data
            perpl_both = (perpl_train, perpl_valid)

            logger.info('> done fitting model. perplexity on training data: %f / on validation data: %f' % perpl_both)

            perplexity_measurments.append(perpl_both)

        return perplexity_measurments
    else:
        logger.info('fitting LDA model to data of shape %s with parameters: %s' % (full_dtm.shape, params))

        corpus_train = dtm_to_gensim_corpus(full_dtm)
        train_model = gensim.models.ldamodel.LdaModel(corpus_train, **params)

        perpl_train = get_model_perplexity(train_model, corpus_train)

        logger.info('> done fitting model. perplexity on training data: %f' % perpl_train)

        return perpl_train
