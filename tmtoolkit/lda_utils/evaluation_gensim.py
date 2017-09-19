# -*- coding: utf-8 -*-

# cross validation after http://ellisp.github.io/blog/2017/01/05/topic-model-cv

import logging
import multiprocessing as mp
import ctypes

import numpy as np
import gensim


EVALUATE_LAST_LOGLIK = 0.05

logger = logging.getLogger('tmtoolkit')

shared_full_data = None
shared_n_folds = None
shared_split_folds = None


def evaluate_topic_models(varying_parameters, constant_parameters, data, n_workers=None, n_folds=1):
    if not hasattr(data, 'dtype') or not hasattr(data, 'shape') or len(data.shape) != 2:
        raise ValueError('`train_data` must be a NumPy array or matrix of two dimensions')

    if data.dtype == np.int:
        arr_ctype = ctypes.c_int
    elif data.dtype == np.int32:
        arr_ctype = ctypes.c_int32
    elif data.dtype == np.int64:
        arr_ctype = ctypes.c_int64
    else:
        raise ValueError('dtype of `train_data` is not supported: `%s`' % data.dtype)

    merged_params = []
    for p in varying_parameters:
        m = p.copy()
        m.update(constant_parameters)
        merged_params.append(m)

    # TODO: the following requires a dense matrix. how to share a sparse matrix?
    shared_full_data_base = mp.Array(arr_ctype, data.A1 if hasattr(data, 'A1') else data.flatten())

    if n_folds > 1:
        split_folds = np.random.randint(0, n_folds, data.shape[0])
    else:
        split_folds = None

    pool = mp.Pool(processes=n_workers,
                   initializer=_init_shared_data,
                   initargs=(shared_full_data_base, data.shape[0], data.shape[1], n_folds, split_folds))
    eval_results = pool.map(_fit_model_using_params, merged_params)
    pool.close()
    pool.join()

    return eval_results


def get_model_perplexity(model, eval_corpus):
    n_words = sum(cnt for document in eval_corpus for _, cnt in document)
    bound = model.bound(eval_corpus)
    perwordbound = bound / n_words

    return np.exp2(-perwordbound)


def dtm_to_gensim_corpus(dtm):
    import gensim

    # DTM with documents to words sparse matrix in COO format has to be converted to transposed sparse matrix in CSC
    # format
    dtm_t = dtm.transpose()
    if hasattr(dtm_t, 'tocsc'):
        dtm_sparse = dtm_t.tocsc()
    else:
        from scipy.sparse.csc import csc_matrix
        dtm_sparse = csc_matrix(dtm_t)

    return gensim.matutils.Sparse2Corpus(dtm_sparse)


def dtm_and_vocab_to_gensim_corpus(dtm, vocab):
    corpus = dtm_to_gensim_corpus(dtm)

    # vocabulary array has to be converted to dict with index -> word mapping
    id2word = {idx: w for idx, w in enumerate(vocab)}

    return corpus, id2word


def _init_shared_data(shared_full_data_base, n_rows, n_cols, n_folds, split_folds):
    global shared_full_data, shared_n_folds, shared_split_folds
    shared_full_data = np.ctypeslib.as_array(shared_full_data_base.get_obj()).reshape(n_rows, n_cols)
    shared_n_folds = n_folds
    shared_split_folds = split_folds


def _fit_model_using_params(params):
    if shared_n_folds is not None and shared_n_folds > 1:
        logger.info('fitting LDA model with %d fold validation to data of shape %s with parameters: %s'
                    % (shared_n_folds, shared_full_data.shape, params))

        perplexity_measurments = []
        for cur_fold in range(shared_n_folds):
            logger.info('> fold %d' % cur_fold)
            dtm_train = shared_full_data[shared_split_folds != cur_fold, :]
            dtm_valid = shared_full_data[shared_split_folds == cur_fold, :]
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
        logger.info('fitting LDA model to data of shape %s with parameters: %s' % (shared_full_data.shape, params))

        corpus_train = dtm_to_gensim_corpus(shared_full_data)
        train_model = gensim.models.ldamodel.LdaModel(corpus_train, **params)

        perpl_train = get_model_perplexity(train_model, corpus_train)

        logger.info('> done fitting model. perplexity on training data: %f' % perpl_train)

        return perpl_train
