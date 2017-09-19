# -*- coding: utf-8 -*-
import logging
import multiprocessing as mp
import ctypes

import numpy as np
from lda import LDA


EVALUATE_LAST_LOGLIK = 0.05

logger = logging.getLogger('tmtoolkit')

shared_full_data = None


def evaluate_topic_models(varying_parameters, constant_parameters, train_data, n_workers=None):
    if not hasattr(train_data, 'dtype') or not hasattr(train_data, 'shape') or len(train_data.shape) != 2:
        raise ValueError('`train_data` must be a NumPy array or matrix of two dimensions')

    if train_data.dtype == np.int:
        arr_ctype = ctypes.c_int
    elif train_data.dtype == np.int32:
        arr_ctype = ctypes.c_int32
    elif train_data.dtype == np.int64:
        arr_ctype = ctypes.c_int64
    else:
        raise ValueError('dtype of `train_data` is not supported: `%s`' % train_data.dtype)

    merged_params = []
    for p in varying_parameters:
        m = p.copy()
        m.update(constant_parameters)
        merged_params.append(m)

    # TODO: the following requires a dense matrix. how to share a sparse matrix?
    # TODO: join with code from evaluation_gensim
    shared_train_data_base = mp.Array(arr_ctype, train_data.A1 if hasattr(train_data, 'A1') else train_data.flatten())

    logger.info('creating pool of %d worker processes' % n_workers or mp.cpu_count())
    pool = mp.Pool(processes=n_workers,
                   initializer=_init_shared_data,
                   initargs=(shared_train_data_base, train_data.shape[0], train_data.shape[1]))
    logger.info('starting evaluation')
    eval_results = pool.map(_fit_model_using_params, merged_params)
    pool.close()
    pool.join()
    logger.info('evaluation done')

    return eval_results


def _init_shared_data(shared_train_data_base, n_rows, n_cols):
    global shared_full_data
    shared_full_data = np.ctypeslib.as_array(shared_train_data_base.get_obj()).reshape(n_rows, n_cols)


def _fit_model_using_params(params):
    logger.info('fitting LDA model to data of shape %s with parameters: %s' % (shared_full_data.shape, params))

    lda_instance = LDA(**params)
    lda_instance.fit(shared_full_data)

    n_last_lls = max(int(round(EVALUATE_LAST_LOGLIK * len(lda_instance.loglikelihoods_))), 1)

    logger.info('> done fitting model. will use last %d log likelihood estimations for evaluation' % n_last_lls)

    if n_last_lls > 1:
        report_ll = np.mean(lda_instance.loglikelihoods_[-n_last_lls:])
    else:
        report_ll = lda_instance.loglikelihoods_[-1]

    logger.info('> log likelihood: %f' % report_ll)

    return report_ll
