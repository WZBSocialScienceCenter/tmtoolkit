# -*- coding: utf-8 -*-
from __future__ import division

import logging
import multiprocessing as mp
import ctypes

import numpy as np


logger = logging.getLogger('tmtoolkit')


def get_split_folds_array(folds, size):
    each = int(round(size / folds))
    folds_arr = np.repeat(np.arange(0, folds), np.repeat(each, folds))

    assert len(folds_arr) >= size

    if len(folds_arr) > size:
        np.concatenate(folds_arr, np.random.randint(0, size - len(folds_arr), folds))

    assert len(folds_arr) == size
    assert min(folds_arr) == 0
    assert max(folds_arr) == folds - 1

    np.random.shuffle(folds_arr)

    return folds_arr


def merge_params(varying_parameters, constant_parameters):
    merged_params = []
    for p in varying_parameters:
        m = p.copy()
        m.update(constant_parameters)
        merged_params.append(m)

    return merged_params


def prepare_shared_data(data):
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

    if hasattr(data, 'format'):  # sparse matrix
        logger.info('initializing evaluation with sparse matrix of format `%s` and shape %dx%d'
                    % (data.format, data.shape[0], data.shape[1]))

        if data.format != 'coo':
            data = data.tocoo()

        full_data_base = None
        sparse_data_base = mp.Array(arr_ctype, data.data)
        sparse_rows_base = mp.Array(ctypes.c_int, data.row)   # TODO: datatype correct?
        sparse_cols_base = mp.Array(ctypes.c_int, data.col)   # TODO: datatype correct?
    else:   # dense matrix
        logger.info('initializing evaluation with dense matrix and shape %dx%d'
                    % (data.shape[0], data.shape[1]))

        full_data_base = mp.Array(arr_ctype, data.A1 if hasattr(data, 'A1') else data.flatten())
        sparse_data_base = None
        sparse_rows_base = None
        sparse_cols_base = None

    return full_data_base, sparse_data_base, sparse_rows_base, sparse_cols_base


def start_multiproc_eval(n_workers, initializer_fn, initializer_args, map_fn, map_args):
    n_workers = n_workers or mp.cpu_count()
    logger.info('creating pool of %d worker processes' % n_workers)
    pool = mp.Pool(processes=n_workers,
                   initializer=initializer_fn,
                   initargs=initializer_args)
    logger.info('starting evaluation')
    eval_results = pool.map(map_fn, map_args)
    pool.close()
    pool.join()
    logger.info('evaluation done')

    return eval_results
