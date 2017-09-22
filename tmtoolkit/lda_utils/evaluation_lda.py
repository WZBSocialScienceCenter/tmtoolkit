# -*- coding: utf-8 -*-
import logging

import numpy as np
from scipy.sparse.coo import coo_matrix
from lda import LDA


from ._evaluation_common import merge_params, prepare_shared_data, start_multiproc_eval


EVALUATE_LAST_LOGLIK = 0.05

logger = logging.getLogger('tmtoolkit')

shared_full_data = None
shared_sparse_data = None
shared_sparse_rows = None
shared_sparse_cols = None


def evaluate_topic_models(varying_parameters, constant_parameters, data, n_workers=None):
    merged_params = merge_params(varying_parameters, constant_parameters)
    shared_full_data_base, shared_sparse_data_base,\
        shared_sparse_rows_base, shared_sparse_cols_base = prepare_shared_data(data)

    initializer_args = (shared_full_data_base, shared_sparse_data_base, shared_sparse_rows_base,
                        shared_sparse_cols_base, data.shape[0], data.shape[1])
    eval_results = start_multiproc_eval(n_workers, _init_shared_data, initializer_args,
                                        _fit_model_using_params, merged_params)

    return eval_results


def _init_shared_data(shared_full_data_base, shared_sparse_data_base, shared_sparse_rows_base,
                      shared_sparse_cols_base, n_rows, n_cols):
    global shared_full_data, shared_sparse_data, shared_sparse_rows, shared_sparse_cols

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


def _fit_model_using_params(params):
    if shared_full_data is not None:
        full_dtm = shared_full_data
    else:
        full_dtm = coo_matrix((shared_sparse_data, (shared_sparse_rows, shared_sparse_cols)))

    logger.info('fitting LDA model to data of shape %s with parameters: %s' % (full_dtm.shape, params))

    lda_instance = LDA(**params)
    lda_instance.fit(full_dtm)

    n_last_lls = max(int(round(EVALUATE_LAST_LOGLIK * len(lda_instance.loglikelihoods_))), 1)

    logger.info('> done fitting model. will use mean of last %d log likelihood estimations for evaluation' % n_last_lls)

    if n_last_lls > 1:
        report_ll = np.mean(lda_instance.loglikelihoods_[-n_last_lls:])
    else:
        report_ll = lda_instance.loglikelihoods_[-1]

    logger.info('> log likelihood: %f' % report_ll)

    return report_ll
