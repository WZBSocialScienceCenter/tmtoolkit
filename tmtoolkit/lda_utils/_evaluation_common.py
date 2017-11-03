# -*- coding: utf-8 -*-
from __future__ import division

import logging
import multiprocessing as mp
import atexit
import ctypes

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist
from scipy.stats import entropy


logger = logging.getLogger('tmtoolkit')


class MultiprocEvaluation(object):
    def __init__(self, worker_class, available_metrics, data, varying_parameters, constant_parameters=None,
                 metric=None, metric_options=None, n_max_processes=None, n_folds=0):
        self.tasks_queues = None
        self.results_queue = None
        self.workers = None

        n_max_processes = n_max_processes or mp.cpu_count()
        if n_max_processes < 1:
            raise ValueError('`n_max_processes` must be at least 1')

        if type(available_metrics) not in (list, tuple) or not available_metrics:
            raise ValueError('`available_metrics` must be a list or tuple with a least one element')

        if metric == 'cross_validation' and n_folds <= 1:
            raise ValueError('`n_folds` must be at least 2 if `metric` is set to "cross_validation"')
        elif n_folds > 1 and metric not in (None, 'cross_validation'):
            raise ValueError('`metric` must be set to "cross_validation" if `n_folds` is greater than 1')

        if metric is None:
            if n_folds <= 1:
                metric = available_metrics  # use all metrics
            else:
                metric = 'cross_validation'

        if metric_options is None:
            metric_options = {}

        if type(metric) not in (list, tuple):
            metric = [metric]

        if type(metric) not in (list, tuple) or not metric:
            raise ValueError('`metric` must be a list or tuple with a least one element')

        for m in metric:
            if m not in available_metrics:
                raise ValueError('invalid metric was passed: "%s". valid metrics: %s' % (m, available_metrics))

        n_varying_params = len(varying_parameters)
        if n_varying_params < 1:
            raise ValueError('`varying_parameters` must contain at least contain one value')

        self.n_workers = min(n_varying_params, n_max_processes)

        self.n_folds = max(n_folds, 0)
        if self.n_folds > 1:
            self.split_folds = get_split_folds_array(n_folds, data.shape[0])
        else:
            self.split_folds = None

        self.worker_class = worker_class

        self.varying_parameters = varying_parameters
        self.constant_parameters = constant_parameters or {}
        self.eval_metric = metric
        self.eval_metric_options = metric_options or {}

        self.sparse_data, self.sparse_row_ind, self.sparse_col_ind = self._prepare_sparse_data(data)

        logger.info('init with %d workers' % self.n_workers)

        atexit.register(self.shutdown_workers)

    def __del__(self):
        """destructor. shutdown all workers"""
        self.shutdown_workers()

    def shutdown_workers(self):
        if not self.workers:
            return

        logger.info('sending shutdown signal to workers')

        [q.put(None) for q in self.tasks_queues]   # `None` is the shutdown signal
        [q.join() for q in self.tasks_queues]

        [w.join() for w in self.workers]

        self.tasks_queues = None
        self.results_queue = None
        self.workers = None
        self.n_workers = 0

    def evaluate(self):
        self._setup_workers(self.worker_class)

        params = merge_params(self.varying_parameters, self.constant_parameters)
        n_tasks = len(params)
        logger.info('starting evaluation process with %d parameter sets and %d processes'
                    % (n_tasks, self.n_workers))

        logger.debug('distributing initial work')
        for i, p in enumerate(params[:self.n_workers]):
            logger.debug('> sending task %d/%d to worker %d' % (i+1, n_tasks, i))
            self.tasks_queues[i].put(p)

        next_p_idx = self.n_workers
        worker_results = []
        while next_p_idx < len(params):
            logger.debug('awaiting result')
            finished_worker, eval_params, eval_result = self.results_queue.get()    # blocking
            logger.debug('> got result from worker %d' % finished_worker)

            worker_results.append((eval_params, eval_result))

            logger.debug('> sending task %d/%d to worker %d' % (next_p_idx + 1, n_tasks, finished_worker))
            self.tasks_queues[finished_worker].put(params[next_p_idx])
            next_p_idx += 1

        logger.debug('awaiting final results')
        [q.join() for q in self.tasks_queues]   # block for last submitted tasks

        for _ in range(self.n_workers):
            _, eval_params, eval_result = self.results_queue.get()  # blocking
            worker_results.append((eval_params, eval_result))

        logger.info('evaluation process finished')

        self.shutdown_workers()

        return worker_results

    def _setup_workers(self, worker_class):
        self.tasks_queues = []
        self.results_queue = mp.Queue()
        self.workers = []

        for i in range(self.n_workers):
            task_q = mp.JoinableQueue()
            w = worker_class(i, self.eval_metric, self.eval_metric_options,
                             self.sparse_data, self.sparse_row_ind, self.sparse_col_ind,
                             self.n_folds, self.split_folds,
                             task_q, self.results_queue, name='MultiprocEvaluationWorker#%d' % i)
            w.start()

            self.workers.append(w)
            self.tasks_queues.append(task_q)

    def _prepare_sparse_data(self, data):
        if not hasattr(data, 'dtype') or not hasattr(data, 'shape') or len(data.shape) != 2:
            raise ValueError('`data` must be a NumPy array or matrix of two dimensions')

        if data.dtype == np.int:
            arr_ctype = ctypes.c_int
        elif data.dtype == np.int32:
            arr_ctype = ctypes.c_int32
        elif data.dtype == np.int64:
            arr_ctype = ctypes.c_int64
        else:
            raise ValueError('dtype of `data` is not supported: `%s`' % data.dtype)

        if not hasattr(data, 'format'):  # dense matrix -> convert to sparse matrix in coo format
            data = coo_matrix(data)
        elif data.format != 'coo':
            data = data.tocoo()

        sparse_data_base = mp.Array(arr_ctype, data.data)
        sparse_rows_base = mp.Array(ctypes.c_int, data.row)  # TODO: datatype correct?
        sparse_cols_base = mp.Array(ctypes.c_int, data.col)  # TODO: datatype correct?

        logger.info('initializing evaluation with sparse matrix of format `%s` and shape %dx%d'
                    % (data.format, data.shape[0], data.shape[1]))

        return sparse_data_base, sparse_rows_base, sparse_cols_base


class MultiprocEvaluationWorkerABC(mp.Process):
    def __init__(self, worker_id, eval_metric, eval_metric_options,
                 sparse_data_base, sparse_row_ind_base, sparse_col_ind_base,
                 n_folds, split_folds,
                 tasks_queue, results_queue,
                 group=None, target=None, name=None, args=(), kwargs=None):
        super(MultiprocEvaluationWorkerABC, self).__init__(group, target, name, args, kwargs or {})

        self.worker_id = worker_id
        self.eval_metric = eval_metric
        self.eval_metric_options = eval_metric_options

        sparse_data = np.ctypeslib.as_array(sparse_data_base.get_obj())
        sparse_row_ind = np.ctypeslib.as_array(sparse_row_ind_base.get_obj())
        sparse_col_ind = np.ctypeslib.as_array(sparse_col_ind_base.get_obj())
        self.data = coo_matrix((sparse_data, (sparse_row_ind, sparse_col_ind)))

        self.n_folds = n_folds
        self.split_folds = split_folds

        self.tasks_queue = tasks_queue
        self.results_queue = results_queue

    def run(self):
        logger.debug('worker `%s`: run' % self.name)

        for params in iter(self.tasks_queue.get, None):
            logger.debug('worker `%s`: received task' % self.name)

            self.fit_model_using_params(params)
            self.tasks_queue.task_done()

        logger.debug('worker `%s`: shutting down' % self.name)
        self.tasks_queue.task_done()

    def fit_model_using_params(self, params):
        raise NotImplementedError('abstract base class method `fit_model_using_params` needs to be defined')

    def send_results(self, params, results):
        self.results_queue.put((self.worker_id, params, results))




