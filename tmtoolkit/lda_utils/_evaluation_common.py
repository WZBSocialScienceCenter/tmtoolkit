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
    def __init__(self, worker_class, data, varying_parameters, constant_parameters=None,
                 metric=None, metric_options=None, n_max_processes=None, n_folds=0):
        self.tasks_queues = None
        self.results_queue = None
        self.workers = None

        n_max_processes = n_max_processes or mp.cpu_count()
        if n_max_processes < 1:
            raise ValueError('`n_max_processes` must be at least 1')

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


def get_split_folds_array(folds, size):
    #each = int(round(size / folds))
    each = size // folds
    folds_arr = np.repeat(np.arange(0, folds), np.repeat(each, folds))

    assert len(folds_arr) <= size

    if len(folds_arr) < size:
        folds_arr = np.concatenate((folds_arr, np.random.randint(0, folds, size-len(folds_arr))))

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


def metric_cao_juan_2009(topic_word_distrib):
    cos_dists = 1 - pdist(topic_word_distrib, metric='cosine')
    return np.mean(cos_dists)


def metric_arun_2010(topic_word_distrib, doc_topic_distrib, doc_lengths):
    # CM1 = SVD(M1)
    cm1 = np.linalg.svd(topic_word_distrib, compute_uv=False)
    cm1 /= np.sum(cm1)  # normalize by L1 norm
    # CM2 = norm(L * M2)
    if doc_lengths.shape[0] != 1:
        doc_lengths = doc_lengths.T
    cm2 = np.array(doc_lengths * np.matrix(doc_topic_distrib))[0]
    #cm2 /= np.linalg.norm(cm2, 2)  # normalize by L2 norm
    cm2 /= np.sum(cm2)          # normalize by L1 norm

    # symmetric Kullback-Leibler divergence KL(cm1||cm2) + KL(cm2||cm1)
    # KL is called entropy in scipy
    return entropy(cm1, cm2) + entropy(cm2, cm1)
    #return np.sum(cm1*np.log(cm1/cm2)) + np.sum(cm2*np.log(cm2/cm1))


def metric_griffiths_2004(logliks):
    """
    Calculates the harmonic mean of the loglikelihood values `logliks` as in Griffiths, Steyvers 2004. Burnin values
    should already be removed from `logliks`.

    Note: requires gmpy2 package for multiple-precision arithmetic due to very large exp() values.
          see https://github.com/aleaxit/gmpy
    """

    import gmpy2

    # using median trick as in Martin Ponweiser's Diploma Thesis 2012
    ll_med = np.median(logliks)
    ps = [gmpy2.exp(ll_med - x) for x in logliks]
    ps_mean = gmpy2.mpfr(0)
    for p in ps:
        ps_mean += p / len(ps)
    return float(ll_med - gmpy2.log(ps_mean))   # after taking the log() we can use a Python float() again
