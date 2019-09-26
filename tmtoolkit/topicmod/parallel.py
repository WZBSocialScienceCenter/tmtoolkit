"""
Base classes for parallel model fitting and evaluation. See the specific functions and classes in
:mod:`~tmtoolkit.topicmod.tm_gensim`, :mod:`~tmtoolkit.topicmod.tm_lda` and :mod:`~tmtoolkit.topicmod.tm_sklearn` for
parallel processing with popular topic modeling packages.

.. note:: The classes and functions in this module are only important if you want to implement your own parallel
          model computation and evaluation.
"""


import atexit
import ctypes
import itertools
import logging
import multiprocessing as mp
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix

logger = logging.getLogger('tmtoolkit')


#%% General parallel model computation


class MultiprocModelsRunner:
    """
    Runner class for distributing and managing worker processes for parallel model computation.
    """

    def __init__(self, worker_class, data, varying_parameters=None, constant_parameters=None, n_max_processes=None):
        """
        Initiate runner class with a model computation worker class `worker_class` (which should be derived from
        :class:`~tmtoolkit.topicmod.parallel.MultiprocModelsWorkerABC`). This class represents the worker
        processes and each will be instantiated with `data` and work on it with a different parameter set that can be
        passed via `varying_parameters`.

        :param worker_class: model computation worker class derived from
                             :class:`~tmtoolkit.topicmod.parallel.MultiprocModelsWorkerABC`
        :param data: the data that the workers use for computations; 2D (sparse) array/matrix or a dict with such
                     matrices; the latter allows to run all computations on different datasets at once
        :param varying_parameters: list of dicts with parameters; each parameter set will be used in a separate
                                   computation
        :param constant_parameters: dict with parameters that are the same for all parallel computations
        :param n_max_processes: maximum number of worker processes to spawn
        """

        self.tasks_queues = None
        self.results_queue = None
        self.workers = None

        n_max_processes = n_max_processes or mp.cpu_count()
        if n_max_processes < 1:
            raise ValueError('`n_max_processes` must be at least 1')

        varying_parameters = varying_parameters or []
        n_varying_params = len(varying_parameters)

        self.worker_class = worker_class

        self.varying_parameters = varying_parameters
        self.constant_parameters = constant_parameters or {}

        self.got_named_data = isinstance(data, dict)
        if self.got_named_data:
            self.data = {lbl: self._prepare_data(d) for lbl, d in data.items()}
        else:
            self.data = {None: self._prepare_data(data)}

        # number of workers: at least as much as needed for the varying params and datasets but at max. n_max_processes
        self.n_workers = min(max(1, n_varying_params) * len(self.data), n_max_processes)

        logger.info('init with %d workers' % self.n_workers)

        atexit.register(self.shutdown_workers)

    def __del__(self):
        """destructor. shutdown all workers"""
        self.shutdown_workers()

    def shutdown_workers(self):
        """Send shutdown signal to all worker processes to stop them."""
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

    def run(self):
        """
        Set up worker processes and run parallel computations. Blocks until all processes are done, then stops all
        workers and returns the results.

        :return: if passed data is 2D array, returns a list with tuples (parameter set, results); if passed data is
                 a dict of 2D arrays, returns dict with same keys as data and the respective results for each dataset
        """

        # set up worker processes
        self._setup_workers(self.worker_class)

        # merge varying and constant parameters to get parameter set for each worker
        params = _merge_params(self.varying_parameters, self.constant_parameters)
        n_params = len(params)

        # create the worker tasks: cartesian product of parameter sets and datasets (here named "docs")
        docs = list(self.data.keys())
        n_docs = len(docs)
        if n_params == 0:
            tasks = list(zip(docs, [{}] * n_docs))
        else:
            tasks = list(itertools.product(docs, params))
        n_tasks = len(tasks)

        logger.info('multiproc models: starting with %d parameter sets on %d documents (= %d tasks) and %d processes'
                    % (n_params, n_docs, n_tasks, self.n_workers))

        # distribute tasks to first "n_workers" workers
        logger.debug('distributing initial work')
        task_idx = 0
        for d, p in tasks[:self.n_workers]:
            logger.debug('> sending task %d/%d to worker %d' % (task_idx + 1, n_tasks, task_idx))
            self.tasks_queues[task_idx].put((d, p))
            task_idx += 1

        # collect results for these workers
        worker_results = []
        while task_idx < n_tasks:  # until all tasks are distributed
            logger.debug('awaiting result')
            finished_worker, w_doc, w_params, w_result = self.results_queue.get()    # blocking
            logger.debug('> got result from worker %d' % finished_worker)

            worker_results.append((w_doc, w_params, w_result))

            # the finished worker is free for a new task, send it
            d, p = tasks[task_idx]
            logger.debug('> sending task %d/%d to worker %d' % (task_idx + 1, n_tasks, finished_worker))
            self.tasks_queues[finished_worker].put((d, p))
            task_idx += 1

        # collect results for the last submitted tasks
        logger.debug('awaiting final results')
        [q.join() for q in self.tasks_queues]   # block for last submitted tasks

        for _ in range(self.n_workers):
            _, w_doc, w_params, w_result = self.results_queue.get()  # blocking
            worker_results.append((w_doc, w_params, w_result))

        logger.info('multiproc models: finished')

        # stop worker processes
        self.shutdown_workers()

        # return results
        if self.got_named_data:
            res = defaultdict(list)
            for d, p, r in worker_results:
                res[d].append((p, r))
            return res
        else:
            _, p, r = zip(*worker_results)
            return list(zip(p, r))

    def _setup_workers(self, worker_class):
        """Create worker classes, their task queues and the result queue."""
        self.tasks_queues = []
        self.results_queue = mp.Queue()
        self.workers = []

        for i in range(self.n_workers):
            task_q = mp.JoinableQueue()
            w = self._new_worker(worker_class, i, task_q, self.results_queue, self.data)
            w.start()

            self.workers.append(w)
            self.tasks_queues.append(task_q)

    def _new_worker(self, worker_class, i, task_queue, results_queue, data):
        """Initiate new worker class."""
        return worker_class(i, task_queue, results_queue, data, name='%s#%d' % (str(worker_class), i))

    @staticmethod
    def _prepare_data(data):
        """
        Prepare 2D array/matrix `data` for parallel processing by converting it to COO sparse matrix (if necessary)
        and creating shared data pointers for direct access for worker processes.
        """
        if hasattr(data, 'dtype'):
            if not hasattr(data, 'shape') or len(data.shape) != 2:
                raise ValueError('`data` must be a NumPy array/matrix or SciPy sparse matrix of two dimensions')

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
        else:
            return data


class MultiprocModelsWorkerABC(mp.Process):
    """
    Abstract base class for parallel model computations worker class.
    """

    package_name = None   # abstract. override in subclass

    def __init__(self, worker_id, tasks_queue, results_queue, data,
                 group=None, target=None, name=None, args=(), kwargs=None):
        """
        Initialize parallel model computations worker class with an ID `worker_id`, a queue to receive tasks from
        `tasks_queue`, a queue to send results to `results_queue` and the `data` to operate on.

        :param worker_id: process ID
        :param tasks_queue: queue to receive tasks from
        :param results_queue: queue to send results to
        :param data: data to operate on; a dict mapping dataset label to a dataset; can be anything but is usually a
                     tuple of shared data pointers for sparse matrix in COO format (see
                     :meth:`tmtookit.topicmod.parallel.MultiprocModelsRunner._prepare_data`)
        :param group: see Python's :class:`multiprocessing.Process` class
        :param target: see Python's :class:`multiprocessing.Process` class
        :param name: see Python's :class:`multiprocessing.Process` class
        :param args: see Python's :class:`multiprocessing.Process` class
        :param kwargs: see Python's :class:`multiprocessing.Process` class
        """
        super(MultiprocModelsWorkerABC, self).__init__(group, target, name, args, kwargs or {})

        logger.debug('worker `%s`: creating worker with ID %d' % (self.name, worker_id))
        self.worker_id = worker_id
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue

        # set up data to operate on
        self.data_per_doc = {}
        for doc_label, mem in data.items():
            if isinstance(mem, tuple) and len(mem) == 3:
                sparse_data_base, sparse_row_ind_base, sparse_col_ind_base = mem
                sparse_data = np.ctypeslib.as_array(sparse_data_base.get_obj())
                sparse_row_ind = np.ctypeslib.as_array(sparse_row_ind_base.get_obj())
                sparse_col_ind = np.ctypeslib.as_array(sparse_col_ind_base.get_obj())
                logger.debug('worker `%s`: creating sparse data matrix for document `%s`' % (self.name, doc_label))
                self.data_per_doc[doc_label] = coo_matrix((sparse_data, (sparse_row_ind, sparse_col_ind)))
            else:
                self.data_per_doc[doc_label] = mem

    def run(self):
        """
        Run the process worker: Calls :meth:`~tmtookit.topicmod.parallel.MultiprocModelsWorkerABC.fit_model` on each
        dataset and parameter set coming from the tasks queue.
        """
        logger.debug('worker `%s`: run' % self.name)

        for doc, params in iter(self.tasks_queue.get, None):
            logger.debug('worker `%s`: received task' % self.name)

            data = self.data_per_doc[doc]
            logger.info('fitting LDA model from package `%s` with parameters: %s' % (self.package_name, params))

            results = self.fit_model(data, params)
            self.send_results(doc, params, results)
            self.tasks_queue.task_done()

        logger.debug('worker `%s`: shutting down' % self.name)
        self.tasks_queue.task_done()

    def fit_model(self, data, params):
        """
        Method stub to implement actually model fitting for `data` with parameter set `params`.

        :param data: data passed to the model fitting algorithm
        :param params: parameter set dict
        :return: model fitting / evaluation results
        """
        raise NotImplementedError('abstract base class method `fit_model` needs to be defined')

    def send_results(self, doc, params, results):
        """
        Put the results into the results queue.

        :param doc: "document" / dataset label
        :param params: used parameter set
        :param results: generated results, e.g. fit model and/or evaluation results
        """
        self.results_queue.put((self.worker_id, doc, params, results))


#%% Parallel model evaluation


class MultiprocEvaluationRunner(MultiprocModelsRunner):
    """
    Specialization of :class:`~tmtookit.topicmod.parallel.MultiprocModelsRunner` for parallel model evaluations.
    """

    def __init__(self, worker_class, available_metrics, data, varying_parameters, constant_parameters=None,
                 metric=None, metric_options=None, n_max_processes=None, return_models=False):
        """
        Initialize evaluation runner.

        :param worker_class: model computation worker class derived from
                             :class:`~tmtoolkit.topicmod.parallel.MultiprocModelsWorkerABC`
        :param available_metrics: list/tuple with available metrics as strings
        :param data: the data that the workers use for computations; 2D (sparse) array/matrix
        :param varying_parameters: list of dicts with parameters; each parameter set will be used in a separate
                                   computation
        :param constant_parameters: dict with parameters that are the same for all parallel computations
        :param metric: string or list of strings; if given, use only this metric(s) for evaluation; must be subset of
                       `available_metrics`
        :param metric_options: dict mapping metric name to dict of options for that metric
        :param n_max_processes: maximum number of worker processes to spawn
        :param return_models: if True, also return the computed models in the evaluation results
        """

        if isinstance(data, dict):
            raise ValueError('`data` cannot be a dict for evaluation')

        super(MultiprocEvaluationRunner, self).__init__(worker_class, data, varying_parameters, constant_parameters,
                                                        n_max_processes)

        if len(self.varying_parameters) < 1:
            raise ValueError('`varying_parameters` must contain at least one value')

        if type(available_metrics) not in (list, tuple) or not available_metrics:
            raise ValueError('`available_metrics` must be a list or tuple with a least one element')

        metric = metric or available_metrics

        if metric_options is None:
            metric_options = {}

        if type(metric) not in (list, tuple):
            metric = [metric]

        if type(metric) not in (list, tuple) or not metric:
            raise ValueError('`metric` must be a list or tuple with a least one element')

        for m in metric:
            if m not in available_metrics:
                raise ValueError('invalid metric was passed: "%s". valid metrics: %s' % (m, available_metrics))

        self.eval_metric = metric
        self.eval_metric_options = metric_options or {}
        self.return_models = return_models

    def _new_worker(self, worker_class, i, task_queue, results_queue, data):
        """Initiate new worker class. Also pass metrics to be used during evaluation."""
        return worker_class(i, self.eval_metric, self.eval_metric_options, self.return_models,
                            task_queue, results_queue, data, name='%s#%d' % (str(worker_class), i))


class MultiprocEvaluationWorkerABC(MultiprocModelsWorkerABC):
    """
    Specialization of :class:`~tmtookit.topicmod.parallel.MultiprocModelsWorkerABC` for parallel model evaluations.
    """

    def __init__(self, worker_id,
                 eval_metric, eval_metric_options, return_models,
                 tasks_queue, results_queue, data,
                 group=None, target=None, name=None, args=(), kwargs=None):
        """
        Initialize parallel model evaluations worker class with an ID `worker_id`, a queue to receive tasks from
        `tasks_queue`, a queue to send results to `results_queue` and the `data` to operate on. Use evaluation
        metrics `eval_metric`.

        :param worker_id: process ID
        :param eval_metric: list/tuple of strings of evaluation metrics to use
        :param eval_metric_options: dict mapping metric name to dict of options for that metric
        :param tasks_queue: queue to receive tasks from
        :param results_queue: queue to send results to
        :param data: data to operate on; a dict mapping dataset label to a dataset; can be anything but is usually a
                     tuple of shared data pointers for sparse matrix in COO format (see
                     :meth:`tmtookit.topicmod.parallel.MultiprocModelsRunner._prepare_data`)
        :param group: see Python's :class:`multiprocessing.Process` class
        :param target: see Python's :class:`multiprocessing.Process` class
        :param name: see Python's :class:`multiprocessing.Process` class
        :param args: see Python's :class:`multiprocessing.Process` class
        :param kwargs: see Python's :class:`multiprocessing.Process` class
        """

        super(MultiprocEvaluationWorkerABC, self).__init__(worker_id,
                                                           tasks_queue, results_queue, data,
                                                           group, target, name, args, kwargs)
        self.eval_metric = eval_metric
        self.eval_metric_options = eval_metric_options
        self.return_models = return_models


#%% Helper functions


def _merge_params(varying_parameters, constant_parameters):
    if not varying_parameters:
        return [constant_parameters]

    merged_params = []
    for p in varying_parameters:
        m = p.copy()
        m.update(constant_parameters)
        merged_params.append(m)

    return merged_params
