# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals
from collections import OrderedDict, defaultdict
import itertools
import logging
import multiprocessing as mp
import atexit
import ctypes

import six
import numpy as np
import pandas as pd
from scipy.sparse.coo import coo_matrix


from ..utils import pickle_data, unpickle_file


logger = logging.getLogger('tmtoolkit')


DEFAULT_TOPIC_NAME_FMT = 'topic_{i1}'
DEFAULT_RANK_NAME_FMT = 'rank_{i1}'


def top_n_from_distribution(distrib, top_n=10, row_labels=None, col_labels=None, val_labels=None):
    """
    Get `top_n` values from LDA model's distribution `distrib` as DataFrame. Can be used for topic-word distributions
    and document-topic distributions. Set `row_labels` to a format string or a list. Set `col_labels` to a format
    string for the column names. Set `val_labels` to return value labels instead of pure values (probabilities).
    """
    if len(distrib) == 0:
        raise ValueError('`distrib` must contain values')

    if top_n < 1:
        raise ValueError('`top_n` must be at least 1')
    elif top_n > len(distrib[0]):
        raise ValueError('`top_n` cannot be larger than num. of values in `distrib` rows')

    if row_labels is None:
        row_label_fixed = 'row_{i0}'
    elif isinstance(row_labels, six.string_types):
        row_label_fixed = row_labels
    else:
        row_label_fixed = None

    if val_labels is not None and type(val_labels) in (list, tuple):
        val_labels = np.array(val_labels)

    if col_labels is None:
        columns = range(top_n)
    else:
        columns = [col_labels.format(i0=i, i1=i+1) for i in range(top_n)]

    df = pd.DataFrame(columns=columns)

    for i, row_distrib in enumerate(distrib):
        if row_label_fixed:
            row_name = row_label_fixed.format(i0=i, i1=i+1)
        else:
            row_name = row_labels[i]

        # `sorter_arr` is an array of indices that would sort another array by `row_distrib` (from low to high!)
        sorter_arr = np.argsort(row_distrib)

        if val_labels is None:
            sorted_vals = row_distrib[sorter_arr][:-(top_n + 1):-1]
        else:
            if isinstance(val_labels, six.string_types):
                sorted_vals = [val_labels.format(i0=i, i1=i+1, val=row_distrib[i]) for i in sorter_arr[::-1]][:top_n]
            else:
                # first brackets: sort vocab by `sorter_arr`
                # second brackets: slice operation that reverts ordering (:-1) and then selects only `n_top` number of
                # elements
                sorted_vals = val_labels[sorter_arr][:-(top_n + 1):-1]

        top_labels_series = pd.Series(sorted_vals, name=row_name, index=columns)

        df = df.append(top_labels_series)

    return df


def _join_value_and_label_dfs(vals, labels, top_n, val_fmt=None, row_labels=None, col_labels=None, index_name=None):
    val_fmt = val_fmt or '{lbl} ({val:.4})'
    col_labels = col_labels or DEFAULT_RANK_NAME_FMT
    index_name = index_name or 'document'

    if col_labels is None:
        columns = range(top_n)
    else:
        columns = [col_labels.format(i0=i, i1=i+1) for i in range(top_n)]

    df = pd.DataFrame(columns=columns)

    for i, (_, row) in enumerate(labels.iterrows()):
        joined = []
        for j, lbl in enumerate(row):
            val = vals.iloc[i, j]
            joined.append(val_fmt.format(lbl=lbl, val=val))

        if row_labels is not None:
            if isinstance(row_labels, six.string_types):
                row_name = row_labels.format(i0=i, i1=i+1)
            else:
                row_name = row_labels[i]
        else:
            row_name = None

        row_data = pd.Series(joined, name=row_name, index=columns)
        df = df.append(row_data)

    df.index.name = index_name

    return df


def ldamodel_top_topic_words(topic_word_distrib, vocab, top_n=10, val_fmt=None, col_labels=None, index_name=None):
    df_values = top_n_from_distribution(topic_word_distrib, top_n=top_n,
                                        row_labels=DEFAULT_TOPIC_NAME_FMT, val_labels=None)
    df_labels = top_n_from_distribution(topic_word_distrib, top_n=top_n,
                                        row_labels=DEFAULT_TOPIC_NAME_FMT, val_labels=vocab)
    return _join_value_and_label_dfs(df_values, df_labels, top_n, row_labels=DEFAULT_TOPIC_NAME_FMT,
                                     val_fmt=val_fmt, col_labels=col_labels, index_name=index_name)


def ldamodel_top_doc_topics(doc_topic_distrib, doc_labels, top_n=3, val_fmt=None, col_labels=None, index_name=None):
    df_values = top_n_from_distribution(doc_topic_distrib, top_n=top_n,
                                        row_labels=doc_labels, val_labels=None)
    df_labels = top_n_from_distribution(doc_topic_distrib, top_n=top_n,
                                        row_labels=doc_labels, val_labels=DEFAULT_TOPIC_NAME_FMT)
    return _join_value_and_label_dfs(df_values, df_labels, top_n, row_labels=doc_labels,
                                     val_fmt=val_fmt, col_labels=col_labels, index_name=index_name)


def ldamodel_full_topic_words(topic_word_distrib, vocab, fmt_rownames=DEFAULT_TOPIC_NAME_FMT):
    if fmt_rownames:
        rownames = [fmt_rownames.format(i0=i, i1=i+1) for i in range(topic_word_distrib.shape[0])]
    else:
        rownames = None

    return pd.DataFrame(topic_word_distrib, columns=vocab, index=rownames)


def ldamodel_full_doc_topics(doc_topic_distrib, doc_labels, fmt_colnames=DEFAULT_TOPIC_NAME_FMT):
    if fmt_colnames:
        colnames = [fmt_colnames.format(i0=i, i1=i+1) for i in range(doc_topic_distrib.shape[0])]
    else:
        colnames = None

    return pd.DataFrame(doc_topic_distrib, columns=colnames, index=doc_labels)


def print_ldamodel_distribution(distrib, row_labels, val_labels, top_n=10):
    """
    Print `n_top` top values from a LDA model's distribution `distrib`. Can be used for topic-word distributions and
    document-topic distributions.
    """

    df_values = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=None)
    df_labels = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=val_labels)

    for i, (ind, row) in enumerate(df_labels.iterrows()):
        print(ind)
        for j, label in enumerate(row):
            val = df_values.iloc[i, j]
            print('> #%d. %s (%f)' % (j + 1, label, val))


def print_ldamodel_topic_words(topic_word_distrib, vocab, n_top=10):
    """Print `n_top` values from a LDA model's topic-word distributions."""
    print_ldamodel_distribution(topic_word_distrib, row_labels=DEFAULT_TOPIC_NAME_FMT, val_labels=vocab,
                                top_n=n_top)


def print_ldamodel_doc_topics(doc_topic_distrib, doc_labels, n_top=3):
    """Print `n_top` values from a LDA model's document-topic distributions."""
    print_ldamodel_distribution(doc_topic_distrib, row_labels=doc_labels, val_labels=DEFAULT_TOPIC_NAME_FMT,
                                top_n=n_top)


def save_ldamodel_summary_to_excel(excel_file, topic_word_distrib, doc_topic_distrib, doc_labels, vocab,
                                   top_n_topics=10, top_n_words=10, dtm=None,
                                   rank_label_fmt=None, topic_label_fmt=None):
    rank_label_fmt = rank_label_fmt or DEFAULT_RANK_NAME_FMT
    topic_label_fmt = topic_label_fmt or DEFAULT_TOPIC_NAME_FMT
    excel_writer = pd.ExcelWriter(excel_file)
    sheets = OrderedDict()

    # doc-topic distribution sheets
    sheets['top_doc_topics_vals'] = top_n_from_distribution(doc_topic_distrib, top_n=top_n_topics,
                                                            row_labels=doc_labels,
                                                            col_labels=rank_label_fmt)
    sheets['top_doc_topics_labels'] = top_n_from_distribution(doc_topic_distrib, top_n=top_n_topics,
                                                              row_labels=doc_labels,
                                                              col_labels=rank_label_fmt,
                                                              val_labels=topic_label_fmt)
    sheets['top_doc_topics_labelled_vals'] = ldamodel_top_doc_topics(doc_topic_distrib, doc_labels, top_n=top_n_topics)

    # topic-word distribution sheets
    sheets['top_topic_word_vals'] = top_n_from_distribution(topic_word_distrib, top_n=top_n_words,
                                                            row_labels=topic_label_fmt,
                                                            col_labels=rank_label_fmt)
    sheets['top_topic_word_labels'] = top_n_from_distribution(topic_word_distrib, top_n=top_n_words,
                                                              row_labels=topic_label_fmt,
                                                              col_labels=rank_label_fmt,
                                                              val_labels=vocab)
    sheets['top_topic_words_labelled_vals'] = ldamodel_top_topic_words(topic_word_distrib, vocab, top_n=top_n_words)

    if dtm is not None:
        doc_lengths = get_doc_lengths(dtm)
        marg_topic_distr = get_marginal_topic_distrib(doc_topic_distrib, doc_lengths)
        row_names = [DEFAULT_TOPIC_NAME_FMT.format(i0=i, i1=i + 1) for i in range(len(marg_topic_distr))]
        sheets['marginal_topic_distrib'] = pd.DataFrame(marg_topic_distr, columns=['marginal_topic_distrib'],
                                                        index=row_names)

    for sh_name, sh_data in sheets.items():
        sh_data.to_excel(excel_writer, sh_name)

    excel_writer.save()

    return sheets


def save_ldamodel_to_pickle(model, vocab, doc_labels, picklefile):
    """Save a LDA model as pickle file."""
    pickle_data({'model': model, 'vocab': vocab, 'doc_labels': doc_labels}, picklefile)


def load_ldamodel_from_pickle(picklefile):
    """Load a LDA model from a pickle file."""
    data = unpickle_file(picklefile)
    return data['model'], data['vocab'], data['doc_labels']


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


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def results_by_parameter(res, param, sort_by=None, sort_desc=False,
                         crossvalid_use_measurment='validation',
                         crossvalid_reduce=False,
                         crossvalid_reduce_fn=None):
    """
    Takes a list of evaluation results `res` returned by a LDA evaluation function (a list in the form
    `[(parameter_set_1, {'<metric_name>': result_1, ...}), ..., (parameter_set_n, {'<metric_name>': result_n, ...})]`)
    and returns a list with tuple pairs using  only the parameter `param` from the parameter sets in the evaluation
    results such that the returned list is
    `[(param_1, {'<metric_name>': result_1, ...}), ..., (param_n, {'<metric_name>': result_n, ...})]`.
    Optionally order either by parameter value (`sort_by=None` - the default) or by result metric
    (`sort_by='<metric name>'`).
    """
    if len(res) == 0:
        return []

    if crossvalid_use_measurment not in ('validation', 'training'):
        raise ValueError('`crossvalid_use_measurment` must be either "validation" or "training" to use the validation '
                         'or training measurements.')

    tuples = [(p[param], r) for p, r in res]

    if type(tuples[0][1]) in (list, tuple):  # cross validation results
        if len(tuples[0][1]) < 1 or len(tuples[0][1][0]) != 2:
            raise ValueError('invalid evaluation results from cross validation passed')

        mean = lambda x: sum(x) / len(x)
        crossvalid_reduce_fn = crossvalid_reduce_fn or mean

        use_measurements_idx = 0 if crossvalid_use_measurment == 'training' else 1
        measurements = [(p, [pair[use_measurements_idx] for pair in r]) for p, r in tuples]
        measurements_reduced = [(p, crossvalid_reduce_fn(r)) for p, r in measurements]

        sort_by_idx = 0 if sort_by is None else 1
        sorted_ind = argsort(list(zip(*measurements_reduced))[sort_by_idx])
        if sort_desc:
            sorted_ind = reversed(sorted_ind)

        if crossvalid_reduce:
            measurements = measurements_reduced
    else:   # single validation results
        if len(tuples[0]) != 2:
            raise ValueError('invalid evaluation results passed')

        params, metric_results = list(zip(*tuples))
        if sort_by:
            sorted_ind = argsort([r[sort_by] for r in metric_results])
        else:
            sorted_ind = argsort(params)

        if sort_desc:
            sorted_ind = reversed(sorted_ind)

        measurements = tuples

    return [measurements[i] for i in sorted_ind]


def plot_eval_results(plt, eval_results, metric=None, normalize_y=None):
    if type(eval_results) not in (list, tuple) or not eval_results:
        raise ValueError('`eval_results` must be a list or tuple with at least one element')

    if type(eval_results[0]) not in (list, tuple) or len(eval_results[0]) != 2:
        raise ValueError('`eval_results` must be a list or tuple containing a (param, values) tuple. '
                         'Maybe `eval_results` must be converted with `results_by_parameter`.')

    if normalize_y is None:
        normalize_y = metric is None

    if metric == 'cross_validation':
        plotting_res = []
        for k, folds in eval_results:
            plotting_res.extend([(k, val, f) for f, val in enumerate(folds)])
        x, y, f = zip(*plotting_res)
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=f, alpha=0.5)
    else:
        if metric is not None and type(metric) not in (list, tuple):
            metric = [metric]
        elif metric is None:
            # remove special evaluation result 'model': the calculated model itself
            all_metrics = set(next(iter(eval_results))[1].keys()) - {'model'}
            metric = sorted(all_metrics)

        if normalize_y:
            res_per_metric = {}
            for m in metric:
                params = list(zip(*eval_results))[0]
                unnorm = np.array([metric_res[m] for _, metric_res in eval_results])
                rng = np.max(unnorm) - np.min(unnorm)
                if np.max(unnorm) < 0:
                    norm = -(np.max(unnorm) - unnorm) / rng
                else:
                    norm = (unnorm-np.min(unnorm)) / rng
                res_per_metric[m] = dict(zip(params, norm))

            eval_results_tmp = []
            for k, _ in eval_results:
                metric_res = {}
                for m in metric:
                    metric_res[m] = res_per_metric[m][k]
                eval_results_tmp.append((k, metric_res))
            eval_results = eval_results_tmp

        fig, ax = plt.subplots()
        x = list(zip(*eval_results))[0]
        for m in metric:
            y = [metric_res[m] for _, metric_res in eval_results]
            ax.plot(x, y, label=m)
        ax.legend(loc='best')


def get_doc_lengths(dtm):
    if isinstance(dtm, np.matrix):
        dtm = dtm.A
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    return np.sum(dtm, axis=1)


def get_term_frequencies(dtm):
    if isinstance(dtm, np.matrix):
        dtm = dtm.A
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    return np.sum(dtm, axis=0)


def get_marginal_topic_distrib(doc_topic_distrib, doc_lengths):
    unnorm = (doc_topic_distrib.T * doc_lengths).sum(axis=1)
    return unnorm / unnorm.sum()


def parameters_for_ldavis(topic_word_distrib, doc_topic_distrib, dtm, vocab, sort_topics=False):
    return dict(
        topic_term_dists=topic_word_distrib,
        doc_topic_dists=doc_topic_distrib,
        vocab=vocab,
        doc_lengths=get_doc_lengths(dtm),
        term_frequency=get_term_frequencies(dtm),
        sort_topics=sort_topics,
    )


def merge_params(varying_parameters, constant_parameters):
    if not varying_parameters:
        return [constant_parameters]

    merged_params = []
    for p in varying_parameters:
        m = p.copy()
        m.update(constant_parameters)
        merged_params.append(m)

    return merged_params


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


class MultiprocModelsRunner(object):
    def __init__(self, worker_class, data, varying_parameters=None, constant_parameters=None, n_max_processes=None):
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

        self.got_named_docs = isinstance(data, dict)
        if self.got_named_docs:
            self.data = {doc_label: self._prepare_sparse_data(doc_data) for doc_label, doc_data in data.items()}
        else:
            self.data = {None: self._prepare_sparse_data(data)}

        self.n_workers = min(max(1, n_varying_params) * len(self.data), n_max_processes)

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

    def run(self):
        self._setup_workers(self.worker_class)

        params = merge_params(self.varying_parameters, self.constant_parameters)
        n_params = len(params)
        docs = list(self.data.keys())
        n_docs = len(docs)
        if n_params == 0:
            tasks = list(zip(docs, [{}] * n_docs))
        else:
            tasks = list(itertools.product(docs, params))
        n_tasks = len(tasks)
        logger.info('multiproc models: starting with %d parameter sets on %d documents (= %d tasks) and %d processes'
                    % (n_params, n_docs, n_tasks, self.n_workers))

        logger.debug('distributing initial work')
        task_idx = 0
        for d, p in tasks[:self.n_workers]:
            logger.debug('> sending task %d/%d to worker %d' % (task_idx + 1, n_tasks, task_idx))
            self.tasks_queues[task_idx].put((d, p))
            task_idx += 1

        worker_results = []
        while task_idx < n_tasks:
            logger.debug('awaiting result')
            finished_worker, w_doc, w_params, w_result = self.results_queue.get()    # blocking
            logger.debug('> got result from worker %d' % finished_worker)

            worker_results.append((w_doc, w_params, w_result))

            d, p = tasks[task_idx]
            logger.debug('> sending task %d/%d to worker %d' % (task_idx + 1, n_tasks, finished_worker))
            self.tasks_queues[finished_worker].put((d, p))
            task_idx += 1

        logger.debug('awaiting final results')
        [q.join() for q in self.tasks_queues]   # block for last submitted tasks

        for _ in range(self.n_workers):
            _, w_doc, w_params, w_result = self.results_queue.get()  # blocking
            worker_results.append((w_doc, w_params, w_result))

        logger.info('multiproc models: finished')

        self.shutdown_workers()

        if self.got_named_docs:
            res = defaultdict(list)
            for d, p, r in worker_results:
                res[d].append((p, r))
            return res
        else:
            _, p, r = zip(*worker_results)
            return list(zip(p, r))

    def _setup_workers(self, worker_class):
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
        return worker_class(i, task_queue, results_queue, data, name='%s#%d' % (str(worker_class), i))

    @staticmethod
    def _prepare_sparse_data(data):
        if not hasattr(data, 'dtype') or not hasattr(data, 'shape') or len(data.shape) != 2:
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


class MultiprocModelsWorkerABC(mp.Process):
    package_name = None   # abstract. override in subclass

    def __init__(self, worker_id, tasks_queue, results_queue, data,
                 group=None, target=None, name=None, args=(), kwargs=None):
        super(MultiprocModelsWorkerABC, self).__init__(group, target, name, args, kwargs or {})

        logger.debug('worker `%s`: creating worker with ID %d' % (self.name, worker_id))
        self.worker_id = worker_id
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue

        self.data_per_doc = {}
        for doc_label, sparse_mem in data.items():
            sparse_data_base, sparse_row_ind_base, sparse_col_ind_base = sparse_mem
            sparse_data = np.ctypeslib.as_array(sparse_data_base.get_obj())
            sparse_row_ind = np.ctypeslib.as_array(sparse_row_ind_base.get_obj())
            sparse_col_ind = np.ctypeslib.as_array(sparse_col_ind_base.get_obj())
            logger.debug('worker `%s`: creating sparse data matrix for document `%s`' % (self.name, doc_label))
            self.data_per_doc[doc_label] = coo_matrix((sparse_data, (sparse_row_ind, sparse_col_ind)))

    def run(self):
        logger.debug('worker `%s`: run' % self.name)

        for doc, params in iter(self.tasks_queue.get, None):
            logger.debug('worker `%s`: received task' % self.name)

            data = self.data_per_doc[doc]
            logger.info('fitting LDA model from package `%s` to data `%s` of shape %s with parameters:'
                        ' %s' % (self.package_name, doc, data.shape, params))

            results = self.fit_model(data, params)
            self.send_results(doc, params, results)
            self.tasks_queue.task_done()

        logger.debug('worker `%s`: shutting down' % self.name)
        self.tasks_queue.task_done()

    def fit_model(self, data, params):
        raise NotImplementedError('abstract base class method `fit_model` needs to be defined')

    def send_results(self, doc, params, results):
        self.results_queue.put((self.worker_id, doc, params, results))


class MultiprocEvaluationRunner(MultiprocModelsRunner):
    def __init__(self, worker_class, available_metrics, data, varying_parameters, constant_parameters=None,
                 metric=None, metric_options=None, n_max_processes=None, return_models=False):  # , n_folds=0
        if isinstance(data, dict):
            raise ValueError('`data` cannot be a dict for evaluation')

        super(MultiprocEvaluationRunner, self).__init__(worker_class, data, varying_parameters, constant_parameters,
                                                        n_max_processes)

        if len(self.varying_parameters) < 1:
            raise ValueError('`varying_parameters` must contain at least one value')

        if type(available_metrics) not in (list, tuple) or not available_metrics:
            raise ValueError('`available_metrics` must be a list or tuple with a least one element')

        # if metric == 'cross_validation' and n_folds <= 1:
        #     raise ValueError('`n_folds` must be at least 2 if `metric` is set to "cross_validation"')
        # elif n_folds > 1 and metric not in (None, 'cross_validation'):
        #     raise ValueError('`metric` must be set to "cross_validation" if `n_folds` is greater than 1')

        # if metric is None:
        #     if n_folds <= 1:
        #         metric = available_metrics  # use all metrics
        #     else:
        #         metric = 'cross_validation'

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

        # currently not supported any more:
        # self.n_folds = max(n_folds, 0)
        # if self.n_folds > 1:
        #     self.split_folds = get_split_folds_array(n_folds, data.shape[0])
        # else:
        #     self.split_folds = None

        self.eval_metric = metric
        self.eval_metric_options = metric_options or {}
        self.return_models = return_models

    def _new_worker(self, worker_class, i, task_queue, results_queue, data):
        return worker_class(i, self.eval_metric, self.eval_metric_options, self.return_models,
                            task_queue, results_queue, data, name='%s#%d' % (str(worker_class), i))


class MultiprocEvaluationWorkerABC(MultiprocModelsWorkerABC):
    def __init__(self, worker_id,
                 eval_metric, eval_metric_options, return_models,
                 tasks_queue, results_queue, data,
                 group=None, target=None, name=None, args=(), kwargs=None):
        super(MultiprocEvaluationWorkerABC, self).__init__(worker_id,
                                                           tasks_queue, results_queue, data,
                                                           group, target, name, args, kwargs)
        self.eval_metric = eval_metric
        self.eval_metric_options = eval_metric_options
        self.return_models = return_models
