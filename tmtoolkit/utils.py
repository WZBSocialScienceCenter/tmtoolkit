"""
Misc. utility functions.
"""

import pickle

import numpy as np
from scipy import sparse


#%% pickle / unpickle

def pickle_data(data, picklefile, **kwargs):
    """
    Save `data` in `picklefile` with Python's :mod:`pickle` module.

    :param data: data to store in `picklefile`
    :param picklefile: target file
    :param kwargs: further parameters passed to :func:`pickle.dump`
    """
    with open(picklefile, 'wb') as f:
        pickle.dump(data, f, **kwargs)


def unpickle_file(picklefile, **kwargs):
    """
    Load data from `picklefile` with Python's :mod:`pickle` module.

    :param picklefile: target file
    :param kwargs: further parameters passed to :func:`pickle.load`
    :return: data stored in `picklefile`
    """
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)


#%% arg type check functions


def require_types(x, valid_types, valid_types_str=(), error_msg=None):
    """
    Check if `x` is an instance of the types in `valid_types` or its type string representation is listed in
    `valid_types_str`. Raise an :exc:`ValueError` if `x` is not of the required type(s).

    :param x: variable to check
    :param valid_types: types to check against
    :param valid_types_str: optional *string* representations of types to check against
    :param error_msg: optional error message to use instead of default exception message
    """
    if not isinstance(x, valid_types) and not any(t in repr(type(x)) for t in valid_types_str):
        raise ValueError(error_msg or ('requires one of those types: %s' % str(valid_types)))


def require_attrs(x, req_attrs, error_msg=None):
    """
    Check if `x` has all attributes listed in `req_attrs`. Raise an :exc:`ValueError` if `x` check fails.

    :param x: variable to check
    :param req_attrs: required attributes as sequence of strings
    :param error_msg: optional error message to use instead of default exception message
    """
    avail_attrs = dir(x)
    if any(a not in avail_attrs for a in req_attrs):
        raise ValueError(error_msg or ('requires attributes: %s' % str(req_attrs)))


def require_listlike(x):
    """
    Check if `x` is a list, tuple or dict values sequence.

    :param x: variable to check
    """
    require_types(x, (tuple, list), ('dict_values',), error_msg='the argument must be list- or tuple-like sequence')


def require_listlike_or_set(x):
    """
    Check if `x` is a list, tuple, dict values sequence or set.

    :param x: variable to check
    """
    require_types(x, (set, tuple, list), ('dict_values',),
                  error_msg='the argument must be list- or tuple-like sequence or a set')


def require_dictlike(x):
    """
    Check if `x` has all attributes implemented that make it a dict-like data structure.

    :param x: variable to check
    """
    require_attrs(x, ('__len__', '__getitem__', '__setitem__', '__delitem__', '__iter__', '__contains__',
                      'items', 'keys', 'get'),
                  error_msg='the argument must be a dict-like data structure')


#%% NumPy array/matrices related helper functions


def empty_chararray():
    """
    Create empty NumPy character array.

    :return: empty NumPy character array
    """
    return np.array([], dtype='<U1')


def mat2d_window_from_indices(mat, row_indices=None, col_indices=None, copy=False):
    """
    Select an area/"window" inside of a 2D array/matrix `mat` specified by either a sequence of
    row indices `row_indices` and/or a sequence of column indices `col_indices`.
    Returns the specified area as a *view* of the data if `copy` is False, else it will return a copy.

    :param mat: a 2D NumPy array
    :param row_indices: list or array of row indices to select or ``None`` to select all rows
    :param col_indices: list or array of column indices to select or ``None`` to select all columns
    :param copy: if True, return result as copy, else as view into `mat`
    :return: window into `mat` as specified by the passed indices
    """
    if not isinstance(mat, np.ndarray) or mat.ndim != 2:
        raise ValueError('`mat` must be a 2D NumPy array')

    if mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError('invalid shape for `mat`: %s' % str(mat.shape))

    if row_indices is None:
        row_indices = slice(None)   # a ":" slice
    elif len(row_indices) == 0:
        raise ValueError('`row_indices` must be non-empty')

    if col_indices is None:
        col_indices = slice(None)   # a ":" slice
    elif len(col_indices) == 0:
        raise ValueError('`col_indices` must be non-empty')

    view = mat[row_indices, :][:, col_indices]

    if copy:
        return view.copy()
    else:
        return view


def normalize_to_unit_range(values):
    """
    Bring a 1D NumPy array with at least two values in `values` to a linearly normalized range of [0, 1].

    Result is ``(x - min(x)) / (max(x) - min(x))`` where ``x`` is `values`. Note that an :exc:`ValueError` is raised
    when ``max(x) - min(x)`` equals 0.

    :param values: 1D NumPy array with at least two values
    :return: `values` linearly normalized to range [0, 1]
    """
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise ValueError('`values` must be a 1D NumPy array')

    if len(values) < 2:
        raise ValueError('`values` must contain at least two values')

    min_ = np.min(values)
    max_ = np.max(values)
    range_ = max_ - min_

    if range_ == 0:
        raise ValueError('range of `values` is 0 -- cannot normalize')

    return (values - min_) / range_


def combine_sparse_matrices_columnwise(matrices, col_labels, row_labels=None, dtype=None):
    """
    Given a sequence of sparse matrices in `matrices` and their corresponding column labels in `col_labels`, stack these
    matrices in rowwise fashion by retaining the column affiliation and filling in zeros, e.g.:

    .. code-block:: text

        m1:
           C A D
           -----
           1 0 3
           0 2 0

        m2:
           D B C A
           -------
           0 0 1 2
           3 4 5 6
           2 1 0 0

    will result in:

    .. code-block:: text

       A B C D
       -------
       0 0 1 3
       2 0 0 0
       2 0 1 0
       6 4 5 3
       0 1 0 2

    (where the first two rows come from ``m1`` and the other three rows from ``m2``).

    The resulting columns will always be sorted in ascending order.

    Additionally you can pass a sequence of row labels for each matrix via `row_labels`. This will also sort the rows in
    ascending order according to the row labels.

    :param matrices: sequence of sparse matrices
    :param col_labels: solumn labels for each matrix in `matrices`; may be sequence of strings or integers
    :param row_labels: optional sequence of row labels for each matrix in `matrices`
    :param dtype: optionally specify the dtype of the resulting sparse matrix
    :return: a tuple with (1) combined sparse matrix in CSR format; (2) column labels of the matrix; (3) optionally
             row labels of the matrix if `row_labels` is not None.
    """
    if not matrices:
        raise ValueError('`matrices` cannot be empty')

    if len(matrices) != len(col_labels):
        raise ValueError('number of matrices in `matrices` must match number of elements in `col_labels`')

    if row_labels is not None and len(matrices) != len(row_labels):
        raise ValueError('number of matrices in `matrices` must match number of elements in `row_labels`')

    # generate common set of column names to be used in the combined matrix
    all_cols = set()
    for i, (mat, cols) in enumerate(zip(matrices, col_labels)):
        if len(cols) != mat.shape[1]:
            raise ValueError('number of columns in supplied matrix `matrices[{i}]` does not match number of columns '
                             'in `col_labels[{i}]`'.format(i=i))

        all_cols.update(cols)

    # generate list of row labels to be used in the combined matrix, if it is given
    if row_labels is not None:
        all_row_labels = []
        for i, (mat, rows) in enumerate(zip(matrices, row_labels)):
            if len(rows) != mat.shape[0]:
                raise ValueError('number of rows in supplied matrix `matrices[{i}]` does not match number of rows '
                                 'in `row_labels[{i}]`'.format(i=i))

            all_row_labels.extend(rows)

        if len(set(all_row_labels)) != len(all_row_labels):
            raise ValueError('there are duplicate elements in `row_labels`, which is not allowed')

        all_row_labels = np.array(all_row_labels)
    else:
        all_row_labels = None

    # sort the column names
    all_cols = np.array(sorted(all_cols))
    n_cols = len(all_cols)

    # iterate through the matrices and their corresponding column names
    parts = []
    for mat, cols in zip(matrices, col_labels):
        if mat.shape[0] == 0: continue   # skip empty matrices

        # create a partial matrix with the complete set of columns
        # use LIL format because its efficient for inserting data
        p = sparse.lil_matrix((mat.shape[0], n_cols), dtype=dtype or mat.dtype)

        # find the column indices into `p` so that the columns of `mat` are inserted at the corresponding columns in `p`
        p_col_ind = np.searchsorted(all_cols, cols)
        p[:, p_col_ind] = mat

        parts.append(p)

    # stack all partial matrices in rowwise fashion to form the result matrix
    res = sparse.vstack(parts)
    assert res.shape[0] == sum(m.shape[0] for m in matrices)
    assert res.shape[1] == n_cols

    if all_row_labels is not None:
        # additionally sort the row labels if they are given
        assert len(all_row_labels) == res.shape[0]
        res = res.tocsr()   # faster row indexing
        row_labels_sort_ind = np.argsort(all_row_labels)
        res = res[row_labels_sort_ind, :]

        return res, all_cols, all_row_labels[row_labels_sort_ind]
    else:
        return res.tocsr(), all_cols


#%% misc functions


def flatten_list(l):
    """
    Flatten a 2D sequence `l` to a 1D list and return it.

    Although ``return sum(l, [])`` looks like a very nice one-liner, it turns out to be much much slower than what is
    implemented below.

    :param l: 2D sequence, e.g. list of lists
    :return: flattened list, i.e. a 1D list that concatenates all elements from each list inside `l`
    """
    flat = []
    for x in l:
        flat.extend(x)

    return flat


def merge_dict_sequences_inplace(a, b):
    """
    Given two sequences of equal length `a` and `b`, where each sequence contains only dicts, update the dicts in
    `a` with the corresponding dict from `b`.

    `a` is updated *in place*, hence no value is returned from this function.

    :param a: a sequence of dicts where each dict will be updated
    :param b: a sequence of dicts used for updating
    """
    require_listlike(a)
    require_listlike(b)

    if len(a) != len(b):
        raise ValueError('`a` and `b` must have the same length')

    for d_a, d_b in zip(a, b):
        d_a.update(d_b)


def greedy_partitioning(elems_dict, k, return_only_labels=False):
    """
    Implementation of greed partitioning algorithm as explained `here <https://stackoverflow.com/a/6670011>`_ for a dict
    `elems_dict` containing elements with label -> weight mapping. A weight can be a number in an arbitrary range. Since
    this is used for task scheduling, you can think if it as the larger the weight, the bigger the task is.

    The elements are placed in `k` bins such that the difference of sums of weights in each bin is minimized.
    The algorithm does not always find the optimal solution.

    If `return_only_labels` is False, returns a list of `k` dicts with label -> weight mapping,
    else returns a list of `k` lists containing only the labels for the respective partitions.

    :param elems_dict: dictionary containing elements with label -> weight mapping
    :param k: number of bins
    :param return_only_labels: if True, only return the labels in each bin
    :return: list with `k` bins, where each each bin is either a dict with label -> weight mapping if
             `return_only_labels` is False or a list of labels
    """
    if k <= 0:
        raise ValueError('`k` must be at least 1')
    elif k == 1:
        return elems_dict
    elif k >= len(elems_dict):
        # if k is bigger than the number of elements, return `len(elems_dict)` bins with each
        # bin containing only a single element
        if return_only_labels:
            return [[k] for k in elems_dict.keys()]
        else:
            return [{k: v} for k, v in elems_dict.items()]

    sorted_elems = sorted(elems_dict.items(), key=lambda x: x[1], reverse=True)
    bins = [[sorted_elems.pop(0)] for _ in range(k)]
    bin_sums = [sum(x[1] for x in b) for b in bins]

    for pair in sorted_elems:
        argmin = min(enumerate(bin_sums), key=lambda x: x[1])[0]
        bins[argmin].append(pair)
        bin_sums[argmin] += pair[1]

    if return_only_labels:
        return [[x[1] for x in b] for b in bins]
    else:
        return [dict(b) for b in bins]


def argsort(seq):
    """
    Same as NumPy's :func:`numpy.argsort` but for Python sequences.

    :param seq: a sequence
    :return: indices into `seq` that sort `seq`
    """
    return sorted(range(len(seq)), key=seq.__getitem__)
