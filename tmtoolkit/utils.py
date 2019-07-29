import pickle

import numpy as np
from scipy import sparse


#%% pickle / unpickle

def pickle_data(data, picklefile):
    """Helper function to pickle `data` in `picklefile`."""
    with open(picklefile, 'wb') as f:
        pickle.dump(data, f, protocol=2)


def unpickle_file(picklefile, **kwargs):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)


#%% arg type check functions


def require_types(x, valid_types, valid_types_str=()):
    if not isinstance(x, valid_types) and not any(t in repr(type(x)) for t in valid_types_str):
        raise ValueError('requires one of those types:', str(valid_types))


def require_attrs(x, req_attrs):
    avail_attrs = dir(x)
    if any(a not in avail_attrs for a in req_attrs):
        raise ValueError('requires attributes:', str(req_attrs))


def require_listlike(x):
    require_types(x, (tuple, list), ('dict_values',))


def require_listlike_or_set(x):
    require_types(x, (set, tuple, list), ('dict_values',))


def require_dictlike(x):
    require_attrs(x, ('__len__', '__getitem__', '__setitem__', '__delitem__', '__iter__', '__contains__',
                      'items', 'keys', 'get'))


#%% NumPy array/matrices related helper functions


def empty_chararray():
    """Create empty NumPy character array"""
    return np.array([], dtype='<U1')


def mat2d_window_from_indices(mat, row_indices=None, col_indices=None, copy=False):
    """
    Select an area/"window" inside of a 2D array/matrix `mat` specified by either a sequence of
    row indices `row_indices` and/or a sequence of column indices `col_indices`.
    Returns the specified area as a *view* of the data if `copy` is False, else it will return a copy.
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
    """Bring a 1D NumPy array with at least two values in `values` to a linearly normalized range of [0, 1]."""
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

    ```
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

       A B C D
       -------
       0 0 1 3
       2 0 0 0
       2 0 1 0
       6 4 5 3
       0 1 0 2

    (where the first two rows come from m1 and the other three rows from m2)
    ```

    The resulting columns will always be sorted in ascending order.
    Additionally pass as sequence of row labels for each matrix via `row_labels`. This will also sort the rows in
    ascending order according to the row labels.

    :param matrices: Sequence of sparse matrices.
    :param col_labels: Column labels for each matrix in `matrices`. May be sequence of strings or integers.
    :param row_labels: Optional sequence of row labels for each matrix in `matrices`.
    :param dtype: Optionally specify the dtype of the resulting sparse matrix.
    :return: A tuple with: combined sparse matrix in CSR format, column labels of the matrix, optionally row labels of
             the matrix if `row_labels` is not None.
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
    Flatten a 2D sequence `l` to a 1D list that is returned.
    Although `return sum(l, [])` looks like a very nice one-liner, it turns out to be much much slower than what is
    implemented below.
    """
    flat = []
    for x in l:
        flat.extend(x)

    return flat


def merge_dict_sequences_inplace(a, b):
    require_listlike(a)
    require_listlike(b)

    if len(a) != len(b):
        raise ValueError('`a` and `b` must have the same length')

    for d_a, d_b in zip(a, b):
        d_a.update(d_b)


def greedy_partitioning(elems_dict, k, return_only_labels=False):
    """
    Implementation of greed partitioning algorithm as explained in https://stackoverflow.com/a/6670011
    for a dict `elems_dict` containing elements with label -> weight mapping. The elements are placed in
    `k` bins such that the difference of sums of weights in each bin is minimized. The algorithm
    does not always find the optimal solution.
    If `return_only_labels` is False, returns a list of `k` dicts with label -> weight mapping,
    else returns a list of `k` lists containing only the labels.
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
    """Same as NumPy argsort but for Python lists"""
    return sorted(range(len(seq)), key=seq.__getitem__)
