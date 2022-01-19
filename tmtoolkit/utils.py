"""
Misc. utility functions.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import codecs
import logging
import os
import pickle
import random
from collections import Counter
from inspect import signature
from typing import Union, List, Any, Optional, Sequence, Dict, Callable, Tuple, Iterable

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.sparse import csr_matrix

from .types import StrOrInt


#%% logging

_default_logging_hndlr: Optional[logging.Handler] = None  # default logging handler


def enable_logging(level: int = logging.INFO, fmt: str = '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                   logging_handler: Optional[logging.Handler] = None, add_logging_handler: bool = True,
                   **stream_hndlr_opts) -> None:
    """
    Enable logging for tmtoolkit package with minimum log level `level` and log message format `fmt`. By default, logs
    to stderr via ``logging.StreamHandler``. You may also pass your own log handler.

    .. seealso:: Currently, only the logging levels INFO and DEBUG are used in tmtoolkit. See the
                 `Python Logging HOWTO guide <https://docs.python.org/3/howto/logging.html>`_ for more information
                 on log levels and formats.

    :param level: minimum log level; default is INFO level
    :param fmt: log message format
    :param logging_handler: pass custom logging handler to be used instead of
    :param add_logging_handler: if True, add the logging handler to the logger
    :param stream_hndlr_opts: optional additional parameters passed to ``logging.StreamHandler``
    """

    global _default_logging_hndlr

    logger = logging.getLogger('tmtoolkit')
    logger.setLevel(level)

    if logging_handler:
        _default_logging_hndlr = logging_handler
    else:
        _default_logging_hndlr = logging.StreamHandler(**stream_hndlr_opts)

    _default_logging_hndlr.setLevel(level)

    if fmt:
        _default_logging_hndlr.setFormatter(logging.Formatter(fmt))

    if add_logging_handler:
        logger.addHandler(_default_logging_hndlr)


def set_logging_level(level: int) -> None:
    """
    Set logging level for tmtoolkit package default logging handler.

    :param level: minimum log level
    """

    logger = logging.getLogger('tmtoolkit')
    logger.setLevel(level)

    if _default_logging_hndlr:
        _default_logging_hndlr.setLevel(level)


def disable_logging() -> None:
    """
    Disable logging for tmtoolkit package.
    """
    set_logging_level(logging.WARNING)  # reset to default level

    if _default_logging_hndlr:
        logger = logging.getLogger('tmtoolkit')
        logger.removeHandler(_default_logging_hndlr)


#%% pickle / unpickle and general file handling


def pickle_data(data: Any, picklefile: str, **kwargs) -> None:
    """
    Save `data` in `picklefile` with Python's :mod:`pickle` module.

    :param data: data to store in `picklefile`
    :param picklefile: either target file path as string or file handle
    :param kwargs: further parameters passed to :func:`pickle.dump`
    """

    if isinstance(picklefile, str):
        with open(picklefile, 'wb') as f:
            pickle.dump(data, f, **kwargs)
    else:
        pickle.dump(data, picklefile, **kwargs)


def unpickle_file(picklefile: str, **kwargs) -> Any:
    """
    Load data from `picklefile` with Python's :mod:`pickle` module.

    .. warning:: Python pickle files may contain malicious code. You should only load pickle files from trusted sources.

    :param picklefile: either target file path as string or file handle
    :param kwargs: further parameters passed to :func:`pickle.load`
    :return: data stored in `picklefile`
    """

    if isinstance(picklefile, str):
        with open(picklefile, 'rb') as f:
            return pickle.load(f, **kwargs)
    else:
        return pickle.load(picklefile, **kwargs)


def path_split(path: str, base: Optional[List[str]] = None) -> List[str]:
    """
    Split path `path` into its components::

        path_split('a/simple/test.txt')
        # ['a', 'simple', 'test.txt']

    :param path: a file path
    :param base: path remainder (used for recursion)
    :return: components of the path as list
    """
    if not base:
        base = []

    if os.path.isabs(path):
        path = path[1:]

    start, end = os.path.split(path)

    if end:
        base.insert(0, end)

    if start:
        return path_split(start, base=base)
    else:
        return base


def read_text_file(fpath: str, encoding: str, read_size: int = -1, force_unix_linebreaks: bool = True) -> str:
    """
    Read the text file at path `fpath` with character encoding `encoding` and return it as string.

    :param fpath: path to file to read
    :param encoding: character encoding
    :param read_size: max. number of characters to read. -1 means read full file.
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
    :return: file content as string
    """
    with codecs.open(fpath, encoding=encoding) as f:
        contents = f.read(read_size)

        if force_unix_linebreaks:
            contents = linebreaks_win2unix(contents)

        return contents


def linebreaks_win2unix(text: str) -> str:
    """
    Convert Windows line breaks ``'\r\n'`` to Unix line breaks ``'\n'``.

    :param text: text string
    :return: text string with Unix line breaks
    """
    while '\r\n' in text:
        text = text.replace('\r\n', '\n')

    return text


#%% NumPy array/matrices related helper functions


def empty_chararray() -> np.ndarray:
    """
    Create empty NumPy character array.

    :return: empty NumPy character array
    """
    return np.array([], dtype='<U1')


def as_chararray(x: Union[np.ndarray, Sequence]) -> np.ndarray:
    """
    Convert a NumPy array or sequence `x` to a NumPy character array. If `x` is already a NumPy character array, return
    a copy of it.

    :param x: NumPy array or sequence
    :return: NumPy character array
    """
    if len(x) > 0:
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, str):
                return x.copy()
            else:
                return x.astype(str)
        elif not isinstance(x, (list, tuple)):
            x = list(x)
        return np.array(x, dtype=str)
    else:
        return empty_chararray()


def mat2d_window_from_indices(mat: np.ndarray,
                              row_indices: Optional[Union[List[int], np.ndarray]] = None,
                              col_indices: Optional[Union[List[int], np.ndarray]] = None,
                              copy=False) -> np.ndarray:
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


def combine_sparse_matrices_columnwise(matrices: Sequence,
                                       col_labels: Sequence[StrOrInt],
                                       row_labels: Sequence[str] = None,
                                       dtype: Optional[Union[str, np.dtype]] = None) \
        -> Union[Tuple[csr_matrix, np.ndarray], Tuple[csr_matrix, np.ndarray, np.ndarray]]:
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
    :param col_labels: column labels for each matrix in `matrices`; may be sequence of strings or integers
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


def dict2df(data: dict, key_name: str = 'key', value_name: str = 'value', sort: Optional[str] = None) -> pd.DataFrame:
    """
    Take a simple dictionary that maps any key to any **scalar** value and convert it to a dataframe that contains
    two columns: one for the keys and one for the respective values. Optionally sort by column `sort`.

    :param data: dictionary that maps keys to **scalar** values
    :param key_name: column name for the keys
    :param value_name: column name for the values
    :param sort: optionally sort by this column; prepend by "-" to indicate descending sorting order, e.g. "-value"
    :return: a dataframe with two columns: one for the keys named `key_name` and one for the respective values named
             `value_name`
    """

    if key_name == value_name:
        raise ValueError('`key_name` and `value_name` must differ')

    df = pd.DataFrame({key_name: data.keys(), value_name: data.values()})
    if sort is not None:
        if sort.startswith('-'):
            asc = False
            sort = sort[1:]
        else:
            asc = True
        return df.sort_values(by=sort, ascending=asc)
    else:
        return df


def flatten_list(l: Iterable[Iterable]) -> list:
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


def _merge_updatable(containers: Sequence, init_fn: Callable, safe: bool = False) -> Union[dict, set, Counter]:
    """Helper function to merge updatable container instances in `containers`."""
    merged = init_fn()
    for x in containers:
        if safe and any(k in merged for k in x):
            raise ValueError('merging these containers would overwrite already existing contents '
                             '(note: `safe` is set to True)')
        merged.update(x)
    return merged


def merge_dicts(dicts: Sequence[dict], sort_keys: bool = False, safe: bool = False) -> dict:
    """
    Merge all dictionaries in `dicts` to form a single dict.

    :param dicts: sequence of dictionaries to merge
    :param sort_keys: sort the keys in the resulting dictionary
    :param safe: if True, raise a ``ValueError`` if sets of keys in `dicts` are not disjoint, else later dicts in the
                 sequence will silently update already existing data with the same key
    :return: merged dictionary
    """
    res = _merge_updatable(dicts, dict, safe=safe)
    if sort_keys:
        return {k: res[k] for k in sorted(res.keys())}
    else:
        return res


def merge_sets(sets: Sequence[set], safe: bool = False) -> set:
    """
    Merge all sets in `sets` to form a single set.

    :param sets: sequence of sets to merge
    :param safe: if True, raise a ``ValueError`` if sets are not disjoint
    :return: merged set
    """
    return _merge_updatable(sets, set, safe=safe)


def sample_dict(d: dict, n: int) -> dict:
    """
    Return a subset of the dictionary `d` as random sample of size `n`.

    :param d: dictionary to sample
    :param n: sample size; must be positive and smaller than or equal to ``len(d)``
    :return: subset of the input dictionary
    """
    return dict(random.sample(list(zip(d.keys(), d.values())), n))


def greedy_partitioning(elems_dict: Dict[str, Union[int, float]], k: int, return_only_labels=False) \
        -> Union[List[Dict[str, Union[int, float]]], List[List[str]]]:
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
        return [list(elems_dict.keys())] if return_only_labels else elems_dict
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
        return [[x[0] for x in b] for b in bins]
    else:
        return [dict(b) for b in bins]


def argsort(seq: Sequence) -> List[int]:
    """
    Same as NumPy's :func:`numpy.argsort` but for Python sequences.

    :param seq: a sequence
    :return: indices into `seq` that sort `seq`
    """
    return sorted(range(len(seq)), key=seq.__getitem__)


def split_func_args(fn: Callable, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split keyword arguments `args` so that all function arguments for `fn` are the first element of the returned tuple
    and the rest of the arguments are the second element of the returned tuple.

    :param fn: a function
    :param args: keyword arguments dict
    :return: tuple with two dict elements: all arguments for `fn` are the first element, the rest of the arguments
             are the second element
    """
    sig = signature(fn)
    argnames = set(args.keys())
    fn_argnames = set(sig.parameters.keys()) & argnames

    return {k: v for k, v in args.items() if k in fn_argnames},\
           {k: v for k, v in args.items() if k not in fn_argnames}
