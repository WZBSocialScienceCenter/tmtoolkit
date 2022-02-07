import logging
import math
import os.path
import string
from datetime import date

import pytest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, isspmatrix_csr

from ._testtools import strategy_dtm_small

from tmtoolkit.utils import (pickle_data, unpickle_file, flatten_list, greedy_partitioning,
                             mat2d_window_from_indices, combine_sparse_matrices_columnwise, path_split, read_text_file,
                             linebreaks_win2unix, split_func_args, empty_chararray, as_chararray, merge_dicts,
                             merge_sets, sample_dict, enable_logging, set_logging_level, disable_logging, dict2df,
                             applychain)

PRINTABLE_ASCII_CHARS = [chr(c) for c in range(32, 127)]


@pytest.mark.parametrize('level, fmt', [
    (logging.DEBUG, '%(levelname)s:%(name)s:%(message)s'),
    (logging.INFO, '%(levelname)s:%(name)s:%(message)s'),
    (logging.WARNING, '%(levelname)s:%(name)s:%(message)s'),
    (logging.INFO, '<default>'),
])
def test_enable_disable_logging(caplog, level, fmt):
    tmtk_logger = logging.getLogger('tmtoolkit')
    tmtk_logger.setLevel(logging.WARNING)      # reset to default level

    tmtk_logger.debug('test line debug 1')
    tmtk_logger.info('test line info 1')
    assert caplog.text == ''

    # pytest caplog fixture uses an extra logging handler (which is already added to the logger)
    if fmt == '<default>':
        enable_logging(level, logging_handler=caplog.handler, add_logging_handler=False)
    else:
        enable_logging(level, fmt, logging_handler=caplog.handler, add_logging_handler=False)

    tmtk_logger.debug('test line debug 2')
    if level == logging.DEBUG:
        assert caplog.text.endswith('DEBUG:tmtoolkit:test line debug 2\n')
        if fmt == '<default>':
            assert caplog.text.startswith(date.today().isoformat())
    else:
        assert caplog.text == ''

    caplog.clear()

    tmtk_logger.info('test line info 2')
    if level <= logging.INFO:
        assert caplog.text.endswith('INFO:tmtoolkit:test line info 2\n')
        if fmt == '<default>':
            assert caplog.text.startswith(date.today().isoformat())
    else:
        assert caplog.text == ''

    if level > logging.DEBUG:   # reduce logging level to DEBUG
        caplog.clear()
        set_logging_level(logging.DEBUG)
        tmtk_logger.debug('test line debug 3')
        assert caplog.text.endswith('DEBUG:tmtoolkit:test line debug 3\n')
        if fmt == '<default>':
            assert caplog.text.startswith(date.today().isoformat())

    caplog.clear()
    disable_logging()

    tmtk_logger.debug('test line debug 4')
    tmtk_logger.info('test line info 4')

    assert caplog.text == ''


def test_pickle_unpickle():
    pfile = os.path.join('tests', 'data', 'test_pickle_unpickle.pickle')
    input_data = ('foo', 123, [])
    pickle_data(input_data, pfile)

    output_data = unpickle_file(pfile)

    for i, o in zip(input_data, output_data):
        assert i == o


def test_path_split():
    assert path_split('') == []
    assert path_split('/') == []
    assert path_split('a') == ['a']
    assert path_split('/a') == ['a']
    assert path_split('/a/') == ['a']
    assert path_split('a/') == ['a']
    assert path_split('a/b') == ['a', 'b']
    assert path_split('a/b/c') == ['a', 'b', 'c']
    assert path_split('/a/b/c') == ['a', 'b', 'c']
    assert path_split('/a/b/c/') == ['a', 'b', 'c']
    assert path_split('/a/../b/c/') == ['a', '..', 'b', 'c']
    assert path_split('/a/b/c/d.txt') == ['a', 'b', 'c', 'd.txt']


def test_read_text_file():
    fpath = os.path.join('tests', 'data', 'gutenberg', 'kafka_verwandlung.txt')
    contents = read_text_file(fpath, encoding='utf-8')
    assert len(contents) > 0
    contents = read_text_file(fpath, encoding='utf-8', read_size=10)
    assert 5 <= len(contents) <= 10
    contents = read_text_file(fpath, encoding='utf-8', read_size=10, force_unix_linebreaks=False)
    assert len(contents) == 10
    contents = read_text_file(fpath, encoding='utf-8', read_size=100)
    assert 0 < len(contents) <= 100


@given(text=st.text(alphabet=list('abc \r\n'), max_size=20))
def test_linebreaks_win2unix(text):
    res = linebreaks_win2unix(text)
    assert '\r\n' not in res
    if '\r\n' in text:
        assert '\n' in res


def test_empty_chararray():
    res = empty_chararray()
    assert isinstance(res, np.ndarray)
    assert len(res) == 0
    assert res.ndim == 1
    assert np.issubdtype(res.dtype, 'str')


@given(x=st.lists(st.integers()),
       as_numpy_array=st.booleans())
def test_as_chararray(x, as_numpy_array):
    x_orig = x
    if as_numpy_array:
        x = np.array(x)

    res = as_chararray(x)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(x)
    assert res.ndim == 1
    assert np.issubdtype(res.dtype, 'str')
    assert res.tolist() == list(map(str, x_orig))


@given(data=st.dictionaries(keys=st.text(string.ascii_letters, min_size=1), values=st.integers(), max_size=10),
       key_name=st.text(string.ascii_letters, min_size=1),
       value_name=st.text(string.ascii_letters, min_size=1),
       sort=st.sampled_from([None, 'key', 'value']),
       asc=st.booleans())
def test_dict2df(data, key_name, value_name, sort, asc):
    if sort == 'key':
        sort_arg = key_name
    elif sort == 'value':
        sort_arg = value_name
    else:
        sort_arg = None

    if not asc and sort is not None:
        sort_arg = '-' + sort_arg

    if key_name == value_name:
        with pytest.raises(ValueError):
            dict2df(data, key_name, value_name, sort=sort_arg)
    else:
        res = dict2df(data, key_name, value_name, sort=sort_arg)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == len(data)
        assert res.columns.tolist() == [key_name, value_name]

        # check key - value mapping
        for k, v in data.items():
            cell = res.loc[res[key_name] == k, value_name].tolist()
            assert len(cell) == 1
            assert cell[0] == v

        # check sort
        if sort == 'key':
            assert res[key_name].tolist() == sorted(data.keys(), reverse=not asc)
        elif sort == 'value':
            assert res[value_name].tolist() == sorted(data.values(), reverse=not asc)
        else:
            assert res[key_name].tolist() == list(data.keys())
            assert res[value_name].tolist() == list(data.values())


@pytest.mark.parametrize('expected, funcs, initial_arg', [
    (None, [], 1),
    (1, [lambda x: x], 1),
    (1, [lambda x: -x, lambda x: -x], 1),
    (2.0, [lambda x: x**2, math.sqrt], 2),
    (8.0, [lambda x: x**2, math.sqrt, lambda x: x**3], 2),
])
def test_applychain(expected, funcs, initial_arg):
    if expected is None:
        with pytest.raises(ValueError):
            applychain(funcs, initial_arg)
    else:
        res = applychain(funcs, initial_arg)
        if isinstance(expected, float):
            assert math.isclose(res, expected)
        else:
            assert res == expected


@given(l=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(), min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])))
def test_flatten_list(l):
    l_ = flatten_list(l)

    assert type(l_) is list
    assert len(l_) == sum(map(len, l))


@given(
    mat=strategy_dtm_small(),
    n_row_indices=st.integers(0, 10),
    n_col_indices=st.integers(0, 10),
    copy=st.booleans()
)
def test_mat2d_window_from_indices(mat, n_row_indices, n_col_indices, copy):
    mat = np.array(mat)

    n_rows, n_cols = mat.shape

    if n_row_indices == 0:
        row_indices = None
    else:
        row_indices = np.random.choice(np.arange(n_rows), size=min(n_rows, n_row_indices), replace=False)

    if n_col_indices == 0:
        col_indices = None
    else:
        col_indices = np.random.choice(np.arange(n_cols), size=min(n_cols, n_col_indices), replace=False)

    window = mat2d_window_from_indices(mat, row_indices, col_indices, copy)

    if row_indices is None:
        asserted_y_shape = n_rows
    else:
        asserted_y_shape = len(row_indices)
    assert window.shape[0] == asserted_y_shape

    if col_indices is None:
        asserted_x_shape = n_cols
    else:
        asserted_x_shape = len(col_indices)
    assert window.shape[1] == asserted_x_shape

    if row_indices is None:
        row_indices_check = np.arange(n_rows)
    else:
        row_indices_check = row_indices

    if col_indices is None:
        col_indices_check = np.arange(n_cols)
    else:
        col_indices_check = col_indices

    for w_y, m_y in enumerate(row_indices_check):
        for w_x, m_x in enumerate(col_indices_check):
            assert window[w_y, w_x] == mat[m_y, m_x]


@given(dicts=st.lists(st.dictionaries(st.text(), st.integers())),
       sort_keys=st.booleans(),
       safe=st.booleans())
def test_merge_dicts(dicts, sort_keys, safe):
    all_keys = set()
    has_common_keys = False
    for d in dicts:
        ks = set(d.keys())
        if not has_common_keys and any(k in all_keys for k in ks):
            has_common_keys = True
        all_keys.update(ks)

    if len(dicts) > 1 and has_common_keys and safe:
        with pytest.raises(ValueError, match=r'^merging these containers would overwrite already existing contents'):
            merge_dicts(dicts, sort_keys=sort_keys, safe=safe)
    else:
        res = merge_dicts(dicts, sort_keys=sort_keys, safe=safe)
        assert isinstance(res, dict)
        n = sum(map(len, dicts))
        if has_common_keys:
            assert len(res) <= n
        else:
            assert len(res) == n
            for d in dicts:
                for k, v in d.items():
                    assert res[k] == v
        assert set(res.keys()) == all_keys
        if sort_keys:
            assert list(res.keys()) == sorted(all_keys)

@given(sets=st.lists(st.sets(st.integers())), safe=st.booleans())
def test_merge_sets(sets, safe):
    all_elems = set()
    has_common_elems = False
    for s in sets:
        if not has_common_elems and any(e in all_elems for e in s):
            has_common_elems = True
        all_elems.update(s)

    if len(sets) > 1 and has_common_elems and safe:
        with pytest.raises(ValueError, match=r'^merging these containers would overwrite already existing contents'):
            merge_sets(sets, safe=safe)
    else:
        res = merge_sets(sets, safe=safe)
        assert res == all_elems


@given(d=st.dictionaries(st.text(), st.integers()), n=st.integers())
def test_sample_dict(d, n):
    if 0 <= n <= len(d):
        res = sample_dict(d, n=n)
        assert isinstance(res, dict)
        assert len(res) == n
        assert set(res.keys()) <= set(d.keys())

        for k, v in res.items():
            assert v == d[k]
    else:
        with pytest.raises(ValueError):
            sample_dict(d, n=n)


@given(elems_dict=st.dictionaries(st.text(string.printable), st.floats(allow_nan=False, allow_infinity=False)),
       k=st.integers())
def test_greedy_partitioning(elems_dict, k):
    if k <= 0:
        with pytest.raises(ValueError):
            greedy_partitioning(elems_dict, k)
    else:
        bins = greedy_partitioning(elems_dict, k)

        if 1 < k <= len(elems_dict):
            assert k == len(bins)
        else:
            assert len(bins) == len(elems_dict)

        if k == 1:
            assert bins == elems_dict
        else:
            assert sum(len(b.keys()) for b in bins) == len(elems_dict)
            assert all((k in elems_dict.keys() for k in b.keys()) for b in bins)

            if k > len(elems_dict):
                assert all(len(b) == 1 for b in bins)


def test_combine_sparse_matrices_columnwise():
    m1 = coo_matrix(np.array([
        [1, 0, 3],
        [0, 2, 0],
    ]))
    
    cols1 = list('CAD')
    rows1 = [4, 0]   # row labels. can be integers!
    
    m2 = coo_matrix(np.array([
        [0, 0, 1, 2],
        [3, 4, 5, 6],
        [2, 1, 0, 0],
    ]))

    cols2 = list('DBCA')
    rows2 = [3, 1, 2]

    m3 = coo_matrix(np.array([
        [9, 8],
    ]))

    cols3 = list('BD')

    m4 = coo_matrix(np.array([
        [9],
        [8]
    ]))

    cols4 = list('A')

    m5 = coo_matrix((0, 0), dtype=int)

    cols5 = []

    expected_1_2 = np.array([
        [0, 0, 1, 3],
        [2, 0, 0, 0],
        [2, 0, 1, 0],
        [6, 4, 5, 3],
        [0, 1, 0, 2],
    ])

    expected_1_5 = np.array([
        [0, 0, 1, 3],
        [2, 0, 0, 0],
        [2, 0, 1, 0],
        [6, 4, 5, 3],
        [0, 1, 0, 2],
        [0, 9, 0, 8],   # 3
        [9, 0, 0, 0],   # 4
        [8, 0, 0, 0],   # 4
    ])

    expected_1_2_rows_sorted = np.array([
        [2, 0, 0, 0],
        [6, 4, 5, 3],
        [0, 1, 0, 2],
        [2, 0, 1, 0],
        [0, 0, 1, 3],
    ])

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise([], [])

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, ))

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, list('X')))

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m2, ), (cols1, cols2))

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2), [])

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2), (rows1, rows1))

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2), (rows1, [0, 0, 0, 0]))

    # matrices 1 and 2, no row re-ordering
    res, res_cols = combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2))
    
    assert isspmatrix_csr(res)
    assert res.shape == (5, 4)
    assert np.all(res.A == expected_1_2)
    assert np.array_equal(res_cols, np.array(list('ABCD')))

    # matrices 1 and 2, re-order rows
    res, res_cols, res_rows = combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2), (rows1, rows2))
    assert isspmatrix_csr(res)
    assert res.shape == (5, 4)
    assert np.all(res.A == expected_1_2_rows_sorted)
    assert np.array_equal(res_cols, np.array(list('ABCD')))
    assert np.array_equal(res_rows, np.arange(5))

    # matrices 1 to 5, no row re-ordering
    res, res_cols = combine_sparse_matrices_columnwise((m1, m2, m3, m4, m5), (cols1, cols2, cols3, cols4, cols5))

    assert isspmatrix_csr(res)
    assert np.all(res.A == expected_1_5)
    assert np.array_equal(res_cols, np.array(list('ABCD')))


@pytest.mark.parametrize('testfn, testargs, expargs1, expargs2', [
    (lambda x, y: ..., {'x': 1, 'y': 2, 'z': 3}, {'x': 1, 'y': 2}, {'z': 3}),
    (lambda: ..., {'x': 1, 'y': 2, 'z': 3}, {}, {'x': 1, 'y': 2, 'z': 3}),
    (lambda x, y, z: ..., {'x': 1, 'y': 2, 'z': 3}, {'x': 1, 'y': 2, 'z': 3}, {}),
])
def test_split_func_args(testfn, testargs, expargs1, expargs2):
    res = split_func_args(testfn, testargs)
    assert isinstance(res, tuple) and len(res) == 2
    args1, args2 = res
    assert args1 == expargs1
    assert args2 == expargs2
