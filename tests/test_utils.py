import string

import pytest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from scipy.sparse import coo_matrix, isspmatrix_csr

from ._testtools import strategy_dtm_small

from tmtoolkit.utils import (pickle_data, unpickle_file, require_listlike_or_set, require_dictlike, require_types,
                             flatten_list, greedy_partitioning,
                             mat2d_window_from_indices, normalize_to_unit_range, combine_sparse_matrices_columnwise,
                             merge_dict_sequences_inplace)

PRINTABLE_ASCII_CHARS = [chr(c) for c in range(32, 127)]


def test_pickle_unpickle():
    pfile = 'tests/data/test_pickle_unpickle.pickle'
    input_data = ('foo', 123, [])
    pickle_data(input_data, pfile)

    output_data = unpickle_file(pfile)

    for i, o in zip(input_data, output_data):
        assert i == o


def test_require_listlike():
    require_listlike_or_set([])
    require_listlike_or_set([123])
    require_listlike_or_set(tuple())
    require_listlike_or_set((1, 2, 3))
    require_listlike_or_set(set())
    require_listlike_or_set({1, 2, 3})

    with pytest.raises(ValueError): require_listlike_or_set({})
    with pytest.raises(ValueError): require_listlike_or_set({'x': 'y'})
    with pytest.raises(ValueError): require_listlike_or_set('a string')


def test_require_dictlike():
    from collections import  OrderedDict
    require_dictlike({})
    require_dictlike(OrderedDict())

    with pytest.raises(ValueError): require_dictlike(set())


def test_require_types():
    types = (set, tuple, list, dict)
    for t in types:
        require_types(t(), (t, ))

    types_shifted = types[1:] + types[:1]

    for t1, t2 in zip(types, types_shifted):
        with pytest.raises(ValueError): require_types(t1, (t2, ))


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


@given(values=st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False)))
def test_normalize_to_unit_range(values):
    values = np.array(values)

    if len(values) < 2:
        with pytest.raises(ValueError):
            normalize_to_unit_range(values)
    else:
        min_ = np.min(values)
        max_ = np.max(values)
        if max_ - min_ == 0:
            with pytest.raises(ValueError):
                normalize_to_unit_range(values)
        else:
            norm = normalize_to_unit_range(values)
            assert isinstance(norm, np.ndarray)
            assert norm.shape == values.shape
            assert np.isclose(np.min(norm), 0)
            assert np.isclose(np.max(norm), 1)


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

    m5 = coo_matrix((0, 0), dtype=np.int)

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


def test_merge_dict_sequences_inplace():
    a = [{'a': [1, 2, 3], 'b': 'bla'}, {'a': [5], 'b': 'bla2'}]
    b = [{'a': [11, 12, 13], 'b': 'bla', 'x': 'new'}, {'a': [99], 'b': 'bla2', 'x': 'new2'}]

    assert merge_dict_sequences_inplace(a, b) is None

    assert a == [{'a': [11, 12, 13], 'b': 'bla', 'x': 'new'},
                 {'a': [99], 'b': 'bla2', 'x': 'new2'}]

    with pytest.raises(ValueError):
        merge_dict_sequences_inplace(a, [])

    with pytest.raises(ValueError):
        merge_dict_sequences_inplace([], b)
