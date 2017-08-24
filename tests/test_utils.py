import string
import random

import pytest
import hypothesis.strategies as st
from hypothesis import given

from tmtoolkit.utils import (pickle_data, unpickle_file, require_listlike, require_dictlike, require_types,
                             simplified_pos, apply_to_mat_column, flatten_list, tuplize, greedy_partitioning)

PRINTABLE_ASCII_CHARS = [chr(c) for c in range(32, 127)]


def test_pickle_unpickle():
    pfile = 'tests/data/test_pickle_unpickle.pickle'
    input_data = ('foo', 123, [])
    pickle_data(input_data, pfile)

    output_data = unpickle_file(pfile)

    for i, o in zip(input_data, output_data):
        assert i == o


def test_require_listlike():
    require_listlike([])
    require_listlike([123])
    require_listlike(tuple())
    require_listlike((1, 2, 3))
    require_listlike(set())
    require_listlike({1, 2, 3})

    with pytest.raises(ValueError): require_listlike({})
    with pytest.raises(ValueError): require_listlike({'x': 'y'})
    with pytest.raises(ValueError): require_listlike('a string')


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


def test_simplified_pos():
    assert simplified_pos('') is None
    assert simplified_pos('N') == 'N'
    assert simplified_pos('V') == 'V'
    assert simplified_pos('ADJ') == 'ADJ'
    assert simplified_pos('ADV') == 'ADV'
    assert simplified_pos('AD') is None
    assert simplified_pos('ADX') is None
    assert simplified_pos('PRP') is None
    assert simplified_pos('XYZ') is None
    assert simplified_pos('NN') == 'N'
    assert simplified_pos('NNP') == 'N'
    assert simplified_pos('VX') == 'V'
    assert simplified_pos('ADJY') == 'ADJ'
    assert simplified_pos('ADVZ') == 'ADV'

    assert simplified_pos('NNP', tagset='penn') == 'N'
    assert simplified_pos('VFOO', tagset='penn') == 'V'
    assert simplified_pos('JJ', tagset='penn') == 'ADJ'
    assert simplified_pos('JJX', tagset='penn') == 'ADJ'
    assert simplified_pos('RB', tagset='penn') == 'ADV'
    assert simplified_pos('RBFOO', tagset='penn') == 'ADV'


@given(l=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(), min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])))
def test_flatten_list(l):
    l_ = flatten_list(l)

    assert type(l_) is list
    assert len(l_) == sum(map(len, l))


@given(seq=st.lists(st.integers()))
def test_tuplize(seq):
    seq_ = tuplize(seq)

    for i, x in enumerate(seq_):
        assert type(x) is tuple
        assert x[0] == seq[i]


@given(mat=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(), min_size=size[0], max_size=size[0]), min_size=size[1], max_size=size[1])
), col_idx=st.integers(-1, 11))
def test_apply_to_mat_column_identity(mat, col_idx):
    identity_fn = lambda x: x

    # transform to list of tuples
    mat = [tuple(row) for row in mat]

    n_rows = len(mat)

    if n_rows > 0:   # make sure the supplied matrix is not ragged
        unique_n_cols = set(map(len, mat))
        assert len(unique_n_cols) == 1
        n_cols = unique_n_cols.pop()
    else:
        n_cols = 0

    if n_rows == 0 or (n_rows > 0 and n_cols == 0) or col_idx < 0 or col_idx >= n_cols:
        with pytest.raises(ValueError):
            apply_to_mat_column(mat, col_idx, identity_fn)
    else:
        assert _mat_equality(mat, apply_to_mat_column(mat, col_idx, identity_fn))


@given(mat=st.lists(st.integers(1, 20), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.text(PRINTABLE_ASCII_CHARS, max_size=20), min_size=size[0], max_size=size[0]), min_size=size[1], max_size=size[1])
))
def test_apply_to_mat_column_transform(mat):
    # transform to list of tuples
    mat = [tuple(row) for row in mat]

    n_rows = len(mat)

    unique_n_cols = set(map(len, mat))
    assert len(unique_n_cols) == 1
    n_cols = unique_n_cols.pop()
    col_idx = random.randrange(0, n_cols)

    mat_t = apply_to_mat_column(mat, col_idx, lambda x: x.upper())

    assert n_rows == len(mat_t)

    for orig, trans in zip(mat, mat_t):
        assert len(orig) == len(trans)
        for x, x_t in zip(orig, trans):
            assert x.upper() == x_t.upper()


@given(mat=st.lists(st.integers(1, 20), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.text(PRINTABLE_ASCII_CHARS, max_size=20), min_size=size[0], max_size=size[0]), min_size=size[1], max_size=size[1])
))
def test_apply_to_mat_column_transform_expand(mat):
    # transform to list of tuples
    mat = [tuple(row) for row in mat]

    n_rows = len(mat)

    unique_n_cols = set(map(len, mat))
    assert len(unique_n_cols) == 1
    n_cols = unique_n_cols.pop()
    col_idx = random.randrange(0, n_cols)

    mat_t = apply_to_mat_column(mat, col_idx, lambda x: (x, x.lower(), x.upper()), expand=True)

    assert n_rows == len(mat_t)

    for orig, trans in zip(mat, mat_t):
        assert len(orig) == len(trans) - 2

        before, x, after = orig[:col_idx+1], orig[col_idx], orig[col_idx+1:]
        before_t, x_t, after_t = trans[:col_idx+1], trans[col_idx:col_idx+3], trans[col_idx+3:]

        assert before == before_t
        assert after == after_t

        assert len(x_t) == 3
        assert x == x_t[0]
        assert x.lower() == x_t[1]
        assert x.upper() == x_t[2]


@given(elems_dict=st.dictionaries(st.text(string.printable), st.floats(allow_nan=False, allow_infinity=False)),
       k=st.integers())
def test_greedy_partitioning(elems_dict, k):
    if k <= 0:
        with pytest.raises(ValueError):
            greedy_partitioning(elems_dict, k)
    else:
        bins = greedy_partitioning(elems_dict, k)

        if k <= len(elems_dict):
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


def _mat_equality(a, b):
    return len(a) == len(b) and all(row_a == row_b for row_a, row_b in zip(a, b))


# @given(example_list=st.lists(st.text()), example_matches=st.lists(st.booleans()), negate=st.booleans())
# def test_filter_elements_in_dict(example_list, example_matches, negate):
#     d = {'foo': example_list}
#     matches = {'foo': example_matches}
#
#     if len(example_list) != len(example_matches):
#         with pytest.raises(ValueError):
#             filter_elements_in_dict(d, matches, negate_matches=negate)
#     else:
#         d_ = filter_elements_in_dict(d, matches, negate_matches=negate)
#         if negate:
#             n = len(example_matches) - sum(example_matches)
#         else:
#             n = sum(example_matches)
#         assert len(d_['foo']) == n
#
#
# def test_filter_elements_in_dict_differentkeys():
#     with pytest.raises(ValueError):
#         filter_elements_in_dict({'foo': []}, {'bar': []})
#     filter_elements_in_dict({'foo': []}, {'bar': []}, require_same_keys=False)
