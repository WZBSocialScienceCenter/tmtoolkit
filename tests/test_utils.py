import string

import pytest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from nltk.corpus import wordnet as wn

from tmtoolkit.utils import (pickle_data, unpickle_file, require_listlike_or_set, require_dictlike, require_types,
                             simplified_pos, flatten_list, greedy_partitioning,
                             mat2d_window_from_indices, normalize_to_unit_range, tokens2ids, ids2tokens,
                             str_multisplit, expand_compound_token,
                             remove_chars_in_tokens, create_ngrams, pos_tag_convert_penn_to_wn,
                             make_index_window_around_matches, token_match_subsequent, token_glue_subsequent)

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


def test_simplified_pos():
    assert simplified_pos('') == ''
    assert simplified_pos('N') == 'N'
    assert simplified_pos('V') == 'V'
    assert simplified_pos('ADJ') == 'ADJ'
    assert simplified_pos('ADV') == 'ADV'
    assert simplified_pos('AD') == ''
    assert simplified_pos('ADX') == ''
    assert simplified_pos('PRP') == ''
    assert simplified_pos('XYZ') == ''
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
    assert simplified_pos('FOOBAR', tagset='penn') == ''


@given(l=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(), min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])))
def test_flatten_list(l):
    l_ = flatten_list(l)

    assert type(l_) is list
    assert len(l_) == sum(map(len, l))


@given(mat=st.lists(st.integers(1, 10), min_size=2, max_size=2).flatmap(
        lambda size: st.lists(
            st.lists(
                st.integers(0, 99),
                min_size=size[0],
                max_size=size[0]
            ),
            min_size=size[1],
            max_size=size[1]
        )
    ),
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


def test_tokens2ids_lists():
    tok = [list('ABC'), list('ACAB'), list('DEA')]  # tokens2ids converts those to numpy arrays

    vocab, tokids = tokens2ids(tok)

    assert isinstance(vocab, np.ndarray)
    assert np.array_equal(vocab, np.array(list('ABCDE')))
    assert len(tokids) == 3
    assert isinstance(tokids[0], np.ndarray)
    assert np.array_equal(tokids[0], np.array([0, 1, 2]))
    assert np.array_equal(tokids[1], np.array([0, 2, 0, 1]))
    assert np.array_equal(tokids[2], np.array([3, 4, 0]))


def test_tokens2ids_nparrays():
    tok = [list('ABC'), list('ACAB'), list('DEA')]  # tokens2ids converts those to numpy arrays
    tok = list(map(np.array, tok))

    vocab, tokids = tokens2ids(tok)

    assert isinstance(vocab, np.ndarray)
    assert np.array_equal(vocab, np.array(list('ABCDE')))
    assert len(tokids) == 3
    assert isinstance(tokids[0], np.ndarray)
    assert np.array_equal(tokids[0], np.array([0, 1, 2]))
    assert np.array_equal(tokids[1], np.array([0, 2, 0, 1]))
    assert np.array_equal(tokids[2], np.array([3, 4, 0]))


@given(tok=st.lists(st.integers(0, 100), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.text(), min_size=0, max_size=size[0]),
                          min_size=0, max_size=size[1])
    )
)
def test_tokens2ids_and_ids2tokens(tok):
    tok = list(map(lambda x: np.array(x, dtype=np.str), tok))

    vocab, tokids = tokens2ids(tok)

    assert isinstance(vocab, np.ndarray)
    if tok:
        assert np.array_equal(vocab, np.unique(np.concatenate(tok)))
    else:
        assert np.array_equal(vocab, np.array([], dtype=np.str))

    assert len(tokids) == len(tok)

    tok2 = ids2tokens(vocab, tokids)
    assert len(tok2) == len(tok)

    for orig_tok, tokid, inversed_tokid_tok in zip(tok, tokids, tok2):
        assert isinstance(tokid, np.ndarray)
        assert len(tokid) == len(orig_tok)
        assert np.array_equal(orig_tok, inversed_tokid_tok)


def test_pos_tag_convert_penn_to_wn():
    assert pos_tag_convert_penn_to_wn('JJ') == wn.ADJ
    assert pos_tag_convert_penn_to_wn('RB') == wn.ADV
    assert pos_tag_convert_penn_to_wn('NN') == wn.NOUN
    assert pos_tag_convert_penn_to_wn('VB') == wn.VERB

    for tag in ('', 'invalid', None):
        assert pos_tag_convert_penn_to_wn(tag) is None


def test_str_multisplit():
    punct = list(string.punctuation)

    assert str_multisplit('Te;s,t', {';', ','}) == ['Te', 's', 't']
    assert str_multisplit('US-Student', punct) == ['US', 'Student']
    assert str_multisplit('-main_file.exe,', punct) == ['', 'main', 'file', 'exe', '']


@given(s=st.text(), split_chars=st.lists(st.characters()))
def test_str_multisplit_hypothesis(s, split_chars):
    res = str_multisplit(s, split_chars)

    assert type(res) is list

    if len(s) == 0:
        assert res == ['']

    if len(split_chars) == 0:
        assert res == [s]

    for p in res:
        assert all(c not in p for c in split_chars)

    n_asserted_parts = 0
    for c in set(split_chars):
        n_asserted_parts += s.count(c)
    assert len(res) == n_asserted_parts + 1


def test_expand_compound_token():
    assert expand_compound_token(['US-Student']) == [['US', 'Student']]
    assert expand_compound_token(['US-Student-X']) == [['US', 'StudentX']]
    assert expand_compound_token(['Student-X']) == [['StudentX']]
    assert expand_compound_token(['Do-Not-Disturb']) == [['Do', 'Not', 'Disturb']]
    assert expand_compound_token(['E-Mobility-Strategy']) == [['EMobility', 'Strategy']]

    assert expand_compound_token(['US-Student', 'Do-Not-Disturb', 'E-Mobility-Strategy'],
                                 split_on_len=None, split_on_casechange=True) == [['USStudent'],
                                                                                  ['Do', 'Not', 'Disturb'],
                                                                                  ['EMobility', 'Strategy']]

    assert expand_compound_token(['US-Student', 'Do-Not-Disturb', 'E-Mobility-Strategy'],
                                 split_on_len=2, split_on_casechange=True) == [['US', 'Student'],
                                                                               ['Do', 'Not', 'Disturb'],
                                                                               ['EMobility', 'Strategy']]

    assert expand_compound_token(['E-Mobility-Strategy'], split_on_len=1) == [['E', 'Mobility', 'Strategy']]

    assert expand_compound_token(['']) == [['']]

    assert expand_compound_token(['Te;s,t'], split_chars=[';', ','], split_on_len=1, split_on_casechange=False) \
           == expand_compound_token(['Te-s-t'], split_chars=['-'], split_on_len=1, split_on_casechange=False) \
           == [['Te', 's', 't']]


@given(s=st.text(), split_chars=st.lists(st.characters(min_codepoint=32)),
       split_on_len=st.integers(0),
       split_on_casechange=st.booleans())
def test_expand_compound_token_hypothesis(s, split_chars, split_on_len, split_on_casechange):
    if not split_on_len and not split_on_casechange:
        with pytest.raises(ValueError):
            expand_compound_token([s], split_chars, split_on_len=split_on_len, split_on_casechange=split_on_casechange)
    else:
        res = expand_compound_token([s], split_chars, split_on_len=split_on_len, split_on_casechange=split_on_casechange)

        assert type(res) is list
        assert len(res) == 1

        res = res[0]

        s_contains_split_char = any(c in s for c in split_chars)
        s_is_split_chars = all(c in split_chars for c in s)

        if not s_contains_split_char:   # nothing to split on
            assert res == [s]

        if len(s) > 0:
            assert all([p for p in res])

        if not s_is_split_chars:
            for p in res:
                assert all(c not in p for c in split_chars)


@given(tokens=st.lists(st.text()), special_chars=st.lists(st.characters()))
def test_remove_chars_in_tokens(tokens, special_chars):
    if len(special_chars) == 0:
        with pytest.raises(ValueError):
            remove_chars_in_tokens(tokens, special_chars)
    else:
        tokens_ = remove_chars_in_tokens(tokens, special_chars)
        assert len(tokens_) == len(tokens)

        for t_, t in zip(tokens_, tokens):
            assert len(t_) <= len(t)
            assert all(c not in t_ for c in special_chars)


@given(tokens=st.lists(st.text()), n=st.integers(0, 4))
def test_create_ngrams(tokens, n):
    n_tok = len(tokens)

    if n < 2:
        with pytest.raises(ValueError):
            create_ngrams(tokens, n)
    else:
        ngrams = create_ngrams(tokens, n, join=False)

        if n_tok < n:
            if n_tok == 0:
                assert ngrams == []
            else:
                assert len(ngrams) == 1
                assert ngrams == [tokens]
        else:
            assert len(ngrams) == n_tok - n + 1
            assert all(len(g) == n for g in ngrams)

            tokens_ = list(ngrams[0])
            if len(ngrams) > 1:
                tokens_ += [g[-1] for g in ngrams[1:]]
            assert tokens_ == tokens

        ngrams_joined = create_ngrams(tokens, n, join=True, join_str='')
        assert len(ngrams_joined) == len(ngrams)

        for g_joined, g_tuple in zip(ngrams_joined, ngrams):
            assert g_joined == ''.join(g_tuple)


@given(matches=st.lists(st.booleans()),
       left=st.integers(min_value=0, max_value=10),
       right=st.integers(min_value=0, max_value=10),
       remove_overlaps=st.booleans())
def test_make_index_window_around_matches_flatten(matches, left, right, remove_overlaps):
    matches = np.array(matches, dtype=np.bool)
    matches_ind = np.where(matches)[0]
    n_true = matches.sum()

    res = make_index_window_around_matches(matches, left, right, flatten=True, remove_overlaps=remove_overlaps)
    assert isinstance(res, np.ndarray)
    assert res.dtype == np.int

    assert len(res) >= n_true

    if len(res) > 0:
        assert np.min(res) >= 0
        assert np.max(res) < len(matches)

    if left == 0 and right == 0:
        assert np.array_equal(matches_ind, res)

    if remove_overlaps:
        assert np.array_equal(res, np.sort(np.unique(res)))

    for i in matches_ind:
        for x in range(i-left, i+right+1):
            if 0 <= x < len(matches):
                assert x in res


@given(matches=st.lists(st.booleans()),
       left=st.integers(min_value=0, max_value=10),
       right=st.integers(min_value=0, max_value=10))
def test_make_index_window_around_matches_not_flattened(matches, left, right):
    matches = np.array(matches, dtype=np.bool)
    matches_ind = np.where(matches)[0]
    n_true = matches.sum()

    res = make_index_window_around_matches(matches, left, right, flatten=False)
    assert isinstance(res, list)
    assert len(res) == n_true == len(matches_ind)

    for win, i in zip(res, matches_ind):
        assert win.dtype == np.int
        assert len(win) > 0
        assert np.min(win) >= 0
        assert np.max(win) < len(matches)

        i_in_win = 0
        for x in range(i-left, i+right+1):
            if 0 <= x < len(matches):
                assert x == win[i_in_win]
                i_in_win += 1


def test_token_match_subsequent():
    tok = ['green', 'test', 'emob', 'test', 'greener', 'tests', 'test', 'test']

    with pytest.raises(ValueError):
        token_match_subsequent('pattern', tok)

    with pytest.raises(ValueError):
        token_match_subsequent(['pattern'], tok)

    assert token_match_subsequent(['a', 'b'], []) == []

    assert token_match_subsequent(['foo', 'bar'], tok) == []

    res = token_match_subsequent(['green*', 'test*'], tok, match_type='glob')
    assert len(res) == 2
    assert np.array_equal(res[0], np.array([0, 1]))
    assert np.array_equal(res[1], np.array([4, 5]))

    res = token_match_subsequent(['green*', 'test*', '*'], tok, match_type='glob')
    assert len(res) == 2
    assert np.array_equal(res[0], np.array([0, 1, 2]))
    assert np.array_equal(res[1], np.array([4, 5, 6]))


@given(tokens=st.lists(st.text()), n_patterns=st.integers(0, 4))
def test_token_match_subsequent_hypothesis(tokens, n_patterns):
    tokens = np.array(tokens)

    n_patterns = min(len(tokens), n_patterns)

    pat_ind = np.arange(n_patterns)
    np.random.shuffle(pat_ind)
    patterns = list(tokens[pat_ind])

    if len(patterns) < 2:
        with pytest.raises(ValueError):
            token_match_subsequent(patterns, tokens)
    else:
        res = token_match_subsequent(patterns, tokens)

        assert isinstance(res, list)
        if len(tokens) == 0:
            assert res == []
        else:
            for ind in res:
                assert len(ind) == len(patterns)
                assert np.all(ind >= 0)
                assert np.all(ind < len(tokens))
                assert np.all(np.diff(ind) == 1)   # subsequent words
                assert np.array_equal(tokens[ind], patterns)


def test_token_glue_subsequent():
    tok = ['green', 'test', 'emob', 'test', 'greener', 'tests', 'test', 'test']

    with pytest.raises(ValueError):
        token_glue_subsequent(tok, 'invalid')

    assert token_glue_subsequent(tok, []) == tok

    matches = token_match_subsequent(['green*', 'test*'], tok, match_type='glob')
    assert token_glue_subsequent(tok, matches) == ['green_test', 'emob', 'test', 'greener_tests', 'test', 'test']

    matches = token_match_subsequent(['green*', 'test*', '*'], tok, match_type='glob')
    assert token_glue_subsequent(tok, matches) == ['green_test_emob', 'test', 'greener_tests_test', 'test']


@given(tokens=st.lists(st.text(string.printable)), n_patterns=st.integers(0, 4))
def test_token_glue_subsequent_hypothesis(tokens, n_patterns):
    tokens_arr = np.array(tokens)

    n_patterns = min(len(tokens), n_patterns)

    pat_ind = np.arange(n_patterns)
    np.random.shuffle(pat_ind)
    patterns = list(tokens_arr[pat_ind])

    if len(patterns) > 1:
        matches = token_match_subsequent(patterns, tokens)
        assert token_glue_subsequent(tokens, []) == tokens

        if len(tokens) == 0:
            assert token_glue_subsequent(tokens, matches) == []
        elif len(matches) == 0:
            assert token_glue_subsequent(tokens, matches) == tokens
        else:
            res = token_glue_subsequent(tokens, matches)
            assert isinstance(res, list)
            assert 0 < len(res) < len(tokens)

            for ind in matches:
                assert '_'.join(tokens_arr[ind]) in res
