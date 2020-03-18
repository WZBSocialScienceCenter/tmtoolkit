"""
Preprocessing: Functional API tests.
"""

from collections import Counter
import math
import string
import random

from hypothesis import given, strategies as st
import pytest
import numpy as np
from spacy.tokens import Doc
from nltk.corpus import wordnet as wn
from scipy.sparse import isspmatrix_coo
import nltk

from ._testtools import strategy_texts, strategy_tokens
from ._testcorpora import corpora_sm

from tmtoolkit._pd_dt_compat import USE_DT, FRAME_TYPE, pd_dt_colnames
from tmtoolkit.utils import flatten_list
from tmtoolkit.preprocess import (DEFAULT_LANGUAGE_MODELS, init_for_language, tokenize, doc_labels, doc_lengths,
    vocabulary, vocabulary_counts, doc_frequencies, ngrams, sparse_dtm, kwic, kwic_table, glue_tokens, simplified_pos,
    tokens2ids, ids2tokens, pos_tag_convert_penn_to_wn, str_multisplit, str_shape, str_shapesplit,
    expand_compound_token, remove_chars, make_index_window_around_matches,
    token_match_subsequent, token_glue_subsequent, transform, to_lowercase, pos_tag, lemmatize, expand_compounds,
    clean_tokens, filter_tokens, filter_documents, filter_documents_by_name, filter_for_pos, filter_tokens_by_mask,
    remove_common_tokens, remove_uncommon_tokens, token_match, filter_tokens_with_kwic
)
from tmtoolkit.preprocess._common import _filtered_docs_tokens, _filtered_doc_tokens


LANGUAGE_CODES = list(sorted(DEFAULT_LANGUAGE_MODELS.keys()))


@pytest.fixture(scope='module', params=LANGUAGE_CODES)
def nlp_all(request):
    return init_for_language(request.param)


@pytest.fixture()
def tokens_en():
    _init_lang('en')
    return tokenize(corpora_sm['en'])


@pytest.fixture()
def tokens_mini():
    _init_lang('en')
    corpus = {'ny': 'I live in New York.',
              'bln': 'I am in Berlin, but my flat is in Munich.',
              'empty': ''}
    return tokenize(corpus)


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


def test_str_shape():
    assert str_shape('') == []
    assert str_shape('xxx') == [0, 0, 0]
    assert str_shape('Xxx') == [1, 0, 0]
    assert str_shape('xxX') == [0, 0, 1]
    assert str_shape('Xxx', lower=1, upper=0) == [0, 1, 1]
    assert str_shape('Xxx', lower=1, upper=0, as_str=True) == '011'
    assert str_shape('Foo', lower='x', upper='X', as_str=True) == 'Xxx'


@given(s=st.text(), lower_int=st.integers(min_value=0, max_value=9), upper_int=st.integers(min_value=0, max_value=9),
       lower=st.characters(), upper=st.characters(),
       as_str=st.booleans(), use_ints=st.booleans())
def test_str_shape_hypothesis(s, lower_int, upper_int, lower, upper, as_str, use_ints):
    if use_ints:
        l = lower_int
        u = upper_int
    else:
        l = lower
        u = upper

    res = str_shape(s, l, u, as_str)

    if as_str:
        assert isinstance(res, str)
        assert all([x in {str(l), str(u)} for x in res])
    else:
        assert isinstance(res, list)
        assert all([x in {l, u} for x in res])

    assert len(s) == len(res)


def test_str_shapesplit():
    assert str_shapesplit('') == ['']
    assert str_shapesplit('NewYork') == ['New', 'York']
    assert str_shapesplit('newYork') == ['new', 'York']
    assert str_shapesplit('newyork') == ['newyork']
    assert str_shapesplit('USflag') == ['US', 'flag']
    assert str_shapesplit('eMail') == ['eMail']
    assert str_shapesplit('foobaR') == ['foobaR']


@given(s=st.text(string.printable), precalc_shape=st.booleans(), min_len=st.integers(min_value=1, max_value=5))
def test_str_shapesplit_hypothesis(s, precalc_shape, min_len):
    if precalc_shape:
        shape = str_shape(s)
    else:
        shape = None

    res = str_shapesplit(s, shape, min_part_length=min_len)

    assert len(res) >= 1
    assert all([isinstance(x, str) for x in res])
    if len(s) >= min_len:
        assert all([min_len <= len(x) <= len(s) for x in res])
    assert ''.join(res) == s


def test_expand_compound_token():
    assert expand_compound_token('US-Student') == ['US', 'Student']
    assert expand_compound_token('US-Student-X') == ['US', 'StudentX']
    assert expand_compound_token('Camel-CamelCase') == ['Camel', 'CamelCase']
    assert expand_compound_token('Camel-CamelCase', split_on_casechange=True) == ['Camel', 'Camel', 'Case']
    assert expand_compound_token('Camel-camelCase') == ['Camel', 'camelCase']
    assert expand_compound_token('Camel-camelCase', split_on_casechange=True) == ['Camel', 'camel', 'Case']
    assert expand_compound_token('Student-X') == ['StudentX']
    assert expand_compound_token('Do-Not-Disturb') == ['Do', 'Not', 'Disturb']
    assert expand_compound_token('E-Mobility-Strategy') == ['EMobility', 'Strategy']

    for inp, expected in zip(['US-Student', 'Do-Not-Disturb', 'E-Mobility-Strategy'],
                             [['USStudent'], ['Do', 'Not', 'Disturb'], ['EMobility', 'Strategy']]):
        assert expand_compound_token(inp, split_on_len=None, split_on_casechange=True) == expected

    for inp, expected in zip(['US-Student', 'Do-Not-Disturb', 'E-Mobility-Strategy'],
                             [['US', 'Student'], ['Do', 'Not', 'Disturb'], ['EMobility', 'Strategy']]):
        assert expand_compound_token(inp, split_on_len=2, split_on_casechange=True) == expected

    assert expand_compound_token('E-Mobility-Strategy', split_on_len=1) == ['E', 'Mobility', 'Strategy']

    assert expand_compound_token('') == ['']

    assert expand_compound_token('Te;s,t', split_chars=[';', ','], split_on_len=1, split_on_casechange=False) \
           == expand_compound_token('Te-s-t', split_chars=['-'], split_on_len=1, split_on_casechange=False) \
           == ['Te', 's', 't']


@given(s=st.text(string.printable), split_chars=st.lists(st.characters(min_codepoint=32)),
       split_on_len=st.integers(1),
       split_on_casechange=st.booleans())
def test_expand_compound_token_hypothesis(s, split_chars, split_on_len, split_on_casechange):
    res = expand_compound_token(s, split_chars, split_on_len=split_on_len, split_on_casechange=split_on_casechange)

    assert isinstance(res, list)
    assert len(res) > 0

    s_contains_split_char = any(c in s for c in split_chars)
    s_is_split_chars = all(c in split_chars for c in s)

    if not s_contains_split_char:   # nothing to split on
        assert res == [s]

    if len(s) > 0:
        assert all([p for p in res])

    if not s_is_split_chars:
        for p in res:
            assert all(c not in p for c in split_chars)


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
    assert np.issubdtype(res.dtype, np.int)

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
        assert np.issubdtype(win.dtype, np.int)
        assert len(win) > 0
        assert np.min(win) >= 0
        assert np.max(win) < len(matches)

        i_in_win = 0
        for x in range(i-left, i+right+1):
            if 0 <= x < len(matches):
                assert x == win[i_in_win]
                i_in_win += 1


@pytest.mark.parametrize('pattern, tokens, match_type, ignore_case, glob_method, expected', [
    ('a', [], 'exact', False, 'match', []),
    ('', [], 'exact', False, 'match', []),
    ('', ['a', ''], 'exact', False, 'match', [False, True]),
    ('a', ['a', 'b', 'c'], 'exact', False, 'match', [True, False, False]),
    ('a', np.array(['a', 'b', 'c']), 'exact', False, 'match', [True, False, False]),
    ('A', ['a', 'b', 'c'], 'exact', False, 'match', [False, False, False]),
    ('A', ['a', 'b', 'c'], 'exact', True, 'match', [True, False, False]),
    (r'foo$', ['a', 'bfoo', 'c'], 'regex', False, 'match', [False, True, False]),
    (r'foo$', ['a', 'bFOO', 'c'], 'regex', False, 'match', [False, False, False]),
    (r'foo$', ['a', 'bFOO', 'c'], 'regex', True, 'match', [False, True, False]),
    (r'foo*', ['a', 'food', 'c'], 'glob', False, 'match', [False, True, False]),
    (r'foo*', ['a', 'FOOd', 'c'], 'glob', False, 'match', [False, False, False]),
    (r'foo*', ['a', 'FOOd', 'c'], 'glob', True, 'match', [False, True, False]),
    (r'foo*', ['a', 'FOOd', 'c'], 'glob', True, 'search', [False, True, False]),
])
def test_token_match(pattern, tokens, match_type, ignore_case, glob_method, expected):
    assert np.array_equal(token_match(pattern, tokens, match_type, ignore_case, glob_method), np.array(expected))


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


@pytest.mark.parametrize(
    'language, docs, expected',
    [
        ('english', [], []),
        ('english', [['A', 'simple', 'example', '.'], ['Simply', 'written', 'documents']],
                    [['A', 'simple', 'example', '.'], ['Simply', 'write', 'document']]),
        ('german', [['Ein', 'einfaches', 'Beispiel', 'in', 'einfachem', 'Deutsch', '.'],
                    ['Die', 'Dokumente', 'sind', 'sehr', 'kurz', '.']],
                    [['Ein', 'einfach', 'Beispiel', 'in', 'einfach', 'Deutsch', '.'],
                     ['Die', 'Dokument', 'sein', 'sehr', 'kurz', '.']]),
    ]
)
def test_lemmatize(language, docs, expected):
    docs_meta = pos_tag(docs, language)
    lemmata = lemmatize(docs, docs_meta, language)

    assert len(lemmata) == len(docs)

    for lem_tok, expected_tok in zip(lemmata, expected):
        assert isinstance(lem_tok, list)
        assert lem_tok == expected_tok

