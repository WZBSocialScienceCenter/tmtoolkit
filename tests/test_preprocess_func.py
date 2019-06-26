"""
Preprocessing: Functional API tests.
"""

from collections import Counter
import math
import string
import random

import hypothesis.strategies as st
from hypothesis import given
import pytest
import numpy as np
from scipy.sparse import isspmatrix_coo

from tmtoolkit.preprocess import tokenize, doc_lengths, vocabulary, vocabulary_counts, doc_frequencies, ngrams, \
    sparse_dtm, kwic, kwic_table


@pytest.mark.parametrize(
    'docs, language',
    [
        ([], 'english'),
        ([''], 'english'),
        (['', ''], 'english'),
        (['Simple test.'], 'english'),
        (['Simple test.', 'Document number 2 (of 3).', 'Number 3'], 'english'),
        (['Ein einfacher Test.\n\nUnd noch ein Satz, Satz, Satz.'], 'german'),
    ]
)
def test_tokenize(docs, language):
    res = tokenize(docs, language)
    assert isinstance(res, list)
    assert len(res) == len(docs)

    for dtok in res:
        assert isinstance(dtok, list)
        for t in dtok:
            assert isinstance(t, str)
            assert len(t) > 0
            assert ' ' not in t


@given(docs=st.lists(st.lists(st.text())))
def test_doc_lengths(docs):
    res = doc_lengths(docs)
    assert isinstance(res, list)
    assert len(res) == len(docs)

    for n, d in zip(res, docs):
        assert isinstance(n, int)
        assert n == len(d)


@given(docs=st.lists(st.lists(st.text())), sort=st.booleans())
def test_vocabulary(docs, sort):
    res = vocabulary(docs, sort=sort)

    if sort:
        assert isinstance(res, list)
        assert sorted(res) == res
    else:
        assert isinstance(res, set)

    for t in res:
        assert any([t in dtok for dtok in docs])


@given(docs=st.lists(st.lists(st.text())))
def test_vocabulary_counts(docs):
    res = vocabulary_counts(docs)

    assert isinstance(res, Counter)
    assert set(res.keys()) == vocabulary(docs)


@given(docs=st.lists(st.lists(st.text())), proportions=st.booleans())
def test_doc_frequencies(docs, proportions):
    res = doc_frequencies(docs, proportions=proportions)

    assert set(res.keys()) == vocabulary(docs)

    if proportions:
        assert isinstance(res, dict)
        assert all([0 < v <= 1 for v in res.values()])
    else:
        assert isinstance(res, Counter)
        assert all([0 < v for v in res.values()])


def test_doc_frequencies_example():
    docs = [
        list('abc'),
        list('abb'),
        list('ccc'),
        list('da'),
    ]

    abs_df = doc_frequencies(docs)
    assert dict(abs_df) == {
        'a': 3,
        'b': 2,
        'c': 2,
        'd': 1,
    }

    rel_df = doc_frequencies(docs, proportions=True)
    math.isclose(rel_df['a'], 3/4)
    math.isclose(rel_df['b'], 2/4)
    math.isclose(rel_df['c'], 2/4)
    math.isclose(rel_df['d'], 1/4)


@given(docs=st.lists(st.lists(st.text(string.printable))), pass_vocab=st.booleans())
def test_sparse_dtm(docs, pass_vocab):
    if pass_vocab:
        vocab = vocabulary(docs, sort=True)
        dtm = sparse_dtm(docs, vocab)
    else:
        dtm, vocab = sparse_dtm(docs)

    assert isspmatrix_coo(dtm)
    assert dtm.shape == (len(docs), len(vocab))
    assert vocab == vocabulary(docs, sort=True)


def test_sparse_dtm_example():
    docs = [
        list('abc'),
        list('abb'),
        list('ccc'),
        [],
        list('da'),
    ]

    dtm, vocab = sparse_dtm(docs)
    assert vocab == list('abcd')
    assert dtm.shape == (5, 4)

    assert np.array_equal(dtm.A, np.array([
        [1, 1, 1, 0],
        [1, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
    ]))


@given(tokens=st.lists(st.text()), n=st.integers(0, 4))
def test_ngrams(tokens, n):
    n_tok = len(tokens)

    if n < 2:
        with pytest.raises(ValueError):
            ngrams([tokens], n)
    else:
        ng = ngrams([tokens], n, join=False)[0]

        if n_tok < n:
            if n_tok == 0:
                assert ng == []
            else:
                assert len(ng) == 1
                assert ng == [tokens]
        else:
            assert len(ng) == n_tok - n + 1
            assert all(len(g) == n for g in ng)

            tokens_ = list(ng[0])
            if len(ng) > 1:
                tokens_ += [g[-1] for g in ng[1:]]
            assert tokens_ == tokens

        ngrams_joined = ngrams([tokens], n, join=True, join_str='')[0]
        assert len(ngrams_joined) == len(ng)

        for g_joined, g_tuple in zip(ngrams_joined, ng):
            assert g_joined == ''.join(g_tuple)


@given(docs=st.lists(st.lists(st.text(string.printable))), search_term_exists=st.booleans(),
       context_size=st.integers(1, 5), non_empty=st.booleans(), glue=st.booleans(), highlight_keyword=st.booleans())
def test_kwic(docs, context_size, search_term_exists, non_empty, glue, highlight_keyword):
    vocab = list(vocabulary(docs) - {''})

    if search_term_exists and len(vocab) > 0:
        s = random.choice(vocab)
    else:
        s = 'thisdoesnotexist'

    res = kwic(docs, s, context_size=context_size, non_empty=non_empty,
               glue=' ' if glue else None,
               highlight_keyword='*' if highlight_keyword else None)

    assert isinstance(res, list)

    if s in vocab:
        for win in res:
            if non_empty:
                assert len(win) > 0

            for w in win:
                if highlight_keyword:
                    assert '*' + s + '*' in w
                else:
                    assert s in w

                if not glue:
                    assert 0 <= len(w) <= context_size * 2 + 1
    else:
        if non_empty:
            assert len(res) == 0
        else:
            assert all([n == 0 for n in map(len, res)])


def test_kwic_example():
    docs = [
        list('abccbc'),
        list('abbdeeffde'),
        list('ccc'),
        [],
        list('daabbbbbbcb'),
    ]

    res = kwic(docs, 'd')

    assert res == [[],
         [['b', 'b', 'd', 'e', 'e'], ['f', 'f', 'd', 'e']],
         [],
         [],
         [['d', 'a', 'a']]
    ]

    res = kwic(docs, 'd', non_empty=True)

    assert res == [
         [['b', 'b', 'd', 'e', 'e'], ['f', 'f', 'd', 'e']],
         [['d', 'a', 'a']]
    ]

    res = kwic(docs, 'd', non_empty=True, glue=' ')

    assert res == [['b b d e e', 'f f d e'], ['d a a']]

    res = kwic(docs, 'd', non_empty=True, glue=' ', highlight_keyword='*')

    assert res == [['b b *d* e e', 'f f *d* e'], ['*d* a a']]

    res = kwic(docs, 'd', highlight_keyword='*')

    assert res == [[],
         [['b', 'b', '*d*', 'e', 'e'], ['f', 'f', '*d*', 'e']],
         [],
         [],
         [['*d*', 'a', 'a']]
    ]

