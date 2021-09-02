"""
Tests for tmtoolkit.tokenseq module.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import string
from collections import Counter

import numpy as np
import pytest
import random
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

from ._testtools import strategy_tokens, strategy_2d_array, strategy_lists_of_tokens
from tmtoolkit.utils import as_chararray, flatten_list
from tmtoolkit import tokenseq


@pytest.mark.parametrize('tokens, expected', [
    ([], []),
    ([''], [0]),
    (['a'], [1]),
    (['abc'], [3]),
    (['abc', 'd'], [3, 1]),
])
def test_token_lengths(tokens, expected):
    assert tokenseq.token_lengths(tokens) == expected


@given(tokens=strategy_tokens(string.printable),
       as_array=st.booleans())
def test_token_lengths_hypothesis(tokens, as_array):
    if as_array:
        tokens = as_chararray(tokens)

    res = tokenseq.token_lengths(tokens)

    assert isinstance(res, list)
    assert len(res) == len(tokens)
    assert all([isinstance(n, int) and n >= 0 for n in res])


@given(xy=strategy_2d_array(int, 0, 100, min_side=2, max_side=100),
       as_prob=st.booleans(),
       n_total_factor=st.floats(min_value=1, max_value=10, allow_nan=False),
       k=st.integers(min_value=0, max_value=5),
       normalize=st.booleans())
def test_pmi_hypothesis(xy, as_prob, n_total_factor, k, normalize):
    size = len(xy)
    xy = xy[:, 0:2]
    x = xy[:, 0]
    y = xy[:, 1]
    xy = np.min(xy, axis=1) * np.random.uniform(0, 1, size)
    n_total = 1 + n_total_factor * (np.sum(x) + np.sum(y))

    if as_prob:
        x = x / n_total
        y = y / n_total
        xy = xy / n_total
        n_total = None

    if k < 1 or (k > 1 and normalize):
        with pytest.raises(ValueError):
            tokenseq.pmi(x, y, xy, n_total=n_total, k=k, normalize=normalize)
    else:
        res = tokenseq.pmi(x, y, xy, n_total=n_total, k=k, normalize=normalize)
        assert isinstance(res, np.ndarray)
        assert len(res) == len(x)

        if np.all(x > 0) and np.all(y > 0):
            assert np.sum(np.isnan(res)) == 0
            if normalize:
                assert np.all(res == tokenseq.npmi(x, y, xy, n_total=n_total))
                assert np.all(res >= -1) and np.all(res <= 1)
            elif k == 2:
                assert np.all(res == tokenseq.pmi2(x, y, xy, n_total=n_total))
            elif k == 3:
                assert np.all(res == tokenseq.pmi3(x, y, xy, n_total=n_total))


@given(xy=arrays(int, array_shapes(min_dims=1, max_dims=1)))
def test_simple_collocation_counts_hypothesis(xy):
    res = tokenseq.simple_collocation_counts(None, None, xy, None)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(xy)


@pytest.mark.parametrize('args, expected', [
    (
        {},
        [(('e', 'f'), 0.8105361810656604),
         (('b', 'c'), 0.6915604067044995),
         (('d', 'e'), 0.6122380649615099),
         (('a', 'b'), 0.43193903282626694),
         (('c', 'd'), 0.43193903282626694),
         (('c', 'e'), 0.3823761795182354),
         (('f', 'b'), 0.18728849070804096),
         (('c', 'b'), 0.14367999690515007),
         (('e', 'b'), 0.044177097787776946)]
    ),
    (
        dict(min_count=2),
        [(('e', 'f'), 0.8105361810656604),
         (('b', 'c'), 0.6915604067044995),
         (('c', 'e'), 0.3823761795182354),
         (('c', 'b'), 0.14367999690515007)]
    ),
    (
        dict(threshold=0.5),
        [(('e', 'f'), 0.8105361810656604),
         (('b', 'c'), 0.6915604067044995),
         (('d', 'e'), 0.6122380649615099)]
    ),
    (
        dict(min_count=2, threshold=0.5, glue='_&_'),
        [('e_&_f', 0.8105361810656604),
         ('b_&_c', 0.6915604067044995)]
    ),
    (
        dict(min_count=2, statistic=tokenseq.pmi),
        [(('e', 'f'), 1.7346010553881064),
         (('b', 'c'), 1.000631880307906),
         (('c', 'e'), 0.8183103235139513),
         (('c', 'b'), 0.30748469974796055)]
    ),
    (
        dict(min_count=2, statistic=tokenseq.pmi2),
        [(('e', 'f'), -0.4054651081081644),
         (('b', 'c'), -0.4462871026284194),
         (('c', 'e'), -1.3217558399823195),
         (('c', 'b'), -1.8325814637483102)]
    ),
    (
        dict(min_count=2, statistic=tokenseq.pmi3),
        [(('b', 'c'), -1.8932060855647448),
         (('e', 'f'), -2.5455312716044354),
         (('c', 'e'), -3.4618220034785905),
         (('c', 'b'), -3.972647627244581)]
    )
])
def test_token_collocations(args, expected):
    sentences = tokens = ['a b c d e f b c b'.split(),
                          'c e b c b c e f'.split()]
    res = tokenseq.token_collocations(sentences, **args)
    colloc, stat = zip(*res)
    expected_colloc, expected_stat = zip(*expected)
    assert colloc == expected_colloc
    assert np.allclose(stat, expected_stat)


@given(sentences=strategy_lists_of_tokens(string.printable),
       threshold=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
       min_count=st.integers(),
       pass_embed_tokens=st.integers(min_value=0, max_value=3),
       statistic=st.sampled_from([tokenseq.pmi, tokenseq.npmi, tokenseq.pmi2, tokenseq.pmi3,
                                  tokenseq.simple_collocation_counts]),
       pass_vocab_counts=st.booleans(),
       glue=st.one_of(st.none(), st.text(string.printable)),
       return_statistic=st.booleans(),
       rank=st.sampled_from([None, 'asc', 'desc'])
       )
def test_token_collocations_hypothesis(sentences, threshold, min_count, pass_embed_tokens, statistic, pass_vocab_counts,
                                       glue, return_statistic, rank):
    ngramsize = 2
    tok = flatten_list(sentences)

    if pass_embed_tokens > 0:
        embed_tokens = random.choices(tok, k=min(pass_embed_tokens, len(tok)))
    else:
        embed_tokens = None

    if pass_vocab_counts:
        vocab_counts = Counter(tok)
    else:
        vocab_counts = None

    args = dict(sentences=sentences, threshold=threshold, min_count=min_count, embed_tokens=embed_tokens,
                statistic=statistic, vocab_counts=vocab_counts, glue=glue,
                return_statistic=return_statistic, rank=rank)

    if min_count < 0:
        with pytest.raises(ValueError):
            tokenseq.token_collocations(**args)
    else:
        res = tokenseq.token_collocations(**args)
        assert isinstance(res, list)
        assert len(res) <= max(1, len(set(tok)) - ngramsize + 1)

        statvalues = []
        for row in res:
            if return_statistic:
                assert isinstance(row, tuple)
                assert len(row) == 2
                colloc, stat = row
                assert isinstance(stat, float)
                if threshold:
                    assert stat >= threshold
                if statistic is tokenseq.simple_collocation_counts:
                    assert stat >= min_count
                if rank:
                    statvalues.append(stat)
            else:
                colloc = row

            if glue is None:
                assert isinstance(colloc, tuple)
                assert all([isinstance(t, str) for t in colloc])
                if embed_tokens:
                    assert len(colloc) >= ngramsize
                else:
                    assert len(colloc) == ngramsize
            else:
                assert isinstance(colloc, str)
                assert glue in colloc
        if rank:
            assert statvalues == sorted(statvalues, reverse=rank == 'desc')


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
    assert np.array_equal(tokenseq.token_match(pattern, tokens, match_type, ignore_case, glob_method),
                          np.array(expected))


@pytest.mark.parametrize('pattern, tokens, match_type, ignore_case, glob_method, expected', [
    ('a', [], 'exact', False, 'match', []),
    ('', [], 'exact', False, 'match', []),
    ('', ['a', ''], 'exact', False, 'match', [False, True]),
    ('a', ['a', 'b', 'c'], 'exact', False, 'match', [True, False, False]),
    (['a'], ['a', 'b', 'c'], 'exact', False, 'match', [True, False, False]),
    (['a', 'c'], ['a', 'b', 'c'], 'exact', False, 'match', [True, False, True]),
    (('a', 'c'), np.array(['a', 'b', 'c']), 'exact', False, 'match', [True, False, True]),
    ({'A'}, ['a', 'b', 'c'], 'exact', True, 'match', [True, False, False]),
    ({'A', 'a'}, ['a', 'b', 'c'], 'exact', True, 'match', [True, False, False]),
    (['A', 'A'], ['a', 'b', 'c'], 'exact', True, 'match', [True, False, False])
])
def test_token_match_multi_pattern(pattern, tokens, match_type, ignore_case, glob_method, expected):
    assert np.array_equal(tokenseq.token_match_multi_pattern(pattern, tokens, match_type, ignore_case, glob_method),
                          np.array(expected))


def test_token_match_subsequent():
    tok = ['green', 'test', 'emob', 'test', 'greener', 'tests', 'test', 'test']

    with pytest.raises(ValueError):
        tokenseq.token_match_subsequent('pattern', tok)

    with pytest.raises(ValueError):
        tokenseq.token_match_subsequent(['pattern'], tok)

    assert tokenseq.token_match_subsequent(['a', 'b'], []) == []

    assert tokenseq.token_match_subsequent(['foo', 'bar'], tok) == []

    res = tokenseq.token_match_subsequent(['green*', 'test*'], tok, match_type='glob')
    assert len(res) == 2
    assert np.array_equal(res[0], np.array([0, 1]))
    assert np.array_equal(res[1], np.array([4, 5]))

    res = tokenseq.token_match_subsequent(['green*', 'test*', '*'], tok, match_type='glob')
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
            tokenseq.token_match_subsequent(patterns, tokens)
    else:
        res = tokenseq.token_match_subsequent(patterns, tokens)

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
        tokenseq.token_join_subsequent(tok, 'invalid')

    assert tokenseq.token_join_subsequent(tok, []) == tok

    matches = tokenseq.token_match_subsequent(['green*', 'test*'], tok, match_type='glob')
    assert tokenseq.token_join_subsequent(tok, matches) == ['green_test', 'emob', 'test', 'greener_tests', 'test',
                                                            'test']

    matches = tokenseq.token_match_subsequent(['green*', 'test*', '*'], tok, match_type='glob')
    assert tokenseq.token_join_subsequent(tok, matches) == ['green_test_emob', 'test', 'greener_tests_test', 'test']


@given(tokens=st.lists(st.text(string.printable)), n_patterns=st.integers(0, 4))
def test_token_glue_subsequent_hypothesis(tokens, n_patterns):
    tokens_arr = np.array(tokens)

    n_patterns = min(len(tokens), n_patterns)

    pat_ind = np.arange(n_patterns)
    np.random.shuffle(pat_ind)
    patterns = list(tokens_arr[pat_ind])

    if len(patterns) > 1:
        matches = tokenseq.token_match_subsequent(patterns, tokens)
        assert tokenseq.token_join_subsequent(tokens, []) == tokens

        if len(tokens) == 0:
            assert tokenseq.token_join_subsequent(tokens, matches) == []
        elif len(matches) == 0:
            assert tokenseq.token_join_subsequent(tokens, matches) == tokens
        else:
            res = tokenseq.token_join_subsequent(tokens, matches)
            assert isinstance(res, list)
            assert 0 < len(res) < len(tokens)

            for ind in matches:
                assert '_'.join(tokens_arr[ind]) in res

