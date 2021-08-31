import string

import numpy as np
import pytest
from hypothesis import given, strategies as st

from ._testtools import strategy_tokens, strategy_2d_array
from tmtoolkit.utils import as_chararray
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
