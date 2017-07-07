import string

import pytest
import hypothesis.strategies as st
from hypothesis import given

from tmtoolkit.preprocess import str_multisplit, expand_compound_token


def test_str_multisplit():
    punct = list(string.punctuation)

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
    assert expand_compound_token('US-Student') == ['US', 'Student']
    assert expand_compound_token('US-Student-X') == ['US', 'StudentX']
    assert expand_compound_token('Student-X') == ['StudentX']
    assert expand_compound_token('Do-Not-Disturb') == ['Do', 'Not', 'Disturb']
    assert expand_compound_token('E-Mobility-Strategy') == ['EMobility', 'Strategy']

    assert expand_compound_token('US-Student', split_on_len=None, split_on_casechange=True) == ['USStudent']
    assert expand_compound_token('Do-Not-Disturb', split_on_len=None, split_on_casechange=True) == ['Do', 'Not', 'Disturb']
    assert expand_compound_token('E-Mobility-Strategy', split_on_len=None, split_on_casechange=True) == ['EMobility', 'Strategy']

    assert expand_compound_token('US-Student', split_on_len=2, split_on_casechange=True) == ['US', 'Student']
    assert expand_compound_token('Do-Not-Disturb', split_on_len=2, split_on_casechange=True) == ['Do', 'Not', 'Disturb']
    assert expand_compound_token('E-Mobility-Strategy', split_on_len=2, split_on_casechange=True) == ['EMobility', 'Strategy']

    assert expand_compound_token('E-Mobility-Strategy', split_on_len=1) == ['E', 'Mobility', 'Strategy']


@given(s=st.text(), split_chars=st.lists(st.characters()), split_on_len=st.integers(0), split_on_casechange=st.booleans())
def test_expand_compound_token_hypothesis(s, split_chars, split_on_len, split_on_casechange):
    if not split_on_len and not split_on_casechange:
        with pytest.raises(ValueError):
            expand_compound_token(s, split_chars, split_on_len=split_on_len, split_on_casechange=split_on_casechange)
    else:
        res = expand_compound_token(s, split_chars, split_on_len=split_on_len, split_on_casechange=split_on_casechange)

        assert type(res) is list

        if len(s) == 0:
            assert res == []

        assert all(p for p in res)

        # if res and split_chars and any(c in s for c in split_chars):
        #     if split_on_len:
        #         assert min(map(len, res)) >= split_on_len

        for p in res:
            assert all(c not in p for c in split_chars)
