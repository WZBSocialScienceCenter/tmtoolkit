"""
Preprocessing: Tests for ._common submodule.
"""

import pytest

from tmtoolkit.preprocess._common import (
    DEFAULT_LANGUAGE_MODELS, load_stopwords, simplified_pos
)


LANGUAGE_CODES = list(sorted(DEFAULT_LANGUAGE_MODELS.keys()))


def test_load_stopwords():
    for code in LANGUAGE_CODES:
        stopwords = load_stopwords(code)

        if code in {'lt', 'ja', 'zh'}:
            assert stopwords is None
        else:
            assert isinstance(stopwords, list)
            assert len(stopwords) > 0
            assert isinstance(next(iter(stopwords)), str)

    with pytest.raises(ValueError):
        load_stopwords('foo')


def test_simplified_pos():
    # tagset "ud"
    assert simplified_pos('') == ''
    assert simplified_pos('NOUN') == 'N'
    assert simplified_pos('PROPN') == 'N'
    assert simplified_pos('VERB') == 'V'
    assert simplified_pos('ADJ') == 'ADJ'
    assert simplified_pos('ADV') == 'ADV'
    assert simplified_pos('FOO') == ''

    assert simplified_pos('', tagset='wn') == ''
    assert simplified_pos('N', tagset='wn') == 'N'
    assert simplified_pos('V', tagset='wn') == 'V'
    assert simplified_pos('ADJ', tagset='wn') == 'ADJ'
    assert simplified_pos('ADV', tagset='wn') == 'ADV'
    assert simplified_pos('AD', tagset='wn') == ''
    assert simplified_pos('ADX', tagset='wn') == ''
    assert simplified_pos('PRP', tagset='wn') == ''
    assert simplified_pos('XYZ', tagset='wn') == ''
    assert simplified_pos('NN', tagset='wn') == 'N'
    assert simplified_pos('NNP', tagset='wn') == 'N'
    assert simplified_pos('VX', tagset='wn') == 'V'
    assert simplified_pos('ADJY', tagset='wn') == 'ADJ'
    assert simplified_pos('ADVZ', tagset='wn') == 'ADV'

    assert simplified_pos('NNP', tagset='penn') == 'N'
    assert simplified_pos('VFOO', tagset='penn') == 'V'
    assert simplified_pos('JJ', tagset='penn') == 'ADJ'
    assert simplified_pos('JJX', tagset='penn') == 'ADJ'
    assert simplified_pos('RB', tagset='penn') == 'ADV'
    assert simplified_pos('RBFOO', tagset='penn') == 'ADV'
    assert simplified_pos('FOOBAR', tagset='penn') == ''
