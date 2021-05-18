"""
Preprocessing: Tests for ._nltk_extras submodule.
"""

from importlib.util import find_spec

import pytest

if not find_spec('spacy'):
    pytest.skip("skipping text processing tests: nltk_extras", allow_module_level=True)

import numpy as np

from tmtoolkit.utils import empty_chararray

from tmtoolkit.preprocess._docfuncs import doc_tokens, tokendocs2spacydocs

try:
    import nltk
    from tmtoolkit.preprocess._nltk_extras import stem, pos_tag_convert_penn_to_wn
except ImportError:
    pytestmark = pytest.mark.skipif(True, reason='nltk not installed')


@pytest.mark.parametrize(
    'docs, language, expected',
    [
        ([], 'english', []),
        ([[]], 'english', [[]]),
        ([[], []], 'english', [[], []]),
        ([['Doing', 'a', 'test', '.'], ['Apples', 'and', 'Oranges']], 'english',
         [['do', 'a', 'test', '.'], ['appl', 'and', 'orang']]),
        ([['Einen', 'Test', 'durchführen'], ['Äpfel', 'und', 'Orangen']], 'german',
         [['ein', 'test', 'durchfuhr'], ['apfel', 'und', 'orang']])
    ]
)
def test_stem(docs, language, expected):
    for docs_type in (0, 1, 2):
        if docs_type == 1:  # arrays
            docs = [np.array(d) if d else empty_chararray() for d in docs]
        elif docs_type == 2:  # spaCy docs
            docs = tokendocs2spacydocs(docs)

        res = stem(docs, language)
        assert isinstance(res, list)
        assert len(res) == len(docs)

        assert doc_tokens(res, to_lists=True) == expected


def test_pos_tag_convert_penn_to_wn():
    from nltk.corpus import wordnet as wn

    assert pos_tag_convert_penn_to_wn('JJ') == wn.ADJ
    assert pos_tag_convert_penn_to_wn('RB') == wn.ADV
    assert pos_tag_convert_penn_to_wn('NN') == wn.NOUN
    assert pos_tag_convert_penn_to_wn('VB') == wn.VERB

    for tag in ('', 'invalid', None):
        assert pos_tag_convert_penn_to_wn(tag) is None

