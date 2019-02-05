import string

import pytest
import hypothesis.strategies as st
from hypothesis import given

from tmtoolkit.filter_tokens import filter_for_token, filter_for_tokenpattern, filter_for_pos


TESTTOKENS = {
    'doc1': [('lorem', ), ('ipsum', )],
    'doc2': [],
    'doc3': [('foo', ), 'bar', 'ipsum'],
    'doc4': ['foo', 'bar', 'lorem'],
    'doc5': ['Lorem'],
    'doc6': ['IPSUM', 'LOREM'],
}

TESTPOS = {
    'doc1': ['NN', 'NNP'],
    'doc2': [],
    'doc3': ['PRP', 'ADV', 'NNP'],
    'doc4': ['PRP', 'ADV', 'NN'],
    'doc5': ['NN'],
    'doc6': ['NNP', 'NN'],
}

TESTPOS_BAD = {
    'doc1': ['NN', 'NNP'],
    'doc2': ['BAD', 'BAD', 'BAD'],
    'doc3': ['PRP', 'ADV', 'NNP'],
    'doc4': ['PRP', 'ADV', 'NN'],
    'doc5': ['NN'],
    'doc6': ['NNP', 'NN'],
}


def _check_found_docs(found_docs, required_doc_labels, matches_removed=False):
    assert set(found_docs.keys()) == set(required_doc_labels)

    for dl, found_tokens in found_docs.items():
        if matches_removed:
            assert len(found_tokens) < len(TESTTOKENS[dl])
        else:
            assert len(found_tokens) == len(TESTTOKENS[dl])


def _check_found_doc_lengths(found_docs, required_doc_lengths):
    for dl, found_tokens in found_docs.items():
        assert len(found_tokens) == required_doc_lengths[dl]


@given(tokens=st.dictionaries(st.text(string.printable), st.lists(st.tuples(st.text()))),
       search_token=st.text())
def test_filter_for_token(tokens, search_token):
    if not search_token:
        with pytest.raises(ValueError):
            filter_for_token(tokens, search_token)
    else:
        found = filter_for_token(tokens, search_token)
        assert len(found) <= len(tokens)

        for dl, dt_ in found.items():
            assert dl in tokens.keys()
            dt = tokens[dl]
            assert dt_ == dt


@given(tokens=st.dictionaries(st.text(string.printable), st.lists(st.tuples(st.text()))),
       search_token=st.text(min_size=1))
def test_filter_for_token_insert_searchtoken(tokens, search_token):
    tokens['FINDME'] = [(search_token, )]
    found = filter_for_token(tokens, search_token)
    assert 1 <= len(found) <= len(tokens)

    assert 'FINDME' in found.keys()
    assert found['FINDME'] == [(search_token, )]


@given(tokens=st.dictionaries(st.text(string.printable),
                              st.lists(st.tuples(st.text(), st.text()))))    # POS tagged tokens (2-tuples)
def test_filter_for_tokenpattern(tokens):
    tokens['FINDME1'] = [('somePattern', )]
    tokens['FINDME2'] = [('other_pattern',)]
    found = filter_for_tokenpattern(tokens, r'[Pp]attern')
    assert 2 <= len(found) <= len(tokens)

    assert 'FINDME1' in found.keys()
    assert found['FINDME1'] == tokens['FINDME1']
    assert 'FINDME2' in found.keys()
    assert found['FINDME2'] == tokens['FINDME2']


@given(tokens=st.dictionaries(st.text(string.printable),
                              st.lists(st.tuples(st.text(), st.text()))))    # POS tagged tokens (2-tuples)
def test_filter_for_tokenpattern_remove_found(tokens):
    tokens['FINDME1'] = [('somePattern', )]
    tokens['FINDME2'] = [('other_pattern',)]
    found = filter_for_tokenpattern(tokens, r'[Pp]attern', remove_found_token=True)
    assert 2 <= len(found) <= len(tokens)

    assert 'FINDME1' in found.keys()
    assert found['FINDME1'] == []
    assert 'FINDME2' in found.keys()
    assert found['FINDME2'] == []


POSSIBLE_POS_TAGS = (
    'NN',
    'NNP',
    'VFIN',
    'ADJ',
    'ADV',
    'DET',
    'ART',
)


@given(tokens=st.dictionaries(st.text(string.printable),
                              st.lists(st.tuples(st.text(), st.sampled_from(POSSIBLE_POS_TAGS)))),    # POS tagged tokens (2-tuples)
       required_pos=st.sampled_from(POSSIBLE_POS_TAGS))
def test_filter_for_pos(tokens, required_pos):
    found = filter_for_pos(tokens, required_pos)

    assert len(found) == len(tokens)
    assert set(found.keys()) == set(tokens.keys())

    for dl, dt_ in found.items():
        dt = tokens[dl]
        assert len(dt_) <= len(dt)

        if dt:
            dt_tokens = list(zip(*dt))[0]
            assert all(tup[0] in dt_tokens and tup[1] == required_pos for tup in dt_)
