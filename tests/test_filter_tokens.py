import pytest

from tmtoolkit.filter_tokens import filter_for_token, filter_for_tokenpattern, filter_for_pos


TESTTOKENS = {
    u'doc1': [u'lorem', u'ipsum'],
    u'doc2': [],
    u'doc3': [u'foo', u'bar', u'ipsum'],
    u'doc4': [u'foo', u'bar', u'lorem'],
    u'doc5': [u'Lorem'],
    u'doc6': [u'IPSUM', u'LOREM'],
}

TESTPOS = {
    u'doc1': ['NN', 'NNP'],
    u'doc2': [],
    u'doc3': ['PRP', 'ADV', 'NNP'],
    u'doc4': ['PRP', 'ADV', 'NN'],
    u'doc5': ['NN'],
    u'doc6': ['NNP', 'NN'],
}

TESTPOS_BAD = {
    u'doc1': ['NN', 'NNP'],
    u'doc2': ['BAD', 'BAD', 'BAD'],
    u'doc3': ['PRP', 'ADV', 'NNP'],
    u'doc4': ['PRP', 'ADV', 'NN'],
    u'doc5': ['NN'],
    u'doc6': ['NNP', 'NN'],
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


def test_filter_for_token():
    _check_found_docs(filter_for_token(dict(), u'lorem'), [])
    _check_found_docs(filter_for_token(TESTTOKENS, u'lorem'),
                      (u'doc1', u'doc4'))
    _check_found_docs(filter_for_token(TESTTOKENS, u'lorem', ignore_case=True),
                      (u'doc1', u'doc4', u'doc5', u'doc6'))
    _check_found_docs(filter_for_token(TESTTOKENS, u'lorem', ignore_case=True, remove_found_token=True),
                      (u'doc1', u'doc4', u'doc5', u'doc6'), matches_removed=True)

    with pytest.raises(ValueError):
        filter_for_token(TESTTOKENS, u'')


def test_filter_for_tokenpattern():
    _check_found_docs(filter_for_tokenpattern(dict(), u'(lorem|ipsum)'), [])
    _check_found_docs(filter_for_tokenpattern(TESTTOKENS, u'(lorem|ipsum)'),
                      (u'doc1', u'doc3', u'doc4'))
    _check_found_docs(filter_for_tokenpattern(TESTTOKENS, u'(lorem|ipsum)', ignore_case=True),
                      (u'doc1', u'doc3', u'doc4', u'doc5', u'doc6'))
    _check_found_docs(filter_for_tokenpattern(TESTTOKENS, u'(lorem|ipsum)', ignore_case=True, remove_found_token=True),
                      (u'doc1', u'doc3', u'doc4', u'doc5', u'doc6'), matches_removed=True)


def test_filter_for_pos():
    res = filter_for_pos(TESTTOKENS, TESTPOS, 'N')
    _check_found_doc_lengths(res, {u'doc1': 2, u'doc2': 0, u'doc3': 1, u'doc4': 1, u'doc5': 1, u'doc6': 2})
    assert res[u'doc1'] == [u'lorem', u'ipsum']
    assert res[u'doc3'] == [u'ipsum']

    res = filter_for_pos(TESTTOKENS, TESTPOS, ('N', 'ADV'))
    _check_found_doc_lengths(res, {u'doc1': 2, u'doc2': 0, u'doc3': 2, u'doc4': 2, u'doc5': 1, u'doc6': 2})
    assert res[u'doc1'] == [u'lorem', u'ipsum']
    assert res[u'doc3'] == [u'bar', u'ipsum']

    res = filter_for_pos(TESTTOKENS, TESTPOS, 'X')
    _check_found_doc_lengths(res, {u'doc1': 0, u'doc2': 0, u'doc3': 0, u'doc4': 0, u'doc5': 0, u'doc6': 0})

    res = filter_for_pos(TESTTOKENS, TESTPOS, 'NN', simplify_pos=False)
    _check_found_doc_lengths(res, {u'doc1': 1, u'doc2': 0, u'doc3': 0, u'doc4': 1, u'doc5': 1, u'doc6': 1})
    assert res[u'doc4'] == [u'lorem']

    with pytest.raises(ValueError):
        filter_for_pos(TESTTOKENS, TESTPOS_BAD, 'N')

