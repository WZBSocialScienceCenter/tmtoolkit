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
import datatable as dt
from nltk.corpus import wordnet as wn
from scipy.sparse import isspmatrix_coo
import nltk

from tmtoolkit.utils import flatten_list
from tmtoolkit.preprocess import (tokenize, doc_lengths, vocabulary, vocabulary_counts, doc_frequencies, ngrams,
    sparse_dtm, kwic, kwic_table, glue_tokens, simplified_pos, tokens2ids, ids2tokens, pos_tag_convert_penn_to_wn,
    str_multisplit, expand_compound_token, remove_chars, make_index_window_around_matches, token_match_subsequent,
    token_glue_subsequent, transform, to_lowercase, stem, pos_tag, lemmatize, expand_compounds, clean_tokens,
    filter_tokens, filter_documents, filter_documents_by_name, filter_for_pos,
    remove_common_tokens, remove_uncommon_tokens
)


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


@pytest.mark.parametrize(
    'docs, language, expected',
    [
        ([], 'english', []),
        ([['']], 'english', [['']]),
        ([[''], []], 'english', [[''], []]),
        ([['Doing', 'a', 'test', '.'], ['Apples', 'and', 'Oranges']], 'english',
         [['do', 'a', 'test', '.'], ['appl', 'and', 'orang']]),
        ([['Einen', 'Test', 'durchführen'], ['Äpfel', 'und', 'Orangen']], 'german',
         [['ein', 'test', 'durchfuhr'], ['apfel', 'und', 'orang']])
    ]
)
def test_stem(docs, language, expected):
    res = stem(docs, language)
    assert isinstance(res, list)
    assert res == expected


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

@given(docs=st.lists(st.lists(st.text(string.printable))), search_term_exists=st.booleans(),
       context_size=st.integers(1, 5))
def test_kwic_table(docs, context_size, search_term_exists):
    vocab = list(vocabulary(docs) - {''})

    if search_term_exists and len(vocab) > 0:
        s = random.choice(vocab)
    else:
        s = 'thisdoesnotexist'

    res = kwic_table(docs, s, context_size=context_size)

    assert isinstance(res, dt.Frame)
    assert res.names == ('doc', 'context', 'kwic')

    if s in vocab:
        assert res.shape[0] > 0
        kwic_col = res[:, dt.f.kwic].to_list()[0]
        for kwic_match in kwic_col:
            assert kwic_match.count('*') >= 2
    else:
        assert res.shape[0] == 0


def test_glue_tokens_example():
    docs = [
        list('abccbc'),
        list('abbdeeffde'),
        list('ccc'),
        [],
        list('daabbbbbbcb'),
    ]

    res = glue_tokens(docs, ('a', 'b'))

    assert res == [
        ['a_b', 'c', 'c', 'b', 'c'],
        ['a_b', 'b', 'd', 'e', 'e', 'f', 'f', 'd', 'e'],
        ['c', 'c', 'c'],
        [],
        ['d', 'a', 'a_b', 'b', 'b', 'b', 'b', 'b', 'c', 'b']
    ]

    meta = [
        {'pos': list('NNVVVV')},
        {'pos': list('ANVXDDAAV')},
        {'pos': list('VVV')},
        {'pos': list()},
        {'pos': list('NVVVAXDVVV')},
    ]

    docs_meta = list(zip(docs, meta))

    res, glued_tok = glue_tokens(docs_meta, ('a', 'b'), return_glued_tokens=True)

    assert res == [
        (['a_b', 'c', 'c', 'b', 'c'], {'pos': [None, 'V', 'V', 'V', 'V']}),
        (['a_b', 'b', 'd', 'e', 'e', 'f', 'f', 'd', 'e'], {'pos': [None, 'V', 'X', 'D', 'D', 'A', 'A', 'V']}),
        (['c', 'c', 'c'], {'pos': ['V', 'V', 'V']}),
        ([], {'pos': []}),
        (['d', 'a', 'a_b', 'b', 'b', 'b', 'b', 'b', 'c', 'b'], {'pos': ['N', 'V', None, 'A', 'X', 'D', 'V', 'V', 'V']})
    ]

@pytest.mark.parametrize(
    'docs, expected',
    [
        ([], []),
        ([['']], [['']]),
        ([[''], []], [[''], []]),
        ([['An', 'US-Student', '.']], [['An', 'US', 'Student', '.']]),
    ]
)
def test_expand_compounds(docs, expected):
    assert expand_compounds(docs) == expected


@given(docs=st.lists(st.lists(st.text(string.printable))),
       pass_docs_meta=st.booleans(),
       remove_punct=st.integers(0, 2),
       remove_stopwords=st.integers(0, 2),
       remove_empty=st.booleans(),
       remove_shorter_than=st.integers(-2, 5),
       remove_longer_than=st.integers(-2, 10),
       remove_numbers=st.booleans())
def test_clean_tokens(docs, pass_docs_meta, remove_punct, remove_stopwords, remove_empty, remove_shorter_than,
                      remove_longer_than, remove_numbers):
    if pass_docs_meta:
        docs_meta = [{'meta_pos': np.random.choice(list('ABC'), len(dtok), replace=True).tolist()} for dtok in docs]
    else:
        docs_meta = None

    if remove_punct == 2:
        remove_punct = np.random.choice(list(string.punctuation), 5, replace=False).tolist()
    else:
        remove_punct = bool(remove_punct)

    if remove_stopwords == 2 and len(flatten_list(docs)) > 0:
        remove_stopwords = np.random.choice(flatten_list(docs), 5, replace=True).tolist()
    else:
        remove_stopwords = bool(remove_stopwords)

    if remove_shorter_than == -2:
        remove_shorter_than = None

    if remove_longer_than == -2:
        remove_longer_than = None

    if remove_shorter_than == -1 or remove_longer_than == -1:
        with pytest.raises(ValueError):
            clean_tokens(docs, docs_meta, remove_punct=remove_punct, remove_stopwords=remove_stopwords,
                         remove_empty=remove_empty, remove_shorter_than=remove_shorter_than,
                         remove_longer_than=remove_longer_than, remove_numbers=remove_numbers)
    else:
        res = clean_tokens(docs, docs_meta, remove_punct=remove_punct, remove_stopwords=remove_stopwords,
                           remove_empty=remove_empty, remove_shorter_than=remove_shorter_than,
                           remove_longer_than=remove_longer_than, remove_numbers=remove_numbers)

        blacklist = set()

        if isinstance(remove_punct, list):
            blacklist.update(remove_punct)
        elif remove_punct is True:
            blacklist.update(string.punctuation)

        if isinstance(remove_stopwords, list):
            blacklist.update(remove_stopwords)
        elif remove_stopwords is True:
            blacklist.update(nltk.corpus.stopwords.words('english'))

        if remove_empty:
            blacklist.update('')

        if docs_meta is None:
            docs_ = res
            docs_meta_ = None
        else:
            assert isinstance(res, tuple) and len(res) == 2
            docs_, docs_meta_ = res

            assert isinstance(docs_meta_, list)
            assert len(docs_meta) == len(docs_meta_)

        assert len(docs) == len(docs_)

        for i, (dtok, dtok_) in enumerate(zip(docs, docs_)):
            assert len(dtok) >= len(dtok_)
            del dtok

            if docs_meta_ is not None:
                assert len(dtok_) == len(docs_meta_[i]['meta_pos'])

            assert all([w not in dtok_ for w in blacklist])

            tok_lengths = np.array(list(map(len, dtok_)))

            if remove_shorter_than is not None:
                assert np.all(tok_lengths >= remove_shorter_than)

            if remove_longer_than is not None:
                assert np.all(tok_lengths <= remove_longer_than)

            if remove_numbers and dtok_:
                assert not np.any(np.char.isnumeric(dtok_))


@pytest.mark.parametrize('docs, docs_meta, search_patterns, expected_docs, expected_docs_meta', [
    ([], None, 'test', [], None),
    ([], [], 'test', [], []),
    ([[]], None, 'test', [[]], None),
    ([['A', 'test', '!'], ['Teeeest', 'nope']], None, 'test', [['test'], []], None),
    ([['A', 'test', '!'], ['Teeeest', 'nope']], [{'meta': list('ABC')}, {'meta': list('CC')}], 'test',
     [['test'], []], [{'meta': ['B']}, {'meta': []}]),
])
def test_filter_tokens(docs, docs_meta, search_patterns, expected_docs, expected_docs_meta):
    # very simple test here
    # more tests are done via TMPreproc

    res = filter_tokens(docs, search_patterns, docs_meta)

    if docs_meta is None:
        res_docs = res
        res_docs_meta = None
    else:
        assert isinstance(res, tuple)
        assert len(res) == 2
        res_docs, res_docs_meta = res

    assert res_docs == expected_docs
    assert res_docs_meta == expected_docs_meta

@pytest.mark.parametrize(
    'docs, docs_meta, doc_labels, search_patterns, expected_docs, expected_docs_meta, expected_doc_labels',
    [
        ([], None, None, 'test', [], None, None),
        ([], [], None, 'test', [], [], None),
        ([[]], None, None, 'test', [], None, None),
        ([['A', 'test', '!'], ['Teeeest', 'nope']], None, None, 'test', [['A', 'test', '!']], None, None),
        ([['A', 'test', '!'], ['Teeeest', 'nope']], [{'meta': list('ABC')}, {'meta': list('CC')}], None, 'test',
         [['A', 'test', '!']], [{'meta': list('ABC')}], None),
        ([['A', 'test', '!'], ['Teeeest', 'nope']], None, [['doc1'], ['doc2']], 'test',
         [['A', 'test', '!']], None, [['doc1']]),
        ([['A', 'test', '!'], ['Teeeest', 'nope']], [{'meta': list('ABC')}, {'meta': list('CC')}], [['doc1'], ['doc2']],
         'test',
         [['A', 'test', '!']], [{'meta': list('ABC')}], [['doc1']]),
    ]
)
def test_filter_documents(docs, docs_meta, doc_labels, search_patterns,
                          expected_docs, expected_docs_meta, expected_doc_labels):
    # very simple test here
    # more tests are done via TMPreproc

    res = filter_documents(docs, search_patterns, docs_meta, doc_labels)

    res_docs_meta = None
    res_doc_labels = None

    if docs_meta is None and doc_labels is None:
        res_docs = res
    else:
        assert isinstance(res, tuple)
        assert len(res) == 1 + sum([docs_meta is not None, doc_labels is not None])

        if len(res) == 2:
            if docs_meta is not None:
                res_docs, res_docs_meta = res
            else:
                res_docs, res_doc_labels = res
        else:
            assert len(res) == 3
            res_docs, res_docs_meta, res_doc_labels = res

    assert res_docs == expected_docs
    assert res_docs_meta == expected_docs_meta
    assert res_doc_labels == expected_doc_labels


@pytest.mark.parametrize(
    'docs, doc_labels, docs_meta, name_patterns, expected_docs, expected_doc_labels, expected_docs_meta',
    [
        ([], [], None, 'test', [], [], None),
        ([[]], ['doc1'], None, 'test', [], [], None),
        ([[]], ['test'], None, 'test', [[]], ['test'], None),
        ([['t1', 't2'], ['foo']], ['test', 'bar'], None, 'test', [['t1', 't2']], ['test'], None),
        ([['t1', 't2'], ['foo']], ['test', 'bar'], [{'meta': ['A', 'B']}, {'meta': ['C']}], 'test',
         [['t1', 't2']], ['test'], [{'meta': ['A', 'B']}]),
    ]
)
def test_filter_documents_by_name(docs, doc_labels, docs_meta, name_patterns,
                                  expected_docs, expected_doc_labels, expected_docs_meta):
    # very simple test here
    # more tests are done via TMPreproc

    res = filter_documents_by_name(docs, doc_labels, docs_meta=docs_meta, name_patterns=name_patterns)

    if docs_meta is not None:
        res_docs, res_doc_labels, res_docs_meta = res
    else:
        res_docs, res_doc_labels = res
        res_docs_meta = None

    assert res_docs == expected_docs
    assert res_doc_labels == expected_doc_labels
    assert res_docs_meta == expected_docs_meta


@pytest.mark.parametrize(
    'docs, docs_meta, required_pos, expected_docs, expected_docs_meta',
    [
        ([], [], 'test', [], []),
        ([[]], [{'meta_pos': []}], 'test', [[]], [{'meta_pos': []}]),
        ([['t1', 't2'], ['foo']], [{'meta_pos': ['A', 'B']}, {'meta_pos': ['A']}], 'A',
         [['t1'], ['foo']], [{'meta_pos': ['A']}, {'meta_pos': ['A']}]),
    ]
)
def test_filter_for_pos(docs, docs_meta, required_pos, expected_docs, expected_docs_meta):
    # very simple test here
    # more tests are done via TMPreproc

    res_docs, res_docs_meta = filter_for_pos(docs, docs_meta, required_pos=required_pos, simplify_pos=False)

    assert res_docs == expected_docs
    assert res_docs_meta == expected_docs_meta


@pytest.mark.parametrize(
    'docs, docs_meta, common, thresh, absolute, expected_docs, expected_docs_meta',
    [
        ([], None, True, 0.75, False, [], None),
        ([[]], None, True, 0.75, False, [[]], None),
        ([['a']] * 10, None, True, 0.9, False, [[]] * 10, None),
        ([['a']] * 9 + [['b']], None, True, 0.9, False, [[]] * 9 + [['b']], None),
        ([['a']] * 9 + [['b']], [{'meta': ['A']}] * 9 + [{'meta': ['B']}], True, 0.9, False,
         [[]] * 9 + [['b']], [{'meta': []}] * 9 + [{'meta': ['B']}]),
        ([['a']] * 9 + [['b']], None, False, 1, True, [['a']] * 9 + [[]], None),
    ]
)
def test_remove_common_uncommon_tokens(docs, docs_meta, common, thresh, absolute, expected_docs, expected_docs_meta):
    # very simple test here
    # more tests are done via TMPreproc
    if common:
        fn = remove_common_tokens
    else:
        fn = remove_uncommon_tokens

    res = fn(docs, docs_meta, df_threshold=thresh, absolute=absolute)

    if docs_meta is None:
        res_docs = res
        res_docs_meta = None
    else:
        res_docs, res_docs_meta = res

    assert res_docs == expected_docs
    assert res_docs_meta == expected_docs_meta


@given(docs=st.lists(st.lists(st.text(string.printable))))
def test_transform(docs):
    expected = [[t.lower() for t in d] for d in docs]

    res1 = transform(docs, str.lower)
    res2 = to_lowercase(docs)

    assert res1 == res2 == expected

    def repeat_token(t, k):
        return t * k

    res = transform(docs, repeat_token, k=3)

    assert len(res) == len(docs)
    for dtok_, dtok in zip(res, docs):
        assert len(dtok_) == len(dtok)
        for t_, t in zip(dtok_, dtok):
            assert len(t_) == 3 * len(t)


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
    assert expand_compound_token('US-Student') == ['US', 'Student']
    assert expand_compound_token('US-Student-X') == ['US', 'StudentX']
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
       split_on_len=st.integers(0),
       split_on_casechange=st.booleans())
def test_expand_compound_token_hypothesis(s, split_chars, split_on_len, split_on_casechange):
    if not split_on_len and not split_on_casechange:
        with pytest.raises(ValueError):
            expand_compound_token(s, split_chars, split_on_len=split_on_len, split_on_casechange=split_on_casechange)
    else:
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


@given(docs=st.lists(st.lists(st.text())), chars=st.lists(st.characters()))
def test_remove_chars(docs, chars):
    if len(chars) == 0:
        with pytest.raises(ValueError):
            remove_chars(docs, chars)
    else:
        docs_ = remove_chars(docs, chars)
        assert len(docs_) == len(docs)

        for d_, d in zip(docs_, docs):
            assert len(d_) == len(d)

            for t_, t in zip(d_, d):
                assert len(t_) <= len(t)
                assert all(c not in t_ for c in chars)


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


@pytest.mark.parametrize(
    'language, docs, expected',
    [
        ('english', [], []),
        ('english', [['A', 'simple', 'example', '.'], ['Simply', 'written', 'documents']],
                    [['DT', 'JJ', 'NN', '.'], ['NNP', 'VBN', 'NNS']]),
        ('german', [['Ein', 'einfaches', 'Beispiel', 'in', 'einfachem', 'Deutsch', '.'],
                    ['Die', 'Dokumente', 'sind', 'sehr', 'kurz', '.']],
                    [['ART', 'ADJA', 'NN', 'APPR', 'ADJA', 'NN', '$.'],
                     ['ART', 'NN', 'VAFIN', 'ADV', 'ADJD', '$.']]),
    ]
)
def test_pos_tag(language, docs, expected):
    tagged_docs = pos_tag(docs, language)
    assert len(tagged_docs) == len(docs)

    for tdoc, exp_tags in zip(tagged_docs, expected):
        assert isinstance(tdoc, dict)
        assert tdoc['meta_pos'] == exp_tags

    tagged_docs = pos_tag(docs, language, doc_meta_key=None)
    assert len(tagged_docs) == len(docs)

    for tdoc, exp_tags in zip(tagged_docs, expected):
        assert isinstance(tdoc, list)
        assert tdoc == exp_tags


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
