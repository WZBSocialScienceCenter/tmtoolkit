"""
Preprocessing: Tests for ._docfuncs submodule.
"""

import math
from collections import Counter

import pytest
import numpy as np
from spacy.tokens import Doc
from scipy.sparse import isspmatrix_coo

from tmtoolkit.utils import empty_chararray
from tmtoolkit.preprocess._common import DEFAULT_LANGUAGE_MODELS
from tmtoolkit.preprocess._docfuncs import (
    init_for_language, tokenize, doc_tokens, doc_lengths, doc_labels, vocabulary, vocabulary_counts, doc_frequencies,
    ngrams, sparse_dtm
)
from ._testcorpora import corpora_sm


LANGUAGE_CODES = list(sorted(DEFAULT_LANGUAGE_MODELS.keys()))

CORPUS_MINI = {
    'ny': 'I live in New York.',
    'bln': 'I am in Berlin, but my flat is in Munich.',
    'empty': ''
}


@pytest.fixture(scope='module', params=LANGUAGE_CODES)
def nlp_all(request):
    return init_for_language(request.param)


@pytest.fixture()
def tokens_en():
    _init_lang('en')
    return tokenize(corpora_sm['en'])


@pytest.fixture()
def tokens_en_arrays():
    _init_lang('en')
    return doc_tokens(tokenize(corpora_sm['en']))


@pytest.fixture()
def tokens_en_lists():
    _init_lang('en')
    return doc_tokens(tokenize(corpora_sm['en']), to_lists=True)


@pytest.fixture()
def tokens_mini():
    _init_lang('en')
    return tokenize(CORPUS_MINI)


@pytest.fixture()
def tokens_mini_plain():
    _init_lang('en')
    return tokenize(CORPUS_MINI, as_spacy_docs=False)


def test_init_for_language():
    # note: this requires all language models to be installed

    with pytest.raises(ValueError, match='either .+ must be given'):
        init_for_language()

    with pytest.raises(ValueError, match='two-letter ISO 639-1 language code'):
        init_for_language('foo')

    with pytest.raises(ValueError, match='is not supported'):
        init_for_language('xx')

    with pytest.raises(OSError, match="Can't find model"):
        init_for_language(language_model='foobar')

    # try loading by language code / language model
    for i, (lang, model) in enumerate(DEFAULT_LANGUAGE_MODELS.items()):
        kwargs = {'language': lang} if i % 2 == 0 else {'language_model': model}

        nlp = init_for_language(**kwargs)

        assert nlp is not None
        assert str(nlp.__class__).startswith("<class 'spacy.lang.")
        assert len(nlp.pipeline) == 1 and nlp.pipeline[0][0] == 'tagger'   # default pipeline only with tagger

    # pass custom param to spacy.load: don't disable anything in the pipeline
    nlp = init_for_language(language='en', disable=[])
    assert len(nlp.pipeline) == 3   # tagger, parser, ner


@pytest.mark.parametrize(
    'testcase, docs, as_spacy_docs, doc_labels, doc_labels_fmt',
    [
        (1, [], True, None, None),
        (2, [''], True, None, None),
        (3, ['', ''], True, None, None),
        (4, ['Simple test.'], True, None, None),
        (5, ['Simple test.', 'Here comes another document.'], True, None, None),
        (6, ['Simple test.', 'Here comes another document.'], True, list('ab'), None),
        (7, ['Simple test.', 'Here comes another document.'], True, None, 'foo_{i0}'),
        (8, {'a': 'Simple test.', 'b': 'Here comes another document.'}, True, None, None),
        (9, ['Simple test.'], False, None, None),
        (10, ['Simple test.', 'Here comes another document.'], False, None, None),
        (11, ['Simple test.', 'Here comes another document.'], False, list('ab'), None),
        (12, ['Simple test.', 'Here comes another document.'], False, None, 'foo_{i0}'),
        (13, {'a': 'Simple test.', 'b': 'Here comes another document.'}, False, None, None),
    ]
)
def test_tokenize(testcase, docs, as_spacy_docs, doc_labels, doc_labels_fmt):
    if testcase == 1:
        _init_lang(None)  # no language initialized

        with pytest.raises(ValueError):
            tokenize(docs)

    _init_lang('en')

    kwargs = {'as_spacy_docs': as_spacy_docs}
    if doc_labels is not None:
        kwargs['doc_labels'] = doc_labels
    if doc_labels_fmt is not None:
        kwargs['doc_labels_fmt'] = doc_labels_fmt

    res = tokenize(docs, **kwargs)

    assert isinstance(res, list)
    assert len(res) == len(docs)
    if as_spacy_docs:
        assert all([isinstance(d, Doc) for d in res])
        assert all([isinstance(d._.label, str) for d in res])  # each doc. has a string label
        # Doc object text must be same as raw text
        assert all([d.text == txt for d, txt in zip(res, docs.values() if isinstance(docs, dict) else docs)])
    else:
        assert all([isinstance(d, list) for d in res])
        assert all([all([isinstance(t, str) for t in d]) for d in res if d])
        # token must occur in raw text
        assert all([[t in txt for t in d] for d, txt in zip(res, docs.values() if isinstance(docs, dict) else docs)])

    if testcase == 1:
        assert len(res) == 0
    elif testcase in {2, 3}:
        assert all([len(d) == 0 for d in res])
    elif testcase in {4, 9}:
        assert len(res) == 1
        assert len(res[0]) == 3
    elif testcase == 5:
        assert len(res) == 2
        assert all([d._.label == ('doc-%d' % (i+1)) for i, d in enumerate(res)])
    elif testcase == 6:
        assert len(res) == 2
        assert all([d._.label == lbl for d, lbl in zip(res, doc_labels)])
    elif testcase == 7:
        assert len(res) == 2
        assert all([d._.label == ('foo_%d' % i) for i, d in enumerate(res)])
    elif testcase == 8:
        assert len(res) == 2
        assert all([d._.label == lbl for d, lbl in zip(res, docs.keys())])
    elif testcase in {10, 11, 12, 13}:
        assert len(res) == 2
    else:
        raise RuntimeError('testcase not covered:', testcase)


@pytest.mark.parametrize(
    'as_spacy_docs', [True, False]
)
def test_tokenize_all_languages(nlp_all, as_spacy_docs):
    firstwords = {
        'en': ('NewsArticles-1', 'Disney'),
        'de': ('sample-9611', 'Sehr'),
        'fr': ('sample-1', "L'"),
        'es': ('sample-1', "El"),
        'pt': ('sample-1', "O"),
        'it': ('sample-1', "Bruce"),
        'nl': ('sample-1', "Cristiano"),
        'el': ('sample-1', "Ο"),
        'nb': ('sample-1', "Frøyningsfjelltromma"),
        'lt': ('sample-1', "Klondaiko"),
    }

    corpus = corpora_sm[nlp_all.lang]

    res = tokenize(corpus, as_spacy_docs=as_spacy_docs)
    assert isinstance(res, list)
    assert len(res) == len(corpus)

    if as_spacy_docs:
        assert all([isinstance(d, Doc) for d in res])
        assert all([isinstance(d._.label, str) for d in res])     # each doc. has a string label

        for doc in res:
            for arrname in ('tokens', 'mask'):
                assert arrname in doc.user_data and isinstance(doc.user_data[arrname], np.ndarray)

            assert np.issubdtype(doc.user_data['mask'].dtype, np.bool_)
            assert np.all(doc.user_data['mask'])
            assert np.issubdtype(doc.user_data['tokens'].dtype, np.unicode_)

        res_dict = {d._.label: d for d in res}
    else:
        assert all([isinstance(d, list) for d in res])
        assert all([all([isinstance(t, str) for t in d]) for d in res if d])

        res_dict = dict(zip(corpus.keys(), res))

    firstdoc_label, firstword = firstwords[nlp_all.lang]
    firstdoc = res_dict[firstdoc_label]
    assert len(firstdoc) > 0

    if as_spacy_docs:
        assert firstdoc[0].text == firstdoc.user_data['tokens'][0] == firstword

        for dl, txt in corpus.items():
            d = res_dict[dl]
            assert d.text == txt
    else:
        assert firstdoc[0] == firstword


def test_doc_tokens(tokens_mini, tokens_mini_plain):
    assert doc_tokens(tokens_mini_plain) == doc_tokens(tokens_mini_plain, to_lists=True) == tokens_mini_plain
    tokens_mini_arrays = [empty_chararray() if len(tok) == 0 else np.array(tok) for tok in tokens_mini_plain]
    doc_tok_arrays = doc_tokens(tokens_mini)
    doc_tok_lists =  doc_tokens(tokens_mini, to_lists=True)
    assert len(doc_tok_arrays) == len(doc_tok_lists) == len(tokens_mini_arrays)
    for tok_arr, tok_list, expected_arr, expected_list in zip(doc_tok_arrays, doc_tok_lists,
                                                              tokens_mini_arrays, tokens_mini_plain):
        assert isinstance(tok_arr, np.ndarray)
        assert np.issubdtype(tok_arr.dtype, np.unicode_)
        assert isinstance(tok_list, list)
        assert np.array_equal(tok_arr, expected_arr)
        assert tok_list == expected_list


def test_doc_lengths(tokens_en, tokens_en_arrays, tokens_en_lists):
    for tokens in (tokens_en, tokens_en_arrays, tokens_en_lists):
        res = doc_lengths(tokens)
        assert isinstance(res, list)
        assert len(res) == len(tokens)

        for n, d in zip(res, tokens):
            assert isinstance(n, int)
            assert n == len(d)
            if tokens is tokens_en and d._.label == 'empty':
                assert n == len(d) == 0


def test_doc_labels(tokens_en, tokens_en_arrays, tokens_en_lists):
    assert set(doc_labels(tokens_en)) == set(corpora_sm['en'])

    _init_lang('en')
    docs = tokenize(['test doc 1', 'test doc 2', 'test doc 3'], doc_labels=list('abc'))
    assert doc_labels(docs) == list('abc')

    with pytest.raises(ValueError):   # require spaCy docs
        doc_labels(tokens_en_arrays)
    with pytest.raises(ValueError):   # require spaCy docs
        doc_labels(tokens_en_lists)


@pytest.mark.parametrize('sort', [False, True])
def test_vocabulary(tokens_en, tokens_en_arrays, tokens_en_lists, sort):
    for tokens in (tokens_en, tokens_en_arrays, tokens_en_lists):
        res = vocabulary(tokens, sort=sort)

        if sort:
            assert isinstance(res, list)
            assert sorted(res) == res
        else:
            assert isinstance(res, set)

        for t in res:
            assert isinstance(t, str)
            assert any([t in dtok for dtok in doc_tokens(tokens)])


def test_vocabulary_counts(tokens_en, tokens_en_arrays, tokens_en_lists):
    for tokens in (tokens_en, tokens_en_arrays, tokens_en_lists):
        res = vocabulary_counts(tokens)
        n_tok = sum(doc_lengths(tokens))

        assert isinstance(res, Counter)
        assert set(res.keys()) == vocabulary(tokens)
        assert all([0 < n <= n_tok for n in res.values()])
        assert any([n > 1 for n in res.values()])


@pytest.mark.parametrize('proportions', [False, True])
def test_doc_frequencies(tokens_en, tokens_en_arrays, tokens_en_lists, proportions):
    for tokens in (tokens_en, tokens_en_arrays, tokens_en_lists):
        res = doc_frequencies(tokens, proportions=proportions)

        assert set(res.keys()) == vocabulary(tokens)

        if proportions:
            assert isinstance(res, dict)
            assert all([0 < v <= 1 for v in res.values()])
        else:
            assert isinstance(res, Counter)
            assert all([0 < v < len(tokens) for v in res.values()])
            assert any([v > 1 for v in res.values()])


def test_doc_frequencies_example():
    docs = [   # also works with simple token lists
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


@pytest.mark.parametrize('n', list(range(0, 5)))
def test_ngrams(tokens_en, tokens_en_arrays, tokens_en_lists, n):
    for tokens in (tokens_en, tokens_en_arrays, tokens_en_lists):
        if n < 2:
            with pytest.raises(ValueError):
                ngrams(tokens, n)
        else:
            docs_unigrams = doc_tokens(tokens)
            docs_ng = ngrams(tokens, n, join=False)
            docs_ng_joined = ngrams(tokens, n, join=True, join_str='')
            assert len(docs_ng) == len(docs_ng_joined) == len(docs_unigrams)

            for doc_ng, doc_ng_joined, tok in zip(docs_ng, docs_ng_joined, docs_unigrams):
                n_tok = len(tok)

                assert len(doc_ng_joined) == len(doc_ng)
                assert all([isinstance(x, list) for x in doc_ng])
                assert all([isinstance(x, str) for x in doc_ng_joined])

                if n_tok < n:
                    if n_tok == 0:
                        assert doc_ng == doc_ng_joined == []
                    else:
                        assert len(doc_ng) == len(doc_ng_joined) == 1
                        assert doc_ng == [tok]
                        assert doc_ng_joined == [''.join(tok)]
                else:
                    assert len(doc_ng) == len(doc_ng_joined) == n_tok - n + 1
                    assert all([len(g) == n for g in doc_ng])
                    assert all([''.join(g) == gj for g, gj in zip(doc_ng, doc_ng_joined)])

                    tokens_ = list(doc_ng[0])
                    if len(doc_ng) > 1:
                        tokens_ += [g[-1] for g in doc_ng[1:]]

                    if tokens is tokens_en_lists:
                        assert tokens_ == tok
                    else:
                        assert tokens_ == tok.tolist()

@pytest.mark.parametrize('pass_vocab', [False, True])
def test_sparse_dtm(tokens_en, tokens_en_arrays, tokens_en_lists, pass_vocab):
    emptydoc_index = [d._.label for d in tokens_en].index('empty')
    for tokens in (tokens_en, tokens_en_arrays, tokens_en_lists):
        if pass_vocab:
            vocab = vocabulary(tokens, sort=True)
            dtm = sparse_dtm(tokens, vocab)
        else:
            dtm, vocab = sparse_dtm(tokens)

        assert isspmatrix_coo(dtm)
        assert dtm.shape == (len(tokens), len(vocab))
        assert vocab == vocabulary(tokens, sort=True)

        doc_ntok = dtm.sum(axis=1)
        assert doc_ntok[emptydoc_index, 0] == 0


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


#%% helper functions

_nlp_instances_cache = {}

def _init_lang(code):
    """
    Helper function to load spaCy language model for language `code`. If `code` is None, reset (i.e. "unload"
    language model).

    Using this instead of pytest fixtures because the language model is only loaded when necessary, which speeds
    up tests.
    """
    from tmtoolkit.preprocess import _docfuncs

    if code is None:  # reset
        _docfuncs.nlp = None
    elif _docfuncs.nlp is None or _docfuncs.nlp.lang != code:  # init if necessary
        if code in _nlp_instances_cache.keys():
            _docfuncs.nlp = _nlp_instances_cache[code]
        else:
            init_for_language(code)
