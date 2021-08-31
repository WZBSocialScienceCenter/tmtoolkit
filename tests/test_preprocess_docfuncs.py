"""
Preprocessing: Tests for ._docfuncs submodule.
"""

from importlib.util import find_spec

import pytest

if not find_spec('spacy'):
    pytest.skip("skipping text processing tests: docfuncs", allow_module_level=True)

import math
import random
import string
from copy import deepcopy
from collections import Counter, OrderedDict

import decorator
from hypothesis import given, strategies as st, settings
import numpy as np
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
from scipy.sparse import isspmatrix_coo

from tmtoolkit.utils import empty_chararray, flatten_list
from tmtoolkit._pd_dt_compat import FRAME_TYPE, USE_DT, pd_dt_colnames
from tmtoolkit.preprocess._common import DEFAULT_LANGUAGE_MODELS, load_stopwords
from tmtoolkit.preprocess._docfuncs import (
    init_for_language, tokenize, doc_tokens, doc_lengths, doc_labels, vocabulary, vocabulary_counts, doc_frequencies,
    ngrams, sparse_dtm, kwic, kwic_table, glue_tokens, expand_compounds, clean_tokens, spacydoc_from_tokens,
    tokendocs2spacydocs, compact_documents, filter_tokens, remove_tokens, filter_tokens_with_kwic,
    filter_tokens_by_mask, remove_tokens_by_mask, filter_documents, remove_documents,
    filter_documents_by_name, remove_documents_by_name, filter_for_pos, pos_tag, pos_tags, remove_common_tokens,
    remove_uncommon_tokens, transform, to_lowercase, remove_chars, tokens2ids, ids2tokens, lemmatize,
    _filtered_doc_tokens
)
from ._testcorpora import corpora_sm
from ._testtools import strategy_lists_of_tokens


LANGUAGE_CODES = list(sorted(DEFAULT_LANGUAGE_MODELS.keys()))

CORPUS_MINI = OrderedDict([
    ('ny', 'I live in New York.'),
    ('bln', 'I am in Berlin, but my flat is in Munich.'),
    ('compounds', 'US-Student is reading an e-mail on eCommerce with CamelCase.'),
    ('empty', '')
])


def cleanup_after_test(fn):
    def wrapper(fn, *args, **kwargs):
        fn(*args, **kwargs)

        try:
            Token.remove_extension('testmeta')
        except ValueError: pass

    return decorator.decorator(wrapper, fn)


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


# when using fixtures with hypothesis, they must be in module scope and should be used "read-only"
@pytest.fixture(scope='module')
def module_tokens_en():
    _init_lang('en')
    return tokenize(corpora_sm['en'])


# when using fixtures with hypothesis, they must be in module scope and should be used "read-only"
@pytest.fixture(scope='module')
def module_tokens_en_arrays():
    _init_lang('en')
    return doc_tokens(tokenize(corpora_sm['en']))


# when using fixtures with hypothesis, they must be in module scope and should be used "read-only"
@pytest.fixture(scope='module')
def module_tokens_en_lists():
    _init_lang('en')
    return doc_tokens(tokenize(corpora_sm['en']), to_lists=True)


@pytest.fixture()
def tokens_mini():
    _init_lang('en')
    return tokenize(CORPUS_MINI)


@pytest.fixture()
def tokens_mini_arrays():
    _init_lang('en')

    # deliberately not using doc_tokens() here
    lists = tokenize(CORPUS_MINI, as_spacy_docs=False)
    return [np.array(doc) if doc else empty_chararray() for doc in lists]


@pytest.fixture()
def tokens_mini_lists():
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
        kwargs = {'language': lang} if i % 2 == 0 else {'language_model': model + '_sm'}

        nlp = init_for_language(**kwargs)

        assert nlp is not None
        assert str(nlp.__class__).startswith("<class 'spacy.lang.")
        if lang != 'ja':   # atm. tagger not available for Japanese
            assert len(nlp.pipeline) == 1 and nlp.pipeline[0][0] == 'tagger'   # default pipeline only with tagger

    # pass custom param to spacy.load: don't disable anything in the pipeline
    nlp = init_for_language(language='en', disable=[])
    assert len(nlp.pipeline) == 3   # tagger, parser, ner


@cleanup_after_test
@pytest.mark.parametrize(
    'testcase, docs, as_spacy_docs, doc_labels, doc_labels_fmt, enable_vectors',
    [
        (1, [], True, None, None, False),
        (2, [''], True, None, None, False),
        (3, ['', ''], True, None, None, False),
        (4, ['Simple test.'], True, None, None, False),
        (5, ['Simple test.', 'Here comes another document.'], True, None, None, False),
        (6, ['Simple test.', 'Here comes another document.'], True, None, None, True),
        (7, ['Simple test.', 'Here comes another document.'], True, list('ab'), None, False),
        (8, ['Simple test.', 'Here comes another document.'], True, None, 'foo_{i0}', False),
        (9, {'a': 'Simple test.', 'b': 'Here comes another document.'}, True, None, None, False),
        (10, ['Simple test.'], False, None, None, False),
        (11, ['Simple test.', 'Here comes another document.'], False, None, None, False),
        (12, ['Simple test.', 'Here comes another document.'], False, list('ab'), None, False),
        (13, ['Simple test.', 'Here comes another document.'], False, None, 'foo_{i0}', False),
        (14, {'a': 'Simple test.', 'b': 'Here comes another document.'}, False, None, None, False),
    ]
)
def test_tokenize(testcase, docs, as_spacy_docs, doc_labels, doc_labels_fmt, enable_vectors):
    if testcase == 1:
        _init_lang(None)  # no language initialized

        with pytest.raises(ValueError):
            tokenize(docs)

    if enable_vectors:
        _init_lang('en')
    else:
        init_for_language(language_model='en_core_web_md')

    kwargs = {'as_spacy_docs': as_spacy_docs, 'enable_vectors': enable_vectors}
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
    elif testcase in {4, 10}:
        assert len(res) == 1
        assert len(res[0]) == 3
    elif testcase == 5:
        assert len(res) == 2
        assert all([d._.label == ('doc-%d' % (i+1)) for i, d in enumerate(res)])
    elif testcase == 6:
        assert len(res) == 2
        assert all([d._.label == ('doc-%d' % (i+1)) for i, d in enumerate(res)])
        assert all([d.has_vector for d in res])
    elif testcase == 7:
        assert len(res) == 2
        assert all([d._.label == lbl for d, lbl in zip(res, doc_labels)])
    elif testcase == 8:
        assert len(res) == 2
        assert all([d._.label == ('foo_%d' % i) for i, d in enumerate(res)])
    elif testcase == 9:
        assert len(res) == 2
        assert all([d._.label == lbl for d, lbl in zip(res, docs.keys())])
    elif testcase in {11, 12, 13, 14}:
        assert len(res) == 2
    else:
        raise RuntimeError('testcase not covered:', testcase)


@cleanup_after_test
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
        'ja': ('sample-1', "アップル"),
        'zh': ('sample-1', "作为"),
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

            assert doc.user_data['mask'].dtype.kind == 'b'
            assert np.all(doc.user_data['mask'])
            assert doc.user_data['tokens'].dtype.kind == 'U'

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


@cleanup_after_test
def test_doc_tokens(tokens_mini, tokens_mini_arrays, tokens_mini_lists):
    doc_tok_arrays = doc_tokens(tokens_mini)
    doc_tok_lists = doc_tokens(tokens_mini, to_lists=True)
    assert len(doc_tok_arrays) == len(doc_tok_lists) == len(tokens_mini_arrays)
    for tok_arr, tok_list, expected_arr, expected_list in zip(doc_tok_arrays, doc_tok_lists,
                                                              tokens_mini_arrays, tokens_mini_lists):
        assert isinstance(tok_arr, np.ndarray)
        assert tok_arr.dtype.kind == 'U'
        assert isinstance(tok_list, list)
        assert np.array_equal(tok_arr, expected_arr)
        assert tok_list == expected_list


@cleanup_after_test
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


@cleanup_after_test
def test_doc_labels(tokens_en, tokens_en_arrays, tokens_en_lists):
    assert set(doc_labels(tokens_en)) == set(corpora_sm['en'])

    _init_lang('en')
    docs = tokenize(['test doc 1', 'test doc 2', 'test doc 3'], doc_labels=list('abc'))
    assert doc_labels(docs) == list('abc')

    with pytest.raises(ValueError):   # require spaCy docs
        doc_labels(tokens_en_arrays)
    with pytest.raises(ValueError):   # require spaCy docs
        doc_labels(tokens_en_lists)


@cleanup_after_test
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


@cleanup_after_test
def test_vocabulary_counts(tokens_en, tokens_en_arrays, tokens_en_lists):
    for tokens in (tokens_en, tokens_en_arrays, tokens_en_lists):
        res = vocabulary_counts(tokens)
        n_tok = sum(doc_lengths(tokens))

        assert isinstance(res, Counter)
        assert set(res.keys()) == vocabulary(tokens)
        assert all([0 < n <= n_tok for n in res.values()])
        assert any([n > 1 for n in res.values()])


@cleanup_after_test
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


@cleanup_after_test
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


@cleanup_after_test
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


@cleanup_after_test
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


@cleanup_after_test
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


@cleanup_after_test
@given(search_term_exists=st.booleans(), context_size=st.integers(1, 5), as_dict=st.booleans(),
       non_empty=st.booleans(), glue=st.booleans(), highlight_keyword=st.booleans())
def test_kwic(module_tokens_en, module_tokens_en_arrays, module_tokens_en_lists,
              search_term_exists, context_size, as_dict, non_empty, glue, highlight_keyword):
    for tokens in (module_tokens_en, module_tokens_en_arrays, module_tokens_en_lists):
        vocab = list(vocabulary(tokens))

        if search_term_exists and len(vocab) > 0:
            s = random.choice(vocab)
        else:
            s = 'thisdoesnotexist'

        glue_arg = ' ' if glue else None
        highlight_arg = '*' if highlight_keyword else None

        kwic_kwargs = dict(context_size=context_size, non_empty=non_empty, as_dict=as_dict, glue=glue_arg,
                           highlight_keyword=highlight_arg)

        if as_dict and tokens is not module_tokens_en:
            with pytest.raises(ValueError):
                kwic(tokens, s, **kwic_kwargs)
            return
        else:
            res = kwic(tokens, s, **kwic_kwargs)

        assert isinstance(res, dict if as_dict else list)

        if as_dict:
            if non_empty:
                assert all(k in doc_labels(tokens) for k in res.keys())
            else:
                assert list(res.keys()) == doc_labels(tokens)
            res = res.values()

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


@cleanup_after_test
def test_kwic_example(tokens_mini, tokens_mini_arrays, tokens_mini_lists):
    for tokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        res = kwic(tokens, 'in', context_size=1)
        assert res == [
            [['live', 'in', 'New']],
            [['am', 'in', 'Berlin'], ['is', 'in', 'Munich']],
            [],
            [],
        ]

        if tokens is tokens_mini:
            res = kwic(tokens, 'in', context_size=1, as_dict=True)
            assert res == {
                'ny': [['live', 'in', 'New']],
                'bln': [['am', 'in', 'Berlin'], ['is', 'in', 'Munich']],
                'compounds': [],
                'empty': [],
            }
        else:
            with pytest.raises(ValueError):
                kwic(tokens, 'in', context_size=1, as_dict=True)

        res = kwic(tokens, 'in', context_size=1, non_empty=True)
        assert res == [
            [['live', 'in', 'New']],
            [['am', 'in', 'Berlin'], ['is', 'in', 'Munich']],
        ]

        res = kwic(tokens, 'in', context_size=1, non_empty=True, glue=' ')
        assert res == [
            ['live in New'],
            ['am in Berlin', 'is in Munich']
        ]

        res = kwic(tokens, 'in', context_size=1, non_empty=True, glue=' ', highlight_keyword='*')
        assert res == [
            ['live *in* New'],
            ['am *in* Berlin', 'is *in* Munich']
        ]

        res = kwic(tokens, 'in', context_size=1, non_empty=True, highlight_keyword='*')
        assert res == [
            [['live', '*in*', 'New']],
            [['am', '*in*', 'Berlin'], ['is', '*in*', 'Munich']],
        ]


@cleanup_after_test
@given(search_term_exists=st.booleans(), context_size=st.integers(1, 5))
def test_kwic_table(module_tokens_en, module_tokens_en_arrays, module_tokens_en_lists,
                    context_size, search_term_exists):
    for tokens in (module_tokens_en, module_tokens_en_arrays, module_tokens_en_lists):
        vocab = list(vocabulary(tokens))

        if search_term_exists and len(vocab) > 0:
            s = random.choice(vocab)
        else:
            s = 'thisdoesnotexist'

        if tokens is module_tokens_en:
            res = kwic_table(tokens, s, context_size=context_size)
        else:
            with pytest.raises(ValueError):
                kwic_table(tokens, s, context_size=context_size)
            return

        assert isinstance(res, FRAME_TYPE)
        assert pd_dt_colnames(res) == ['doc', 'context', 'kwic']

        if s in vocab:
            assert res.shape[0] > 0

            if USE_DT:
                kwic_col = res[:, 'kwic'].to_list()[0]
                docs_col = res[:, 'doc'].to_list()[0]
            else:
                kwic_col = res.loc[:, 'kwic'].to_list()
                docs_col = res.loc[:, 'doc'].to_list()

            assert all(dl in doc_labels(tokens) for dl in set(docs_col))

            for kwic_match in kwic_col:
                assert kwic_match.count('*') >= 2
        else:
            assert res.shape[0] == 0


@cleanup_after_test
def test_glue_tokens_example(tokens_mini, tokens_mini_arrays, tokens_mini_lists):
    for testtokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        tokens = doc_tokens(testtokens)
        res = glue_tokens(testtokens, ('New', 'York'))
        tokens_ = doc_tokens(res)
        assert isinstance(res, list)
        assert len(res) == len(testtokens) == len(tokens_)

        if testtokens is tokens_mini:
            assert all([d1 is d2 for d1, d2 in zip(res, testtokens)])   # modifies in-place
        else:
            assert all([d1 is not d2 for d1, d2 in zip(res, testtokens)])  # does *not* modify in-place

        for i, (d_, d) in enumerate(zip(tokens_, tokens)):
            if testtokens is not tokens_mini_lists:
                d = d.tolist()
                if testtokens is tokens_mini:
                    d_ = d_.tolist()

            if i == 0:
                assert d_ == ['I', 'live', 'in', 'New_York', '.']
            else:
                assert d_ == d

        res, glued = glue_tokens(res, ('in', '*'), glue='/', match_type='glob', return_glued_tokens=True)
        tokens_ = doc_tokens(res)

        if testtokens is tokens_mini:
            assert all([d1 is d2 for d1, d2 in zip(res, testtokens)])   # modifies in-place
        else:
            assert all([d1 is not d2 for d1, d2 in zip(res, testtokens)])  # does *not* modify in-place

        assert glued == {'in/New_York', 'in/Berlin', 'in/Munich'}

        for i, (d_, d) in enumerate(zip(tokens_, tokens)):
            if testtokens is not tokens_mini_lists:
                d = d.tolist()
                if testtokens is tokens_mini:
                    d_ = d_.tolist()

            if i == 0:
                assert d_ == ['I', 'live', 'in/New_York', '.']
            elif i == 1:
                assert d_ == ['I', 'am', 'in/Berlin', ',', 'but', 'my', 'flat', 'is', 'in/Munich', '.']
            else:
                assert d_ == d


@cleanup_after_test
@pytest.mark.parametrize(
    'testcase, split_chars, split_on_len, split_on_casechange',
    [
        (1, ['-'], 2, False),
        (2, ['-'], 2, True),
        (3, [], 2, True),
    ]
)
def test_expand_compounds(tokens_mini, tokens_mini_arrays, tokens_mini_lists,
                          testcase, split_chars, split_on_len, split_on_casechange):
    for testtokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        res = expand_compounds(testtokens, split_chars=split_chars, split_on_len=split_on_len,
                               split_on_casechange=split_on_casechange)
        assert isinstance(res, list)
        assert len(res) == len(testtokens)

        got_compounds = False
        for expdoc, origdoc in zip(res, testtokens):
            if len(origdoc) == 0:
                assert len(expdoc) == 0
            else:
                assert len(expdoc) >= len(origdoc)

            if 'CamelCase' in vocabulary([origdoc]):   # doc with compounds
                got_compounds = True
                expdoc_tokens = _filtered_doc_tokens(expdoc, as_list=True)
                if testcase == 1:
                    assert expdoc_tokens == ['US', '-', 'Student', 'is', 'reading', 'an', 'e', '-', 'mail', 'on',
                                             'eCommerce', 'with', 'CamelCase', '.']
                elif testcase in {2, 3}:
                    assert expdoc_tokens == ['US', '-', 'Student', 'is', 'reading', 'an', 'e', '-', 'mail', 'on',
                                             'eCommerce', 'with', 'Camel', 'Case', '.']
            else:  # doc without compounds
                if testtokens is tokens_mini_arrays:
                    assert np.array_equal(expdoc, origdoc)
                else:
                    assert _filtered_doc_tokens(expdoc, as_list=True) == _filtered_doc_tokens(origdoc, as_list=True)

        assert got_compounds


@cleanup_after_test
@pytest.mark.parametrize(
    'docs, expected',
    [
        ([], []),
        ([['']], [['']]),
        ([[''], []], [[''], []]),
        ([['An', 'US-Student', '.']], [['An', 'US', 'Student', '.']]),
    ]
)
def test_expand_compounds_examples(docs, expected):
    assert expand_compounds(docs) == expected


@cleanup_after_test
@settings(deadline=1000)
@given(docs=strategy_lists_of_tokens(string.printable),
       docs_type=st.integers(0, 2),
       remove_punct=st.integers(0, 2),
       remove_stopwords=st.integers(0, 2),
       remove_empty=st.booleans(),
       remove_shorter_than=st.integers(-2, 5),
       remove_longer_than=st.integers(-2, 10),
       remove_numbers=st.booleans())
def test_clean_tokens(docs, docs_type, remove_punct, remove_stopwords, remove_empty,
                      remove_shorter_than, remove_longer_than, remove_numbers):
    _init_lang('en')

    docs_as_tokens = doc_tokens(docs, to_lists=True)
    docs_vocab = list(vocabulary(docs_as_tokens))

    if docs_type == 1:     # arrays
        docs = [np.array(d) if d else empty_chararray() for d in docs]
    elif docs_type == 2:   # spaCy docs
        docs = tokendocs2spacydocs(docs)

    if remove_punct == 2:
        remove_punct = np.random.choice(list(string.punctuation), 5, replace=False).tolist()
    else:
        remove_punct = bool(remove_punct)

    if remove_stopwords == 2 and docs_vocab:
        remove_stopwords = np.random.choice(docs_vocab, 5, replace=True).tolist()
    else:
        remove_stopwords = bool(remove_stopwords)

    if remove_shorter_than == -2:
        remove_shorter_than = None

    if remove_longer_than == -2:
        remove_longer_than = None

    if remove_shorter_than == -1 or remove_longer_than == -1:
        with pytest.raises(ValueError):
            clean_tokens(docs, remove_punct=remove_punct, remove_stopwords=remove_stopwords,
                         remove_empty=remove_empty, remove_shorter_than=remove_shorter_than,
                         remove_longer_than=remove_longer_than, remove_numbers=remove_numbers)
    else:
        docs_ = clean_tokens(docs, remove_punct=remove_punct, remove_stopwords=remove_stopwords,
                             remove_empty=remove_empty, remove_shorter_than=remove_shorter_than,
                             remove_longer_than=remove_longer_than, remove_numbers=remove_numbers)

        if docs_type == 2:
            docs_ = compact_documents(docs_)

        blacklist = set()

        if isinstance(remove_punct, list):
            blacklist.update(remove_punct)
        elif remove_punct is True:
            blacklist.update(string.punctuation)

        if isinstance(remove_stopwords, list):
            blacklist.update(remove_stopwords)
        elif remove_stopwords is True:
            blacklist.update(load_stopwords('en'))

        if remove_empty:
            blacklist.update('')

        assert len(docs) == len(docs_)

        for i, (dtok, dtok_) in enumerate(zip(docs, docs_)):
            dtok_ = _filtered_doc_tokens(dtok_, as_list=True)
            assert len(dtok) >= len(dtok_)
            del dtok

            if remove_punct is not True or docs_type != 2:      # clean_tokens uses is_punct attrib. when using spaCy
                assert all([w not in dtok_ for w in blacklist])

            tok_lengths = np.array(list(map(len, dtok_)))

            if remove_shorter_than is not None:
                assert np.all(tok_lengths >= remove_shorter_than)

            if remove_longer_than is not None:
                assert np.all(tok_lengths <= remove_longer_than)

            if remove_numbers and len(dtok_) > 0 and docs_type != 2:  # clean_tokens uses like_num attrib for spaCy docs
                assert not np.any(np.char.isnumeric(dtok_))


@cleanup_after_test
@pytest.mark.parametrize('search_patterns, by_meta, match_type, ignore_case, inverse, expected_docs', [
    ('in', False, 'exact', False, False, [['in'], ['in', 'in'], [], []]),
    (['New', 'Berlin'], False, 'exact', False, False, [['New'], ['Berlin'], [], []]),
    ('IN', False, 'exact', False, False, [[], [], [], []]),
    ('IN', False, 'exact', True, False, [['in'], ['in', 'in'], [], []]),
    ('bar', True, 'exact', False, False, [['I'], ['I'], ['US'], []]),
    (r'\w+am', False, 'regex', False, False, [[], [], ['CamelCase'], []]),
    (r'\w+AM', False, 'regex', True, False, [[], [], ['CamelCase'], []]),
    (r'^[a-z]+', False, 'regex', False, True, [['I', 'New', 'York', '.'],
                                               ['I', 'Berlin', ',', 'Munich', '.'],
                                               ['US', '-', 'Student', '-', 'CamelCase', '.'], []]),
    (r'^b', True, 'regex', False, False, [['I'], ['I'], ['US'], []]),
    ('*in', False, 'glob', False, False, [['in'], ['in', 'Berlin', 'in'], ['reading'], []]),
    ('*IN', False, 'glob', True, False, [['in'], ['in', 'Berlin', 'in'], ['reading'], []]),
])
def test_filter_tokens(tokens_mini, tokens_mini_arrays, tokens_mini_lists, search_patterns, by_meta, match_type,
                       ignore_case, inverse, expected_docs):
    kwargs = {'match_type': match_type, 'ignore_case': ignore_case, 'inverse': inverse}

    if by_meta:
        for d in tokens_mini:
            if len(d) > 0:
                d[0]._.testmeta = 'bar'
        kwargs['by_meta'] = 'testmeta'

    # check empty
    assert filter_tokens([], search_patterns, **kwargs) == []

    # check non-empty
    for testtokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        if by_meta and testtokens is not tokens_mini:
            with pytest.raises(ValueError):  # requires spacy docs
                filter_tokens(testtokens, search_patterns, **kwargs)
        else:
            res = filter_tokens(testtokens, search_patterns, **kwargs)
            assert isinstance(res, list)
            assert len(res) == len(testtokens)

            restokens = doc_tokens(res, to_lists=True)
            assert restokens == expected_docs

            if inverse:
                assert restokens == doc_tokens(remove_tokens(testtokens, search_patterns,
                                                             match_type=match_type, ignore_case=ignore_case),
                                               to_lists=True)


@cleanup_after_test
@given(docs=strategy_lists_of_tokens(string.printable),
       docs_type=st.integers(0, 2),
       search_term_exists=st.booleans(),
       context_size=st.integers(0, 5),
       invert=st.booleans())
def test_filter_tokens_with_kwic_hypothesis(docs, docs_type, search_term_exists, context_size, invert):
    vocab = list(vocabulary(docs) - {''})

    if docs_type == 1:     # arrays
        docs = [np.array(d) if d else empty_chararray() for d in docs]
    elif docs_type == 2:   # spaCy docs
        docs = tokendocs2spacydocs(docs)

    if search_term_exists and len(vocab) > 0:
        s = random.choice(vocab)
    else:
        s = 'thisdoesnotexist'

    res = filter_tokens_with_kwic(docs, s, context_size=context_size, inverse=invert)
    res_filter_tokens = filter_tokens(docs, s, inverse=invert)
    res_kwic = kwic(docs, s, context_size=context_size, inverse=invert)

    assert isinstance(res, list)
    assert len(res) == len(docs) == len(res_filter_tokens) == len(res_kwic)

    for d, d_, d_ft, d_kwic in zip(docs, res, res_filter_tokens, res_kwic):
        if docs_type == 0:
            assert isinstance(d_, list)
        elif docs_type == 1:
            assert isinstance(d_, np.ndarray)
            d = d.tolist()
            d_ = d_.tolist()
            d_ft = d_ft.tolist()
        else:
            assert docs_type == 2
            assert isinstance(d_, Doc)
            d = _filtered_doc_tokens(d, as_list=True)
            d_ = _filtered_doc_tokens(d_, as_list=True)
            d_ft = _filtered_doc_tokens(d_ft, as_list=True)

        assert len(d_) <= len(d)

        if context_size == 0:
            assert d_ == d_ft
        else:
            assert all([t in d for t in d_])
            assert len(d_kwic) == len(d_ft)

            if len(d_) > 0 and len(vocab) > 0 and not invert:
                assert (s in d_) == search_term_exists

            if not invert:
                d_kwic_flat = flatten_list(d_kwic)
                assert set(d_kwic_flat) == set(d_)


@cleanup_after_test
@pytest.mark.parametrize('search_patterns, context_size, invert, expected_docs', [
    ('in', 1, False, [
     ['live', 'in', 'New'],
     ['am', 'in', 'Berlin', 'is', 'in', 'Munich'],
     [],
     []
    ]),
    ('in', 2, False, [
     ['I', 'live', 'in', 'New', 'York'],
     ['I', 'am', 'in', 'Berlin', ',', 'flat', 'is', 'in', 'Munich', '.'],
     [],
     []
    ]),
    ('is', 3, True, [
     ['I', 'live', 'in', 'New', 'York', '.'],
     ['I', 'am', 'in', 'Berlin', ','],
     ['-', 'mail', 'on', 'eCommerce', 'with', 'CamelCase', '.'],
     []
    ]),
    (['New', 'Berlin'], 1, False, [
        ['in', 'New', 'York'],
        ['in', 'Berlin', ','],
        [],
        []
    ]),
])
def test_filter_tokens_with_kwic_example(tokens_mini, tokens_mini_arrays, tokens_mini_lists, search_patterns,
                                         context_size, invert, expected_docs):
    kwargs = {
        'context_size': context_size,
        'inverse': invert
    }

    # check empty
    assert filter_tokens_with_kwic([], search_patterns, **kwargs) == []

    # check non-empty
    for testtokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        res = filter_tokens_with_kwic(testtokens, search_patterns, **kwargs)

        assert isinstance(res, list)
        assert len(res) == len(testtokens)

        restokens = doc_tokens(res, to_lists=True)
        assert restokens == expected_docs


@cleanup_after_test
@given(docs=strategy_lists_of_tokens(string.printable),
       docs_type=st.integers(0, 2),
       inverse=st.booleans())
@settings(deadline=1000)
def test_filter_tokens_by_mask(docs, docs_type, inverse):
    docs_copy = docs
    if docs_type == 1:     # arrays
        docs = [np.array(d) if d else empty_chararray() for d in docs]
    elif docs_type == 2:   # spaCy docs
        docs = tokendocs2spacydocs(docs)

        if inverse:
            from copy import deepcopy
            docs_copy = deepcopy(docs)

    mask = [[random.choice([False, True]) for _ in range(n)] for n in map(len, docs)]

    res = filter_tokens_by_mask(docs, mask, inverse=inverse)
    assert isinstance(res, list)
    assert len(res) == len(docs)

    res = doc_tokens(res, to_lists=True)

    if inverse:
        res2 = doc_tokens(remove_tokens_by_mask(docs_copy, mask), to_lists=True)
        assert res == res2

    for i, (dtok, dmsk) in enumerate(zip(docs, mask)):
        n = len(dmsk) - sum(dmsk) if inverse else sum(dmsk)
        assert len(res[i]) == n


@cleanup_after_test
@pytest.mark.parametrize(
    'search_patterns, by_meta, matches_threshold, match_type, ignore_case, inverse_result, inverse_matches, '
    'expected_doc_indices',
    [
        ('in', False, 1, 'exact', False, False, False, (0, 1)),
        ('bar', True, 1, 'exact', False, False, False, (0, )),
        ('bar', True, 2, 'exact', False, False, False, ()),
        (r'^Camel', False, 1, 'regex', False, False, False, (2, )),
        ('*in*', False, 1, 'glob', False, False, False, (0, 1, 2)),
        ('in', False, 2, 'exact', False, False, False, (1, )),
        (['New', 'Berlin'], False, 1, 'exact', False, False, False, (0, 1)),
        ('i', False, 1, 'exact', True, False, False, (0, 1)),
        ('in', False, 1, 'exact', False, True, False, (2, 3)),
        ('in', False, 6, 'exact', False, False, True, (1, 2)),
    ]
)
def test_filter_documents(tokens_mini, tokens_mini_arrays, tokens_mini_lists, search_patterns, by_meta,
                          matches_threshold, match_type, ignore_case, inverse_result, inverse_matches,
                          expected_doc_indices):
    kwargs = {
        'matches_threshold': matches_threshold,
        'match_type': match_type,
        'ignore_case': ignore_case,
        'inverse_result': inverse_result,
        'inverse_matches': inverse_matches
    }

    expected_docs = [tokens_mini_lists[i] for i in expected_doc_indices]

    if by_meta:
        tokens_mini[0][0]._.testmeta = 'bar'
        kwargs['by_meta'] = 'testmeta'

    # check empty
    assert filter_documents([], search_patterns, **kwargs) == []

    # check non-empty
    for testtokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        if by_meta and testtokens is not tokens_mini:
            with pytest.raises(ValueError):  # requires spacy docs
                filter_documents(testtokens, search_patterns, **kwargs)
        else:
            res = filter_documents(testtokens, search_patterns, **kwargs)
            assert isinstance(res, list)
            assert len(res) == len(expected_doc_indices) == len(expected_docs)

            result_docs = doc_tokens(res, to_lists=True)
            assert result_docs == expected_docs

            if inverse_result:
                kwargs_copy = kwargs.copy()
                del kwargs_copy['inverse_result']
                assert result_docs == doc_tokens(remove_documents(testtokens, search_patterns, **kwargs_copy),
                                                 to_lists=True)


@cleanup_after_test
@pytest.mark.parametrize(
    'name_patterns, match_type, ignore_case, inverse, expected_doc_indices',
    [
        ('ny', 'exact', False, False, (0, )),
        (r'\w+n\w+', 'regex', False, False, (2, )),
        ('comp*', 'glob', False, False, (2, )),
        (['ny', 'bln'], 'exact', False, False, (0, 1)),
        (['ny', 'bln'], 'exact', False, True, (2, 3)),
        (['ny', 'bln'], 'exact', False, True, (2, 3)),
        ([], 'exact', False, False, None),  # raise exception
        ('NY', 'exact', False, False, ()),
        ('NY', 'exact', True, False, (0, )),
        ('NY', 'exact', True, True, (1, 2, 3)),
    ]
)
def test_filter_documents_by_name(tokens_mini, tokens_mini_arrays, tokens_mini_lists, name_patterns, match_type,
                                  ignore_case, inverse, expected_doc_indices):
    kwargs = {
        'match_type': match_type,
        'ignore_case': ignore_case,
        'inverse': inverse
    }

    if expected_doc_indices is None:
        with pytest.raises(ValueError):
            filter_documents_by_name(tokens_mini, name_patterns, **kwargs)
        return

    expected_docs = [tokens_mini_lists[i] for i in expected_doc_indices]

    # check empty
    assert filter_documents_by_name([], name_patterns, **kwargs) == []

    # check non-empty
    for testtokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        kwargs_copy = kwargs.copy()

        if testtokens is not tokens_mini:
            kwargs_copy['labels'] = list(CORPUS_MINI.keys())

        res = filter_documents_by_name(testtokens, name_patterns, **kwargs_copy)

        assert isinstance(res, list)
        assert len(res) == len(expected_doc_indices) == len(expected_docs)

        result_docs = doc_tokens(res, to_lists=True)
        assert result_docs == expected_docs

        if inverse:
            del kwargs_copy['inverse']
            assert result_docs == doc_tokens(remove_documents_by_name(testtokens, name_patterns, **kwargs_copy),
                                             to_lists=True)


@cleanup_after_test
@pytest.mark.parametrize(
    'required_pos, simplify_pos, pos_attrib, inverse, expected_docs',
    [
        ('N', True, 'pos_', False, [
            ['New', 'York'],
            ['Berlin', 'flat', 'Munich'],
            ['US', 'Student', 'e', '-', 'mail', 'eCommerce', 'CamelCase'],
            []
        ]),
        (['N', 'V'], True, 'pos_', False, [
            ['live', 'New', 'York'],
            ['Berlin', 'flat', 'Munich'],
            ['US', 'Student', 'reading', 'e', '-', 'mail', 'eCommerce', 'CamelCase'],
            []
        ]),
        ('PROPN', False, 'pos_', False, [
            ['New', 'York'],
            ['Berlin', 'Munich'],
            ['US', 'Student', 'eCommerce', 'CamelCase'],
            []
        ]),
        (96, False, 'pos', False, [
            ['New', 'York'],
            ['Berlin', 'Munich'],
            ['US', 'Student', 'eCommerce', 'CamelCase'],
            []
        ]),
        ([95, 96], False, 'pos', False, [
            ['I', 'New', 'York'],
            ['I', 'Berlin', 'Munich'],
            ['US', 'Student', 'eCommerce', 'CamelCase'],
            []
        ]),
        (['N', 'V', 'ADJ', 'ADV'], True, 'pos_', True, [
            ['I', 'in', '.'],
            ['I', 'am', 'in', ',', 'but', 'my', 'is', 'in', '.'],
            ['-', 'is', 'an', 'on', 'with', '.'],
            []
        ]),
    ]
)
def test_filter_for_pos(tokens_mini, tokens_mini_arrays, tokens_mini_lists, required_pos, simplify_pos, pos_attrib,
                        inverse, expected_docs):
    kwargs = {
        'simplify_pos': simplify_pos,
        'pos_attrib': pos_attrib,
        'inverse': inverse
    }

    # check empty
    assert filter_for_pos([], required_pos, **kwargs) == []

    # check non-empty
    for testtokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        if testtokens is tokens_mini:
            _init_lang('en')
            pos_tag(testtokens)
            res = filter_for_pos(testtokens, required_pos, **kwargs)

            assert isinstance(res, list)

            result_docs = doc_tokens(res, to_lists=True)
            assert result_docs == expected_docs
        else:
            with pytest.raises(ValueError):
                filter_for_pos(testtokens, required_pos, **kwargs)    # requires spaCy docs


@cleanup_after_test
def test_filter_multiple(tokens_mini, tokens_mini_arrays, tokens_mini_lists):
    for testtokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        if testtokens is tokens_mini:
            _init_lang('en')
            pos_tag(testtokens)
            res = filter_for_pos(testtokens, ['N', 'V'])
            res = filter_documents(res, ['New', 'Berlin'])
            res = filter_tokens(res, 'New')
            assert doc_tokens(res, to_lists=True) == doc_tokens(compact_documents(res), to_lists=True) == [
                ['New'], []
            ]
        else:
            res = filter_documents(testtokens, ['New', 'Berlin'])
            res = filter_tokens(res, 'New')
            assert doc_tokens(res, to_lists=True) == [
                ['New'], []
            ]


@cleanup_after_test
@pytest.mark.parametrize(
    'docs, common, thresh, absolute, expected_docs',
    [
        ([], True, 0.75, False, []),
        ([[]], True, 0.75, False, [[]]),
        ([['a']] * 10, True, 0.9, False, [[]] * 10),
        ([['a']] * 9 + [['b']], True, 0.9, False, [[]] * 9 + [['b']]),
        ([['a']] * 9 + [['b']], False, 1, True, [['a']] * 9 + [[]]),
    ]
)
def test_remove_common_uncommon_tokens(docs, common, thresh, absolute, expected_docs):
    if common:
        fn = remove_common_tokens
    else:
        fn = remove_uncommon_tokens

    for docs_type in (0, 1, 2):
        if docs_type == 1:  # arrays
            docs = [np.array(d) if d else empty_chararray() for d in docs]
        elif docs_type == 2:  # spaCy docs
            docs = tokendocs2spacydocs(docs)

        res = fn(docs, df_threshold=thresh, absolute=absolute)

        assert isinstance(res, list)
        assert len(res) == len(docs)
        assert doc_tokens(res, to_lists=True) == expected_docs


@cleanup_after_test
@given(
    docs=strategy_lists_of_tokens(string.printable, min_size=1),
    docs_type=st.integers(0, 2)
)
@settings(deadline=1000)
def test_transform_and_to_lowercase(docs, docs_type):
    expected = [[t.lower() for t in d] for d in docs]

    if docs_type == 1:  # arrays
        docs = [np.array(d) if d else empty_chararray() for d in docs]
    elif docs_type == 2:  # spaCy docs
        labels = ['doc%d' % i for i in range(len(docs))]
        docs = tokendocs2spacydocs(docs, doc_labels=labels)

    res1 = transform(docs, str.lower)
    assert isinstance(res1, list)
    assert len(res1) == len(docs)
    if docs_type == 2:
        assert doc_labels(res1) == labels

    res2 = to_lowercase(docs)
    assert isinstance(res2, list)
    assert len(res2) == len(docs)
    if docs_type == 2:
        assert doc_labels(res2) == labels

    if len(docs) > 0:
        assert type(next(iter(docs))) is type(next(iter(res1)))
        assert type(next(iter(docs))) is type(next(iter(res2)))

    res1_tokens = doc_tokens(res1, to_lists=True)
    res2_tokens = doc_tokens(res2, to_lists=True)
    assert res1_tokens == expected
    assert res2_tokens == expected

    def repeat_token(t, k):
        return t * k

    res = transform(docs, repeat_token, k=3)

    assert len(res) == len(docs)
    for dtok_, dtok in zip(res, docs):
        assert len(dtok_) == len(dtok)
        for t_, t in zip(dtok_, dtok):
            assert len(t_) == 3 * len(t)


@cleanup_after_test
@given(
    docs=strategy_lists_of_tokens(min_size=1),
    docs_type=st.integers(0, 2),
    chars=st.lists(st.characters())
)
@settings(deadline=1000)
def test_remove_chars(docs, docs_type, chars):
    if docs_type == 1:  # arrays
        docs = [np.array(d) if d else empty_chararray() for d in docs]
    elif docs_type == 2:  # spaCy docs
        labels = ['doc%d' % i for i in range(len(docs))]
        docs = tokendocs2spacydocs(docs, doc_labels=labels)

    if len(chars) == 0:
        with pytest.raises(ValueError):
            remove_chars(docs, chars)
    else:
        docs_ = remove_chars(docs, chars)
        assert isinstance(docs_, list)
        assert len(docs_) == len(docs)

        if len(docs) > 0:
            assert type(next(iter(docs))) is type(next(iter(docs_)))

        if docs_type == 2:
            assert doc_labels(docs_) == labels

        res_tokens = doc_tokens(docs_, to_lists=True)
        for d_, d in zip(res_tokens, docs):
            if docs_type == 2:
                # spaCy does not allow "empty" tokens, hence this may happen if all characters of a token are
                # removed
                assert len(d_) <= len(d)
            else:
                assert len(d_) == len(d)

            if len(d_) == len(d):
                for t_, t in zip(d_, d):
                    assert len(t_) <= len(t)
                    assert all(c not in t_ for c in chars)


@cleanup_after_test
@pytest.mark.parametrize(
    'chars, expected_docs',
    [
        (('.', ',', '-'), [
            ['I', 'live', 'in', 'New', 'York', ''],
            ['I', 'am', 'in', 'Berlin', '', 'but', 'my', 'flat', 'is', 'in', 'Munich', ''],
            ['US', '', 'Student', 'is', 'reading', 'an', 'e', '', 'mail', 'on', 'eCommerce', 'with', 'CamelCase', ''],
            []
        ]),
        (('e',), [
            ['I', 'liv', 'in', 'Nw', 'York', '.'],
            ['I', 'am', 'in', 'Brlin', ',', 'but', 'my', 'flat', 'is', 'in', 'Munich', '.'],
            ['US', '-', 'Studnt', 'is', 'rading', 'an', '', '-', 'mail', 'on', 'Commrc', 'with', 'CamlCas', '.'],
            []
        ]),
    ]
)
def test_remove_chars_examples(tokens_mini, tokens_mini_arrays, tokens_mini_lists, chars, expected_docs):
    for testtokens in (tokens_mini, tokens_mini_arrays, tokens_mini_lists):
        if testtokens is tokens_mini:
            # because spaCy docs never contain empty tokens:
            expected_docs_copy = [[t for t in d if t] for d in expected_docs]
        else:
            expected_docs_copy = deepcopy(expected_docs)

        assert doc_tokens(remove_chars(testtokens, chars), to_lists=True) == expected_docs_copy


@cleanup_after_test
@given(docs=strategy_lists_of_tokens(string.printable, min_size=1), pass_vocab=st.booleans(), pass_doc_labels=st.booleans(),
       return_vocab=st.booleans())
def test_tokendocs2spacydocs(docs, pass_vocab, pass_doc_labels, return_vocab):
    input_vocab = vocabulary(docs, sort=True)

    if pass_vocab:
        vocab = input_vocab
    else:
        vocab = None

    if pass_doc_labels:
        dlabels = ['doc%d' % i for i in range(len(docs))]
    else:
        dlabels = None

    res = tokendocs2spacydocs(docs, vocab=vocab, doc_labels=dlabels, return_vocab=return_vocab)

    if return_vocab:
        assert isinstance(res, tuple)
        spacydocs, returned_vocab = res
        assert isinstance(returned_vocab, Vocab)
        assert set(t.text for t in returned_vocab) == set(input_vocab)
    else:
        spacydocs = res

    assert len(spacydocs) == len(docs)
    assert all(isinstance(d, Doc) for d in spacydocs)
    assert doc_tokens(spacydocs, to_lists=True) == docs

    if pass_doc_labels:
        assert all(dl == 'doc%d' % i for i, dl in enumerate(doc_labels(spacydocs)))
    else:
        assert all(dl == '' for dl in doc_labels(spacydocs))


@cleanup_after_test
@given(docs=strategy_lists_of_tokens(string.printable))
@settings(deadline=1000)
def test_token2ids_and_inverse(docs):
    docs, vocab = tokendocs2spacydocs(docs, return_vocab=True)
    tokids = tokens2ids(docs)
    assert isinstance(tokids, list)
    assert len(tokids) == len(docs)
    assert all([isinstance(ids, np.ndarray) for ids in tokids])

    docs_ = ids2tokens(vocab, tokids)
    assert all([d.text == d_.text for d, d_ in zip(docs, docs_)])


@cleanup_after_test
def test_pos_tag_en(tokens_mini, tokens_mini_arrays, tokens_mini_lists):
    with pytest.raises(ValueError):   # only spaCy docs
        pos_tag(tokens_mini_arrays)
    with pytest.raises(ValueError):   # only spaCy docs
        pos_tag(tokens_mini_lists)

    # check empty
    assert pos_tag([]) == []

    # check non-empty
    tagged_docs = pos_tag(tokens_mini)
    assert isinstance(tagged_docs, list)
    assert len(tagged_docs) == len(tokens_mini)
    assert all([d_ is d for d_, d in zip(tagged_docs, tokens_mini)])

    tags = pos_tags(tagged_docs)
    assert isinstance(tags, list)
    assert len(tags) == len(tokens_mini)
    assert tags == [['PRON', 'VERB', 'ADP', 'PROPN', 'PROPN', 'PUNCT'],
         ['PRON',
          'AUX',
          'ADP',
          'PROPN',
          'PUNCT',
          'CCONJ',
          'DET',
          'NOUN',
          'AUX',
          'ADP',
          'PROPN',
          'PUNCT'],
         ['PROPN',
          'PUNCT',
          'PROPN',
          'AUX',
          'VERB',
          'DET',
          'NOUN',
          'NOUN',
          'NOUN',
          'ADP',
          'PROPN',
          'ADP',
          'PROPN',
          'PUNCT'],
         []
    ]


@cleanup_after_test
def test_lemmatize(tokens_mini, tokens_mini_arrays, tokens_mini_lists):
    with pytest.raises(ValueError):   # only spaCy docs
        lemmatize(tokens_mini_arrays)
    with pytest.raises(ValueError):   # only spaCy docs
        lemmatize(tokens_mini_lists)

    # check empty
    assert lemmatize([]) == []

    # check non-empty
    assert lemmatize(tokens_mini) == [
        ['I', 'live', 'in', 'New', 'York', '.'],
        ['I',
         'be',
         'in',
         'Berlin',
         ',',
         'but',
         'my',
         'flat',
         'be',
         'in',
         'Munich',
         '.'],
        ['US',
         '-',
         'Student',
         'be',
         'read',
         'a',
         'e',
         '-',
         'mail',
         'on',
         'eCommerce',
         'with',
         'CamelCase',
         '.'],
        []
    ]


@cleanup_after_test
@given(
    pass_vocab=st.integers(min_value=0, max_value=2),
    pass_spaces=st.integers(min_value=0, max_value=3),
    pass_lemmata=st.integers(min_value=0, max_value=3),
    pass_label=st.booleans()
)
def test_spacydoc_from_tokens(module_tokens_en_arrays, module_tokens_en_lists,
                              pass_vocab, pass_spaces, pass_lemmata, pass_label):
    for testtokens in (module_tokens_en_arrays, module_tokens_en_lists):
        for tokdoc in testtokens:
            should_raise = False
            tokdoc_list = tokdoc.tolist() if testtokens is module_tokens_en_arrays else tokdoc

            if pass_vocab:
                vocab = vocabulary([tokdoc_list])
                if pass_vocab == 2:
                    vocab = np.array(vocab) if vocab else empty_chararray()
            else:
                vocab = None

            if pass_spaces:
                if pass_spaces == 1:
                    spaces = [' '] * len(tokdoc)
                elif pass_spaces == 2:
                    spaces = np.repeat(' ', len(tokdoc)) if len(tokdoc) > 0 else empty_chararray()
                elif pass_spaces == 3:
                    spaces = [' '] * (len(tokdoc) + 1)   # wrong number of spaces
                    should_raise = True
                else:
                    pytest.fail('invalid value for pass_spaces: %d' % pass_spaces)
                    return
            else:
                spaces = None

            if pass_lemmata:
                if pass_lemmata == 1:
                    lemmata = tokdoc_list
                elif pass_lemmata == 2:
                    lemmata = np.array(tokdoc) if len(tokdoc) > 0 else empty_chararray()
                elif pass_lemmata == 3:
                    lemmata = tokdoc_list + ['foo']   # wrong number of lemmata
                    should_raise = True
                else:
                    pytest.fail('invalid value for pass_lemmata: %d' % pass_lemmata)
                    return
            else:
                lemmata = None

            if pass_label:
                label = 'testdoc'
            else:
                label = None

            if should_raise:
                with pytest.raises(ValueError):
                    spacydoc_from_tokens(tokdoc, vocab=vocab, spaces=spaces, lemmata=lemmata, label=label)
                return
            else:
                doc = spacydoc_from_tokens(tokdoc, vocab=vocab, spaces=spaces, lemmata=lemmata, label=label)
                assert isinstance(doc, Doc)
                assert len(doc) == len(tokdoc)
                assert all([t.text == t_ for t, t_ in zip(doc, tokdoc)])
                assert np.array_equal(doc.user_data['tokens'], tokdoc if len(tokdoc) > 0 else empty_chararray())
                assert np.array_equal(doc.user_data['mask'], np.repeat(True, len(tokdoc)))

                if pass_lemmata:
                    assert all([t.lemma_ == t_ for t, t_ in zip(doc, tokdoc)])


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

    Token.set_extension('testmeta', default='foo', force=True)

    if code is None:  # reset
        _docfuncs.nlp = None
    elif _docfuncs.nlp is None or _docfuncs.nlp.lang != code:  # init if necessary
        if code in _nlp_instances_cache.keys():
            _docfuncs.nlp = _nlp_instances_cache[code]
        else:
            init_for_language(code)
