"""
Preprocessing: Tests for ._docfuncs submodule.
"""

import pytest
import numpy as np
from spacy.tokens import Doc

from tmtoolkit.preprocess._common import DEFAULT_LANGUAGE_MODELS
from tmtoolkit.preprocess._docfuncs import init_for_language, tokenize

from ._testcorpora import corpora_sm


LANGUAGE_CODES = list(sorted(DEFAULT_LANGUAGE_MODELS.keys()))


@pytest.fixture(scope='module', params=LANGUAGE_CODES)
def nlp_all(request):
    return init_for_language(request.param)


@pytest.fixture()
def tokens_en():
    _init_lang('en')
    return tokenize(corpora_sm['en'])


@pytest.fixture()
def tokens_mini():
    _init_lang('en')
    corpus = {'ny': 'I live in New York.',
              'bln': 'I am in Berlin, but my flat is in Munich.',
              'empty': ''}
    return tokenize(corpus)


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
            assert np.issubdtype(doc.user_data['tokens'].dtype, np.str_)

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
