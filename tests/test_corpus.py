"""
Tests for tmtoolkit.corpus module.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import string
from importlib.util import find_spec
from copy import copy, deepcopy

import pytest
from hypothesis import given, strategies as st

from tmtoolkit.utils import flatten_list
from ._testtools import strategy_str_str_dict_printable

if not find_spec('spacy'):
    pytest.skip("skipping tmtoolkit.corpus tests (spacy not installed)", allow_module_level=True)

import spacy
from spacy.tokens import Doc

from tmtoolkit import corpus as c
from tmtoolkit._pd_dt_compat import USE_DT, FRAME_TYPE
from ._testtextdata import textdata_sm

textdata_en = textdata_sm['en']
textdata_de = textdata_sm['de']


#%% tests setup

# use same instance several times (because some tests are super slow otherwise)
spacy_instance_en_sm = spacy.load('en_core_web_sm')


@pytest.fixture
def corpus_en():
    return c.Corpus(textdata_en, language='en')

@pytest.fixture
def corpora_en_serial_and_parallel():
    return (c.Corpus(textdata_en, language='en', max_workers=1),
            c.Corpus(textdata_en, language='en', max_workers=2))


@pytest.fixture
def corpus_de():
    return c.Corpus(textdata_de, language='de')



#%% test fixtures


def test_fixtures_n_docs_and_doc_labels(corpus_en, corpus_de):
    assert corpus_en.n_docs == len(textdata_en)
    assert corpus_de.n_docs == len(textdata_de)

    assert set(corpus_en.doc_labels) == set(textdata_en.keys())
    assert set(corpus_de.doc_labels) == set(textdata_de.keys())


#%% test init


def test_corpus_no_lang_given():
    with pytest.raises(ValueError) as exc:
        c.Corpus({})

    assert str(exc.value).endswith('either `language` or `language_model` must be given')


def test_empty_corpus():
    for w in (1, 2):
        corp = c.Corpus({}, language='en', max_workers=w)

        assert corp.n_docs == 0
        assert corp.doc_labels == []

        c.to_lowercase(corp)
        c.filter_clean_tokens(corp)
        c.filter_documents(corp, 'foobar')
        c.set_token_attr(corp, 'fooattr', {'footoken': 'somevalue'})

        assert corp.n_docs == 0
        assert corp.doc_labels == []

        _check_copies(corp, copy(corp))


def test_corpus_init():
    with pytest.raises(ValueError) as exc:
        c.Corpus(textdata_en, language='00')
    assert 'is not supported' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        c.Corpus(textdata_en, language='fail')
    assert str(exc.value) == '`language` must be a two-letter ISO 639-1 language code'

    with pytest.raises(ValueError) as exc:
        c.Corpus(textdata_en)
    assert str(exc.value) == 'either `language` or `language_model` must be given'

    corp = c.Corpus(textdata_en, language='en')
    assert 'ner' not in corp.nlp.component_names
    assert 'parser' not in corp.nlp.component_names

    corp = c.Corpus(textdata_en, language='en', spacy_exclude=[])
    assert 'ner' in corp.nlp.component_names
    assert 'parser' in corp.nlp.component_names

    corp = c.Corpus(textdata_en, language='en', spacy_opts={'vocab': True})
    assert corp._spacy_opts['vocab'] is True

    _check_copies(corp, copy(corp))
    _check_copies(corp, deepcopy(corp), is_deepcopy=True)


@given(docs=strategy_str_str_dict_printable(),
       punctuation=st.one_of(st.none(), st.lists(st.text(string.punctuation, min_size=1, max_size=1))),
       max_workers=st.one_of(st.none(),
                             st.integers(min_value=-4, max_value=4),
                             st.floats(allow_nan=False, allow_infinity=False)),
       workers_timeout=st.integers())
def test_corpus_init_and_properties_hypothesis(docs, punctuation, max_workers, workers_timeout):
    args = dict(docs=docs, spacy_instance=spacy_instance_en_sm, punctuation=punctuation,
                max_workers=max_workers, workers_timeout=workers_timeout)

    if isinstance(max_workers, float) and not 0 <= max_workers <= 1:
        with pytest.raises(ValueError) as exc:
            c.Corpus(**args)

        assert str(exc.value) == '`max_workers` must be an integer, a float in [0, 1] or None'
    else:
        corp = c.Corpus(**args)
        assert corp.nlp == spacy_instance_en_sm
        if punctuation is None:
            assert corp.punctuation == list(string.punctuation) + [' ', '\r', '\n', '\t']
        else:
            assert corp.punctuation == punctuation

        assert 0 < corp.max_workers <= 4
        if corp.max_workers == 1:
            assert corp.procexec is None
            assert corp.workers_docs == []
        else:
            assert len(corp.workers_docs) == min(corp.max_workers, len(docs))
            workers_docs_flat = flatten_list(corp.workers_docs)
            workers_docs_flat_set = set(flatten_list(corp.workers_docs))
            assert len(workers_docs_flat) == len(workers_docs_flat_set)
            assert workers_docs_flat_set == set(docs.keys())

        assert corp.workers_timeout == workers_timeout
        assert str(corp) == repr(corp)
        assert str(corp).startswith(f'<Corpus [{len(docs)}')
        assert len(corp) == len(docs)
        assert bool(corp) == bool(docs)

        corp['_new_doc'] = 'Foo bar.'
        assert '_new_doc' in corp
        assert len(corp) == len(docs) + 1
        assert corp['_new_doc'] == ['Foo', 'bar', '.']
        del corp['_new_doc']
        assert '_new_doc' not in corp
        assert len(corp) == len(docs)

        for (lbl, tok), lbl2, tok2 in zip(corp.items(), corp.keys(), corp.values()):
            assert lbl in docs.keys()
            assert isinstance(tok, list)
            assert lbl == lbl2
            assert tok == tok2

        assert not corp.docs_filtered
        assert not corp.tokens_filtered
        assert not corp.is_filtered
        assert not corp.tokens_processed
        assert not corp.is_processed
        assert corp.uses_unigrams
        assert corp.token_attrs == corp.STD_TOKEN_ATTRS
        assert corp.custom_token_attrs_defaults == {}
        assert corp.doc_attrs == []
        assert corp.doc_attrs_defaults == {}
        assert corp.ngrams == 1
        assert corp.ngrams_join_str == ' '
        assert corp.language == 'en'
        assert corp.language_model == 'en_core_web_sm'
        assert corp.doc_labels == sorted(docs.keys())
        assert isinstance(corp.docs, dict)
        assert set(corp.docs.keys()) == set(docs.keys())
        assert set(corp.spacydocs.keys()) == set(docs.keys())
        assert corp.n_docs == len(docs)
        assert corp.n_docs_masked == 0
        assert not corp.ignore_doc_filter
        assert corp.spacydocs is corp.spacydocs_ignore_filter

        if corp:
            lbl = next(iter(docs.keys()))
            assert isinstance(corp[lbl], list)
            assert isinstance(corp.get(lbl), list)
            assert corp.get(1312, None) is None
            assert next(iter(corp)) == next(iter(corp.spacydocs.keys()))
            assert isinstance(next(iter(corp.docs.values())), list)
            assert isinstance(next(iter(corp.spacydocs.values())), Doc)


@pytest.mark.skip       # TODO: re-enable
def test_corpus_init_otherlang_by_langcode():
    for langcode, docs in textdata_sm.items():
        if langcode in {'en', 'de'}: continue  # this is already tested

        corp = c.Corpus(docs, language=langcode)

        assert set(corp.doc_labels) == set(docs.keys())
        assert corp.language == langcode
        assert corp.language_model.startswith(langcode)
        assert corp.max_workers == 1

        for d in corp.spacydocs.values():
            assert isinstance(d, Doc)



#%% helper functions


def _check_copies(corp_a, corp_b, is_deepcopy=False):
    attrs_a = dir(corp_a)
    attrs_b = dir(corp_b)

    # check if simple attributes are the same
    simple_state_attrs = ('docs_filtered', 'tokens_filtered', 'is_filtered', 'tokens_processed',
                          'is_processed', 'uses_unigrams', 'token_attrs', 'custom_token_attrs_defaults', 'doc_attrs',
                          'doc_attrs_defaults', 'ngrams', 'ngrams_join_str', 'language', 'language_model',
                          'doc_labels', 'n_docs', 'n_docs_masked', 'ignore_doc_filter', 'workers_docs',
                          'max_workers')

    for attr in simple_state_attrs:
        assert attr in attrs_a
        assert attr in attrs_b
        assert getattr(corp_a, attr) == getattr(corp_b, attr)

    # check if tokens are the same
    tok_a = c.doc_tokens(corp_a)
    tok_b = c.doc_tokens(corp_b)
    assert tok_a == tok_b

    if is_deepcopy:
        assert corp_a.nlp is not corp_b.nlp
        assert corp_a.nlp.meta == corp_b.nlp.meta
    else:
        assert corp_a.nlp is corp_b.nlp

    # check if token dataframes are the same
    assert _dataframes_equal(c.tokens_datatable(corp_a), c.tokens_datatable(corp_b))


def _dataframes_equal(df1, df2):
    # so far, datatable doesn't seem to support dataframe comparisons
    if USE_DT:
        if isinstance(df1, FRAME_TYPE):
            df1 = df1.to_pandas()
        if isinstance(df2, FRAME_TYPE):
            df2 = df2.to_pandas()
    return df1.shape == df2.shape and (df1 == df2).all(axis=1).sum() == len(df1)

