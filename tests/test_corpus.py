"""
Tests for tmtoolkit.corpus module.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

from importlib.util import find_spec
from copy import copy

import pytest

if not find_spec('spacy'):
    pytest.skip("skipping tmtoolkit.corpus tests (spacy not installed)", allow_module_level=True)

from spacy.tokens import Doc

from tmtoolkit import corpus as c
from tmtoolkit._pd_dt_compat import USE_DT, FRAME_TYPE
from ._testtextdata import textdata_sm

textdata_en = textdata_sm['en']
textdata_de = textdata_sm['de']


#%% tests setup


@pytest.fixture
def corpus_en():
    return c.Corpus(textdata_en, language='en')


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
    corp = c.Corpus({}, language='en')

    assert corp.n_docs == 0
    assert corp.doc_labels == []

    c.to_lowercase(corp)
    c.filter_clean_tokens(corp)
    c.filter_documents(corp, 'foobar')
    c.set_token_attr(corp, 'fooattr', {'footoken': 'somevalue'})

    assert corp.n_docs == 0
    assert corp.doc_labels == []

    _check_copies(corp, copy(corp))


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


def _check_copies(corp_a, corp_b):
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

