"""
Tests for tmtoolkit.corpus module.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""
import functools
import string
from collections import Counter
from importlib.util import find_spec
from copy import copy, deepcopy

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from tmtoolkit.utils import flatten_list
from ._testtools import strategy_str_str_dict_printable

if not find_spec('spacy'):
    pytest.skip("skipping tmtoolkit.corpus tests (spacy not installed)", allow_module_level=True)

import spacy
from spacy.tokens import Doc

from tmtoolkit import corpus as c
from tmtoolkit._pd_dt_compat import USE_DT, FRAME_TYPE, f, pd_dt_colnames
from ._testtextdata import textdata_sm

textdata_en = textdata_sm['en']
textdata_de = textdata_sm['de']


#%% tests setup

# note: scope='module' means that fixture is created once per test module (not per test function)

@pytest.fixture(scope='module')
def spacy_instance_en_sm():
    return spacy.load('en_core_web_sm')


@pytest.fixture
def corpus_en():
    return c.Corpus(textdata_en, language='en')


@pytest.fixture(scope='module')
def corpus_en_module():
    return c.Corpus(textdata_en, language='en')


@pytest.fixture
def corpora_en_serial_and_parallel():
    return (c.Corpus(textdata_en, language='en', max_workers=1),
            c.Corpus(textdata_en, language='en', max_workers=2))


@pytest.fixture(scope='module')
def corpora_en_serial_and_parallel_module():
    return (c.Corpus(textdata_en, language='en', max_workers=1),
            c.Corpus(textdata_en, language='en', max_workers=2))


@pytest.fixture(scope='module')
def corpora_en_serial_and_parallel_also_w_vectors_module():
    return (c.Corpus(textdata_en, language='en', max_workers=1),
            c.Corpus(textdata_en, language='en', max_workers=2),
            c.Corpus(textdata_en, language_model='en_core_web_md', max_workers=1),
            c.Corpus(textdata_en, language_model='en_core_web_md', max_workers=2))


@pytest.fixture
def corpus_de():
    return c.Corpus(textdata_de, language='de')


@pytest.fixture(scope='module')
def corpus_de_module():
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
def test_corpus_init_and_properties_hypothesis(spacy_instance_en_sm, docs, punctuation, max_workers, workers_timeout):
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


#%%


@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       only_non_empty=st.booleans(),
       tokens_as_hashes=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos',
                                                           c.Corpus.STD_TOKEN_ATTRS,
                                                           c.Corpus.STD_TOKEN_ATTRS + ['mask'],
                                                           c.Corpus.STD_TOKEN_ATTRS + ['doc_mask'],
                                                           ['doc_mask', 'mask'],
                                                           ['mask'],
                                                           ['doc_mask'],
                                                           c.Corpus.STD_TOKEN_ATTRS + ['nonexistent']])),
       with_mask=st.booleans(),
       with_spacy_tokens=st.booleans(),
       as_datatables=st.booleans(),
       as_arrays=st.booleans())
def test_doc_tokens_hypothesis(corpora_en_serial_and_parallel_module, **args):
    for corp in list(corpora_en_serial_and_parallel_module) + [None]:
        if corp is None:
            corp = corpora_en_serial_and_parallel_module[0].spacydocs   # test dict of SpaCy docs

        if args['select'] == 'nonexistent':
            with pytest.raises(KeyError):
                c.doc_tokens(corp, **args)
        elif isinstance(args['with_attr'], list) and 'nonexistent' in args['with_attr'] \
                and args['select'] not in ('empty', []):
            with pytest.raises(AttributeError):
                c.doc_tokens(corp, **args)
        else:
            res = c.doc_tokens(corp, **args)
            assert isinstance(res, dict)

            if args['select'] is None:
                if args['only_non_empty']:
                    assert len(res) == len(corp) - 1
                else:
                    assert len(res) == len(corp)
            else:
                assert set(res.keys()) == set(args['select']) if isinstance(args['select'], list) \
                    else args['select']

            if res:
                if args['only_non_empty']:
                    assert 'empty' not in res.keys()

                if args['as_datatables']:
                    assert all([isinstance(v, FRAME_TYPE) for v in res.values()])
                    cols = [tuple(pd_dt_colnames(v)) for v in res.values()]
                    assert len(set(cols)) == 1
                    attrs = next(iter(set(cols)))
                else:
                    if args['with_attr'] or args['with_mask'] or args['with_spacy_tokens']:
                        assert all([isinstance(v, dict) for v in res.values()])
                        if args['as_arrays']:
                            assert all([isinstance(arr, np.ndarray) for v in res.values()
                                        for k, arr in v.items() if k != 'doc_mask'])

                        cols = [tuple(v.keys()) for v in res.values()]
                        assert len(set(cols)) == 1
                        attrs = next(iter(set(cols)))
                    else:
                        assert all([isinstance(v, np.ndarray if args['as_arrays'] else list) for v in res.values()])
                        attrs = None

                firstattrs = ['token']
                lastattrs = []

                if args['with_spacy_tokens']:
                    firstattrs.append('text')

                if args['with_mask']:
                    firstattrs = ['doc_mask'] + firstattrs
                    lastattrs = ['mask']

                if args['with_attr'] is True:
                    assert attrs == tuple(firstattrs + c.Corpus.STD_TOKEN_ATTRS + lastattrs)
                elif args['with_attr'] is False:
                    if args['as_datatables']:
                        assert attrs == tuple(firstattrs + lastattrs)
                    else:
                        if args['with_mask'] or args['with_spacy_tokens']:
                            assert attrs == tuple(firstattrs + lastattrs)
                        else:
                            assert attrs is None
                else:
                    if isinstance(args['with_attr'], str):
                        assert attrs == tuple(firstattrs + [args['with_attr']] + lastattrs)
                    else:
                        if 'mask' in args['with_attr'] and 'mask' in lastattrs:
                            args['with_attr'] = [a for a in args['with_attr'] if a != 'mask']
                        if 'doc_mask' in args['with_attr']:
                            if 'doc_mask' not in firstattrs:
                                firstattrs = ['doc_mask'] + firstattrs
                            args['with_attr'] = [a for a in args['with_attr'] if a != 'doc_mask']
                        assert attrs == tuple(firstattrs + args['with_attr'] + lastattrs)


def test_doc_lengths(corpora_en_serial_and_parallel_module):
    expected = {
        'empty': 0,
        'small1': 1,
        'small2': 7
    }
    for corp in corpora_en_serial_and_parallel_module:
        res = c.doc_lengths(corp)
        assert isinstance(res, dict)
        assert set(res.keys()) == set(corp.keys())
        for lbl, n in res.items():
            assert n >= 0
            if lbl in expected:
                assert n == expected[lbl]
            else:
                assert n >= len(textdata_en[lbl].split())


def test_doc_token_lengths(corpora_en_serial_and_parallel_module):
    expected = {
        'empty': [],
        'small1': [3],
        'small2': [4, 2, 1, 5, 7, 8, 1]
    }

    for corp in corpora_en_serial_and_parallel_module:
        res = c.doc_token_lengths(corp)
        assert isinstance(res, dict)
        assert set(res.keys()) == set(corp.keys())

        for lbl, toklengths in res.items():
            assert isinstance(toklengths, list)
            assert all([n >= 0 for n in toklengths])
            if lbl in expected:
                assert toklengths == expected[lbl]


@pytest.mark.parametrize('sort', [False, True])
def test_doc_labels(corpora_en_serial_and_parallel_module, sort):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.doc_labels(corp, sort=sort)
        assert isinstance(res, list)
        if sort:
            assert res == sorted(textdata_en.keys())
        else:
            assert res == list(textdata_en.keys())


@pytest.mark.parametrize('collapse', [None, ' ', '__'])
def test_doc_texts(corpora_en_serial_and_parallel_module, collapse):
    expected = {
        ' ': {
            'empty': '',
            'small1': 'the',
            'small2': 'This is a small example document .'
        },
        '__': {
            'empty': '',
            'small1': 'the',
            'small2': 'This__is__a__small__example__document__.'
        }
    }

    for corp in corpora_en_serial_and_parallel_module:
        res = c.doc_texts(corp, collapse=collapse)
        assert isinstance(res, dict)
        assert set(res.keys()) == set(corp.keys())

        for lbl, txt in res.items():
            assert isinstance(txt, str)
            if collapse is None:
                assert txt == textdata_en[lbl]
            else:
                if lbl in expected[collapse]:
                    assert txt == expected[collapse][lbl]


@pytest.mark.parametrize('proportions', [False, True])
def test_doc_frequencies(corpora_en_serial_and_parallel_module, proportions):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.doc_frequencies(corp, proportions=proportions)
        assert isinstance(res, dict)
        assert set(res.keys()) == c.vocabulary(corp)

        if proportions:
            assert all([0 < v <= 1 for v in res.values()])
            assert np.isclose(res['the'], 5/7)
        else:
            assert all([0 < v < len(corp) for v in res.values()])
            assert any([v > 1 for v in res.values()])
            assert res['the'] == 5


@pytest.mark.parametrize('omit_empty', [False, True])
def test_doc_vectors(corpora_en_serial_and_parallel_also_w_vectors_module, omit_empty):
    for i_corp, corp in enumerate(corpora_en_serial_and_parallel_also_w_vectors_module):
        if i_corp < 2:
            with pytest.raises(RuntimeError):
                c.doc_vectors(corp, omit_empty=omit_empty)
        else:
            res = c.doc_vectors(corp, omit_empty=omit_empty)
            assert isinstance(res, dict)

            if omit_empty:
                assert set(res.keys()) == set(corp.keys()) - {'empty'}
            else:
                assert set(res.keys()) == set(corp.keys())

            for vec in res.values():
                assert isinstance(vec, np.ndarray)
                assert len(vec) > 0


@pytest.mark.parametrize('omit_oov', [False, True])
def test_token_vectors(corpora_en_serial_and_parallel_also_w_vectors_module, omit_oov):
    for i_corp, corp in enumerate(corpora_en_serial_and_parallel_also_w_vectors_module):
        if i_corp < 2:
            with pytest.raises(RuntimeError):
                c.token_vectors(corp, omit_oov=omit_oov)
        else:
            res = c.token_vectors(corp, omit_oov=omit_oov)
            assert isinstance(res, dict)
            assert set(res.keys()) == set(corp.keys())

            doc_length = c.doc_lengths(corp)

            for lbl, mat in res.items():
                assert isinstance(mat, np.ndarray)

                if omit_oov:
                    assert len(mat) == sum([not t.is_oov for t in corp.spacydocs[lbl]])
                else:
                    assert len(mat) == doc_length[lbl]

                if len(mat) > 0:
                    assert mat.ndim == 2


@given(tokens_as_hashes=st.booleans(),
       force_unigrams=st.booleans(),
       sort=st.booleans())
def test_vocabulary_hypothesis(corpora_en_serial_and_parallel_module, tokens_as_hashes, force_unigrams, sort):
    for corp in corpora_en_serial_and_parallel_module:
        # TODO: check force_unigrams when ngrams enabled/disabled
        res = c.vocabulary(corp, tokens_as_hashes=tokens_as_hashes, force_unigrams=force_unigrams, sort=sort)

        if sort:
            assert isinstance(res, list)
            assert sorted(res) == res
        else:
            assert isinstance(res, set)

        assert len(res) > 0

        if tokens_as_hashes:
            expect_type = int
        else:
            expect_type = str

        assert all([isinstance(t, expect_type) for t in res])


@settings(deadline=None)
@given(tokens_as_hashes=st.booleans(),
       force_unigrams=st.booleans())
def test_vocabulary_counts(corpora_en_serial_and_parallel_module, tokens_as_hashes, force_unigrams):
    for corp in corpora_en_serial_and_parallel_module:
        # TODO: check force_unigrams when ngrams enabled/disabled
        res = c.vocabulary_counts(corp, tokens_as_hashes=tokens_as_hashes, force_unigrams=force_unigrams)

        assert isinstance(res, Counter)
        assert len(res) > 0

        if tokens_as_hashes:
            expect_type = int
        else:
            expect_type = str

        assert all([isinstance(t, expect_type) for t in res.keys()])
        assert all([n > 0 for n in res.values()])

        vocab = c.vocabulary(corp, tokens_as_hashes=tokens_as_hashes, force_unigrams=force_unigrams)
        assert vocab == set(res.keys())


def test_vocabulary_size(corpora_en_serial_and_parallel_module):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.vocabulary_size(corp)

        assert isinstance(res, int)
        assert res > 0


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       tokens_as_hashes=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos',
                                                           c.Corpus.STD_TOKEN_ATTRS,
                                                           c.Corpus.STD_TOKEN_ATTRS + ['mask'],
                                                           c.Corpus.STD_TOKEN_ATTRS + ['doc_mask'],
                                                           ['doc_mask', 'mask'],
                                                           ['mask'],
                                                           ['doc_mask'],
                                                           c.Corpus.STD_TOKEN_ATTRS + ['nonexistent']])),
       with_mask=st.booleans(),
       with_spacy_tokens=st.booleans())
def test_tokens_datatable_hypothesis(corpora_en_serial_and_parallel_module, **args):
    for corp in corpora_en_serial_and_parallel_module:
        if args['select'] == 'nonexistent':
            with pytest.raises(KeyError):
                c.tokens_datatable(corp, **args)
        elif isinstance(args['with_attr'], list) and 'nonexistent' in args['with_attr'] \
                and args['select'] not in ('empty', []):
            with pytest.raises(AttributeError):
                c.tokens_datatable(corp, **args)
        else:
            res = c.tokens_datatable(corp, **args)
            assert isinstance(res, FRAME_TYPE)

            cols = pd_dt_colnames(res)
            assert cols[:2] == ['doc', 'position']
            assert 'token' in cols
            docs_set = set(res['doc'].to_list()[0])
            if args['select'] is None:
                assert docs_set == set(corp.keys()) - {'empty'}
            else:
                select_set = {args['select']} if isinstance(args['select'], str) else set(args['select'])
                assert docs_set == select_set - {'empty'}

            if USE_DT:  # TODO: also check pandas DataFrames
                dlengths = c.doc_lengths(corp)
                for lbl in docs_set:
                        tokpos = res[f.doc == lbl, 'position'].to_list()[0]
                        assert tokpos == list(range(dlengths[lbl]))

            if res.shape[0] > 0:   # can only guarantee the columns when we actually have observations
                if args['with_attr'] is True:
                    assert set(cols) & set(c.Corpus.STD_TOKEN_ATTRS) == set(c.Corpus.STD_TOKEN_ATTRS)
                elif isinstance(args['with_attr'], str):
                    assert args['with_attr'] in cols
                elif isinstance(args['with_attr'], list):
                    assert set(cols) & set(args['with_attr']) == set(args['with_attr'])

                if args['with_mask']:
                    assert set(cols) & {'doc_mask', 'mask'} == {'doc_mask', 'mask'}


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

