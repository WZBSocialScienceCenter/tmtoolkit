"""
Tests for tmtoolkit.corpus module.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""
import random
import string
import tempfile
from collections import Counter
from importlib.util import find_spec
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings

if not find_spec('spacy'):
    pytest.skip("skipping tmtoolkit.corpus tests (spacy not installed)", allow_module_level=True)

import spacy
from spacy.tokens import Doc
from scipy.sparse import csr_matrix

from tmtoolkit import tokenseq
from tmtoolkit.utils import flatten_list
from tmtoolkit.corpus._common import LANGUAGE_LABELS
from tmtoolkit import corpus as c
from ._testtools import strategy_str_str_dict_printable
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
            c.Corpus(textdata_en, language='en', max_workers=2),
            c.Corpus({}, language='en'))


@pytest.fixture(scope='module')
def corpora_en_serial_and_parallel_module():
    return (c.Corpus(textdata_en, language='en', max_workers=1),
            c.Corpus(textdata_en, language='en', max_workers=2),
            c.Corpus({}, language='en'))


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

        _check_copies(corp, copy(corp), same_nlp_instance=True)


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

    _check_copies(corp, copy(corp), same_nlp_instance=True)
    _check_copies(corp, deepcopy(corp), same_nlp_instance=False)


@settings(deadline=None)
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
       as_tables=st.booleans(),
       as_arrays=st.booleans())
def test_doc_tokens_hypothesis(corpora_en_serial_and_parallel_module, **args):
    for corp in list(corpora_en_serial_and_parallel_module) + [None]:
        if corp is None:
            corp = corpora_en_serial_and_parallel_module[0].spacydocs   # test dict of SpaCy docs

        if args['select'] == 'nonexistent' or (args['select'] is not None and args['select'] != [] and len(corp) == 0):
            with pytest.raises(KeyError):
                c.doc_tokens(corp, **args)
        elif len(corp) > 0 and isinstance(args['with_attr'], list) and 'nonexistent' in args['with_attr'] \
                and args['select'] not in ('empty', []):
            with pytest.raises(AttributeError):
                c.doc_tokens(corp, **args)
        else:
            res = c.doc_tokens(corp, **args)
            assert isinstance(res, dict)

            if args['select'] is None:
                if args['only_non_empty'] and len(corp) > 0:
                    assert len(res) == len(corp) - 1
                else:
                    assert len(res) == len(corp)
            else:
                assert set(res.keys()) == set(args['select']) if isinstance(args['select'], list) \
                    else args['select']

            if res:
                if args['only_non_empty']:
                    assert 'empty' not in res.keys()

                if args['as_tables']:
                    assert all([isinstance(v, pd.DataFrame) for v in res.values()])
                    cols = [tuple(v.columns) for v in res.values()]
                    assert len(set(cols)) == 1
                    attrs = next(iter(set(cols)))

                    for v in res.values():
                        if len(v) > 0:
                            assert np.issubdtype(v['token'].dtype,
                                                 np.uint64 if args['tokens_as_hashes'] else np.dtype('O'))
                else:
                    if args['with_attr'] or args['with_mask'] or args['with_spacy_tokens']:
                        assert all([isinstance(v, dict) for v in res.values()])
                        if args['as_arrays']:
                            assert all([isinstance(arr, np.ndarray) for v in res.values()
                                        for k, arr in v.items() if k != 'doc_mask'])

                        cols = [tuple(v.keys()) for v in res.values()]
                        assert len(set(cols)) == 1
                        attrs = next(iter(set(cols)))
                        res_tokens = [v['token'] for v in res.values()]
                    else:
                        assert all([isinstance(v, np.ndarray if args['as_arrays'] else list) for v in res.values()])
                        attrs = None
                        res_tokens = res.values()

                    if args['tokens_as_hashes']:
                        if args['as_arrays']:
                            assert all([np.issubdtype(v.dtype, np.uint64) for v in res_tokens])
                        else:
                            assert all([isinstance(t, int) for v in res_tokens for t in v])
                    else:
                        if args['as_arrays']:
                            assert all([np.issubdtype(v.dtype, str) for v in res_tokens])
                        else:
                            assert all([isinstance(t, str) for v in res_tokens for t in v])

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
                    if args['as_tables']:
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
                        with_attr = args['with_attr']
                        if 'mask' in args['with_attr'] and 'mask' in lastattrs:
                            with_attr = [a for a in with_attr if a != 'mask']
                        if 'doc_mask' in with_attr:
                            if 'doc_mask' not in firstattrs:
                                firstattrs = ['doc_mask'] + firstattrs
                            with_attr = [a for a in with_attr if a != 'doc_mask']
                        assert attrs == tuple(firstattrs + with_attr + lastattrs)


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

        if len(corp):
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

        if len(corp) > 0:
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

        if len(corp) > 0:
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

        if len(corp) > 0:
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
        if len(corp) > 0:
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
def test_tokens_table_hypothesis(corpora_en_serial_and_parallel_module, **args):
    for corp in corpora_en_serial_and_parallel_module:
        if args['select'] == 'nonexistent' or (args['select'] is not None and args['select'] != [] and len(corp) == 0):
            with pytest.raises(KeyError):
                c.tokens_table(corp, **args)
        elif len(corp) > 0 and isinstance(args['with_attr'], list) and 'nonexistent' in args['with_attr'] \
                and args['select'] not in ('empty', []):
            with pytest.raises(AttributeError):
                c.tokens_table(corp, **args)
        else:
            res = c.tokens_table(corp, **args)
            assert isinstance(res, pd.DataFrame)

            cols = res.columns.tolist()
            assert cols[:2] == ['doc', 'position']
            assert 'token' in cols
            docs_set = set(res['doc'])
            if args['select'] is None:
                assert docs_set == set(corp.keys()) - {'empty'}
            else:
                select_set = {args['select']} if isinstance(args['select'], str) else set(args['select'])
                assert docs_set == select_set - {'empty'}

            dlengths = c.doc_lengths(corp)
            for lbl in docs_set:
                tokpos = res[res.doc == lbl].position.tolist()
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


@given(tokens_as_hashes=st.booleans(), as_array=st.booleans())
def test_corpus_tokens_flattened(corpora_en_serial_and_parallel_module, tokens_as_hashes, as_array):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.corpus_tokens_flattened(corp, tokens_as_hashes=tokens_as_hashes, as_array=as_array)

        if as_array:
            assert isinstance(res, np.ndarray)
            expected_tok_type = np.uint64 if tokens_as_hashes else str
            assert all([isinstance(t, expected_tok_type)for t in res])
        else:
            assert isinstance(res, list)
            expected_tok_type = int if tokens_as_hashes else str
            assert all([isinstance(t, expected_tok_type) for t in res])

        assert len(res) == sum(c.doc_lengths(corp).values())


def test_corpus_num_tokens(corpora_en_serial_and_parallel_module):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.corpus_num_tokens(corp)
        assert res == sum(c.doc_lengths(corp).values())
        if len(corp) == 0:
            assert res == 0


def test_corpus_num_chars(corpora_en_serial_and_parallel_module):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.corpus_num_chars(corp)
        if len(corp) == 0:
            assert res == 0
        else:
            assert res > 0


@settings(deadline=None)
@given(threshold=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
       min_count=st.integers(),
       embed_tokens_min_docfreq=st.one_of(st.none(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
       pass_embed_tokens=st.integers(min_value=0, max_value=3),
       statistic=st.sampled_from([tokenseq.pmi, tokenseq.npmi, tokenseq.pmi2, tokenseq.pmi3,
                                  tokenseq.simple_collocation_counts]),
       return_statistic=st.booleans(),
       rank=st.sampled_from([None, 'asc', 'desc']),
       as_table=st.booleans(),
       glue=st.one_of(st.none(), st.text(string.printable)))
def test_corpus_collocations_hypothesis(corpora_en_serial_and_parallel_module, **args):
    pass_embed_tokens = args.pop('pass_embed_tokens')

    for corp in corpora_en_serial_and_parallel_module:
        if pass_embed_tokens > 0:
            vocab = list(c.vocabulary(corp))
            args['embed_tokens_set'] = random.choices(vocab, k=min(pass_embed_tokens, len(vocab)))
        else:
            args['embed_tokens_set'] = None

        if args['as_table'] and args['glue'] is None:
            with pytest.raises(ValueError):
                c.corpus_collocations(corp, **args)
        elif (isinstance(args['embed_tokens_min_docfreq'], int) and args['embed_tokens_min_docfreq'] < 1) or \
             (isinstance(args['embed_tokens_min_docfreq'], float) and not 0 <= args['embed_tokens_min_docfreq'] <= 1):
            with pytest.raises(ValueError):
                c.corpus_collocations(corp, **args)
        elif args['min_count'] < 0:
            with pytest.raises(ValueError):
                c.corpus_collocations(corp, **args)
        else:
            res = c.corpus_collocations(corp, **args)

            if args['as_table']:
                assert isinstance(res, pd.DataFrame)
                if args['return_statistic']:
                    assert res.columns.tolist() == ['collocation', 'statistic']
                else:
                    assert res.columns.tolist() == ['collocation']

                if args['glue'] != '':
                    assert all([args['glue'] in colloc for colloc in res['collocation']])
            else:
                assert isinstance(res, list)
                # the rest is already checked in test_tokenseq::test_token_collocations* tests


@given(max_documents=st.one_of(st.none(), st.integers()),
       max_tokens_string_length=st.one_of(st.none(), st.integers()))
def test_corpus_summary(corpora_en_serial_and_parallel_module, max_documents, max_tokens_string_length):
    for corp in corpora_en_serial_and_parallel_module:
        if (max_documents is not None and max_documents < 0) or \
                (max_tokens_string_length is not None and max_tokens_string_length < 0):
            with pytest.raises(ValueError):
                c.corpus_summary(corp, max_documents=max_documents, max_tokens_string_length=max_tokens_string_length)
        else:
            res = c.corpus_summary(corp, max_documents=max_documents, max_tokens_string_length=max_tokens_string_length)
            assert isinstance(res, str)
            assert str(len(corp)) in res
            assert str(corp.n_docs_masked) in res
            assert LANGUAGE_LABELS[corp.language].capitalize() in res
            assert str(c.corpus_num_tokens(corp)) in res
            assert str(c.vocabulary_size(corp)) in res

            lines = res.split('\n')
            n_docs_printed = corp.print_summary_default_max_documents if max_documents is None else max_documents
            assert len(lines) == 2 + min(len(corp), n_docs_printed + bool(len(corp) > n_docs_printed))

            if corp.tokens_processed:
                assert 'processed' in lines[-1]
            if corp.tokens_filtered:
                assert 'filtered' in lines[-1]
            if corp.ngrams > 1:
                assert f'{corp.ngrams}-grams' in lines[-1]


def test_print_summary(capsys, corpora_en_serial_and_parallel_module):
    for corp in corpora_en_serial_and_parallel_module:
        c.print_summary(corp)
        assert capsys.readouterr().out == c.corpus_summary(corp) + '\n'


@pytest.mark.parametrize('as_table, dtype, return_doc_labels, return_vocab', [
    (False, None, False, False),
    (True, None, False, False),
    (False, 'uint16', False, False),
    (True, 'uint16', False, False),
    (False, 'float64', False, False),
    (True, 'float64', False, False),
    (False, None, True, False),
    (True, None, True, False),
    (False, None, False, True),
    (True, None, False, True),
    (False, None, True, True),
    (True, None, True, True),
])
def test_dtm(corpora_en_serial_and_parallel_module, as_table, dtype, return_doc_labels, return_vocab):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.dtm(corp, as_table=as_table, dtype=dtype,
                    return_doc_labels=return_doc_labels, return_vocab=return_vocab)

        expected_vocab = c.vocabulary(corp, sort=True)
        expected_labels = c.doc_labels(corp, sort=True)

        if return_doc_labels and return_vocab:
            assert isinstance(res, tuple)
            assert len(res) == 3
            dtm, doclabels, vocab = res
            assert isinstance(doclabels, list)
            assert isinstance(vocab, list)
        elif return_doc_labels and not return_vocab:
            assert isinstance(res, tuple)
            assert len(res) == 2
            dtm, doclabels = res
            assert isinstance(doclabels, list)
            vocab = None
        elif not return_doc_labels and return_vocab:
            assert isinstance(res, tuple)
            assert len(res) == 2
            dtm, vocab = res
            assert isinstance(vocab, list)
            doclabels = None
        else:
            dtm = res
            vocab = None
            doclabels = None

        assert dtm.ndim == 2
        assert len(corp) == dtm.shape[0] == len(expected_labels)
        assert len(expected_vocab) == dtm.shape[1] == len(expected_vocab)

        if as_table:
            assert isinstance(dtm, pd.DataFrame)
            assert dtm.index.tolist() == expected_labels
            assert dtm.columns.tolist() == expected_vocab

            if len(corp) > 0:
                assert np.sum(dtm.iloc[expected_labels.index('empty'), :]) == 0
                assert np.sum(dtm.iloc[:, expected_vocab.index('the')]) > 1
                assert dtm.iloc[expected_labels.index('small1'), expected_vocab.index('the')] == 1
        else:
            assert isinstance(dtm, csr_matrix)
            assert dtm.dtype is np.dtype(dtype or 'uint32')

            if len(corp) > 0:
                assert np.sum(dtm[expected_labels.index('empty'), :]) == 0
                assert np.sum(dtm[:, expected_vocab.index('the')]) > 1
                assert dtm[expected_labels.index('small1'), expected_vocab.index('the')] == 1

        if doclabels is not None:
            assert doclabels == expected_labels
        if vocab is not None:
            assert vocab == expected_vocab


@settings(deadline=None)
@given(n=st.integers(-1, 5),
       join=st.booleans(),
       join_str=st.text(string.printable, max_size=3))
def test_ngrams_hypothesis(corpora_en_serial_and_parallel_module, n, join, join_str):
    # note: proper ngram tests are done in test_tokenseq.py for token_ngrams
    for corp in corpora_en_serial_and_parallel_module:
        args = dict(n=n, join=join, join_str=join_str)

        if n < 2:
            with pytest.raises(ValueError):
                c.ngrams(corp, **args)
        else:
            res = c.ngrams(corp, **args)
            assert isinstance(res, dict)
            assert set(corp.keys()) == set(res.keys())

            for lbl, ng in res.items():
                dtok = corp[lbl]
                n_tok = len(dtok)
                assert isinstance(ng, list)

                if n_tok < n:
                    if n_tok == 0:
                        assert ng == []
                    else:
                        assert len(ng) == 1
                        if join:
                            assert ng == [join_str.join(dtok)]
                        else:
                            assert ng == [dtok]
                else:
                    if join:
                        assert all([isinstance(g, str) for g in ng])
                        assert all([join_str in g for g in ng])
                    else:
                        assert all([isinstance(g, list) for g in ng])
                        assert all([len(g) == n for g in ng])


@settings(deadline=None)
@given(search_term_exists=st.booleans(),
       context_size=st.one_of(st.integers(-1, 5), st.tuples(st.integers(-1, 5), st.integers(-1, 5))),
       by_attr=st.sampled_from([None, 'nonexistent', 'pos', 'lemma']),
       inverse=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos', 'mask', ['pos', 'mask']])),
       as_tables=st.booleans(),
       only_non_empty=st.booleans(),
       glue=st.one_of(st.none(), st.text(string.printable, max_size=3)),
       highlight_keyword=st.one_of(st.none(), st.text(string.printable, max_size=3)))
def test_kwic_hypothesis(corpora_en_serial_and_parallel_module, **args):
    search_term_exists = args.pop('search_term_exists')
    matchattr = args['by_attr'] or 'token'

    for corp in corpora_en_serial_and_parallel_module:
        if matchattr == 'token':
            vocab = list(c.vocabulary(corp))
        else:
            if matchattr != 'nonexistent':
                vocab = list(set(flatten_list([attrs[matchattr]
                                               for attrs in c.doc_tokens(corp, with_attr=matchattr).values()])))
            else:
                vocab = []

        if search_term_exists and len(vocab) > 0:
            s = random.choice(vocab)
        else:
            s = 'thisdoesnotexist'

        csize = args['context_size']
        if (isinstance(csize, int) and csize <= 0) or \
                (isinstance(csize, tuple) and (any(x < 0 for x in csize) or all(x == 0 for x in csize))):
            with pytest.raises(ValueError):
                c.kwic(corp, s, **args)
        elif args['glue'] is not None and args['with_attr']:
            with pytest.raises(ValueError):
                c.kwic(corp, s, **args)
        elif args['by_attr'] == 'nonexistent' and len(corp) > 0:
            with pytest.raises(AttributeError):
                c.kwic(corp, s, **args)
        else:
            res = c.kwic(corp, s, **args)
            assert isinstance(res, dict)

            if args['only_non_empty']:
                assert all([len(dkwic) > 0 for dkwic in res.values()])
            else:
                assert set(res.keys()) == set(corp.keys())

            res_windows = {}
            if args['as_tables']:
                for lbl, dkwic in res.items():
                    assert isinstance(dkwic, pd.DataFrame)

                    if len(dkwic) > 0:
                        if args['glue'] is None:
                            expected_cols = ['doc', 'context', 'position', matchattr]
                            if args['with_attr'] is True:
                                expected_cols.extend([a for a in c.Corpus.STD_TOKEN_ATTRS if a != args['by_attr']])
                            elif isinstance(args['with_attr'], list):
                                expected_cols.extend([a for a in args['with_attr'] if a != args['by_attr']])
                            if isinstance(args['with_attr'], str) and args['with_attr'] != args['by_attr']:
                                expected_cols.append(args['with_attr'])
                        else:
                            expected_cols = ['doc', 'context', matchattr]
                        assert dkwic.columns.tolist() == expected_cols

                        contexts = np.sort(np.unique(dkwic['context'])).tolist()
                        assert contexts == list(range(np.max(dkwic['context'])+1))
                    else:
                        contexts = []

                    dwindows = []
                    for ctx in contexts:
                        dkwic_ctx = dkwic.loc[dkwic['context'] == ctx, :]

                        if args['glue'] is None:
                            assert np.all(0 <= dkwic_ctx['position'])
                            assert np.all(dkwic_ctx['position'] < len(corp[lbl]))
                            dwindows.append(dkwic_ctx[matchattr].tolist())
                        else:
                            assert len(dkwic_ctx[matchattr]) == 1
                            dwindows.append(dkwic_ctx[matchattr].tolist()[0])

                    if dwindows or not args['only_non_empty']:
                        res_windows[lbl] = dwindows
            else:
                if args['with_attr']:
                    for lbl, dkwic in res.items():
                        assert lbl in corp.keys()
                        assert isinstance(dkwic, list)
                        for ctx in dkwic:
                            assert isinstance(ctx, dict)
                            expected_keys = {matchattr}
                            if args['with_attr'] is True:
                                expected_keys.update(c.Corpus.STD_TOKEN_ATTRS)
                            elif isinstance(args['with_attr'], list):
                                expected_keys.update(args['with_attr'])
                            elif isinstance(args['with_attr'], str):
                                expected_keys.add(args['with_attr'])
                            assert set(ctx.keys()) == expected_keys
                    res_windows = {lbl: [ctx[matchattr] for ctx in dkwic] for lbl, dkwic in res.items()}
                else:
                    res_windows = res

            if s in vocab:
                for win in res_windows.values():
                    for w in win:
                        if not args['inverse']:
                            if args['highlight_keyword'] is not None:
                                assert (args['highlight_keyword'] + s + args['highlight_keyword']) in w
                            else:
                                assert s in w

                            if args['glue'] is not None:
                                assert isinstance(w, str)
                                assert w.count(args['glue']) >= 2
                            else:
                                assert isinstance(w, list)
                                if isinstance(csize, int):
                                    assert 1 <= len(w) <= csize * 2 + 1
                                else:
                                    assert 1 <= len(w) <= sum(csize) + 1
            else:
                if args['only_non_empty']:
                    if args['inverse']:
                        assert len(res_windows) == max(len(corp) - 1, 0)   # -1 because of empty doc.
                    else:
                        assert len(res_windows) == 0
                else:
                    if args['inverse']:
                        assert all([len(win) > 0 for lbl, win in res_windows.items() if lbl != 'empty'])
                    else:
                        assert all([n == 0 for n in map(len, res_windows.values())])


def test_kwic_example(corpora_en_serial_and_parallel_module):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.kwic(corp, 'small', context_size=1)
        if len(corp) > 0:
            assert res['small1'] == res['empty'] == []
            assert res['small2'] == [['a', 'small', 'example']]

        res = c.kwic(corp, 'small', context_size=(0, 1))
        if len(corp) > 0:
            assert res['small1'] == res['empty'] == []
            assert res['small2'] == [['small', 'example']]

        res = c.kwic(corp, '*a*', match_type='glob', context_size=(0, 1))
        if len(corp) > 0:
            assert res['small1'] == res['empty'] == []
            assert res['small2'] == [['a', 'small'], ['small', 'example'], ['example', 'document']]

        res = c.kwic(corp, '*a*', match_type='glob', context_size=(0, 1), glue=' ', highlight_keyword='*')
        if len(corp) > 0:
            assert res['small1'] == res['empty'] == []
            assert res['small2'] == ['*a* small', '*small* example', '*example* document']

        res = c.kwic(corp, 'small', context_size=1, glue=' ')
        if len(corp) > 0:
            assert res['small1'] == res['empty'] == []
            assert res['small2'] == ['a small example']

        res = c.kwic(corp, 'small', context_size=1, glue=' ', only_non_empty=True)
        if len(corp) > 0:
            assert 'empty' not in res.keys()
            assert 'small1' not in res.keys()
            assert res['small2'] == ['a small example']


@settings(deadline=None)
@given(search_term_exists=st.booleans(),
       context_size=st.one_of(st.integers(-1, 5), st.tuples(st.integers(-1, 5), st.integers(-1, 5))),
       by_attr=st.sampled_from([None, 'nonexistent', 'pos', 'lemma']),
       inverse=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos', 'mask', ['pos', 'mask']])),
       glue=st.one_of(st.none(), st.text(string.printable, max_size=3)),
       highlight_keyword=st.one_of(st.none(), st.text(string.printable, max_size=3)))
def test_kwic_table_hypothesis(corpora_en_serial_and_parallel_module, **args):
    search_term_exists = args.pop('search_term_exists')
    matchattr = args['by_attr'] or 'token'

    for corp in corpora_en_serial_and_parallel_module:
        if matchattr == 'token':
            vocab = list(c.vocabulary(corp))
        else:
            if matchattr != 'nonexistent':
                vocab = list(set(flatten_list([attrs[matchattr]
                                               for attrs in c.doc_tokens(corp, with_attr=matchattr).values()])))
            else:
                vocab = []

        if search_term_exists and len(vocab) > 0:
            s = random.choice(vocab)
        else:
            s = 'thisdoesnotexist'

        csize = args['context_size']
        if (isinstance(csize, int) and csize <= 0) or \
                (isinstance(csize, tuple) and (any(x < 0 for x in csize) or all(x == 0 for x in csize))):
            with pytest.raises(ValueError):
                c.kwic_table(corp, s, **args)
        elif args['glue'] is not None and args['with_attr']:
            with pytest.raises(ValueError):
                c.kwic_table(corp, s, **args)
        elif args['by_attr'] == 'nonexistent' and len(corp) > 0:
            with pytest.raises(AttributeError):
                c.kwic_table(corp, s, **args)
        else:
            res = c.kwic_table(corp, s, **args)
            assert isinstance(res, pd.DataFrame)
            if args['glue'] is None:
                expected_cols = ['doc', 'context', 'position', matchattr]
                if args['with_attr'] is True:
                    expected_cols.extend([a for a in c.Corpus.STD_TOKEN_ATTRS if a != args['by_attr']])
                elif isinstance(args['with_attr'], list):
                    expected_cols.extend([a for a in args['with_attr'] if a != args['by_attr']])
                if isinstance(args['with_attr'], str) and args['with_attr'] != args['by_attr']:
                    expected_cols.append(args['with_attr'])
            else:
                expected_cols = ['doc', 'context', matchattr]
            assert res.columns.tolist() == expected_cols

            doclabels = set(res['doc'].unique())
            assert doclabels & set(corp.keys()) == doclabels
            for lbl in doclabels:
                dkwic = res.loc[res['doc'] == lbl, :]
                if args['glue'] is None:
                    contexts = np.sort(np.unique(dkwic['context'])).tolist()
                    assert contexts == list(range(max(contexts)+1))
                else:
                    contexts = dkwic['context'].tolist()
                    assert contexts == list(range(len(dkwic)))

                if len(dkwic) > 0:
                    assert np.issubdtype(dkwic[matchattr], object)

                    if args['glue'] is None:
                        assert np.all(0 <= dkwic['position'])
                        assert np.all(dkwic['position'] < len(corp[lbl]))

                        if not args['inverse']:
                            dkwic_tok = dkwic[matchattr].tolist()

                            if args['highlight_keyword']:
                                assert args['highlight_keyword'] + s + args['highlight_keyword'] in dkwic_tok
                            else:
                                assert s in dkwic_tok
                    else:
                        if len(corp[lbl]) > 1:
                            assert all([args['glue'] in x for x in dkwic[matchattr]])

                        if not args['inverse']:
                            assert all([s in x for x in dkwic[matchattr]])
                            if args['highlight_keyword']:
                                assert all([x.count(args['highlight_keyword']) == 2 for x in dkwic[matchattr]])


def test_save_load_corpus(corpora_en_serial_and_parallel_module):
    for corp in corpora_en_serial_and_parallel_module:
        with tempfile.TemporaryFile(suffix='.pickle') as ftemp:
            c.save_corpus_to_picklefile(corp, ftemp)
            ftemp.seek(0)
            unpickled_corp = c.load_corpus_from_picklefile(ftemp)

            _check_copies(corp, unpickled_corp, same_nlp_instance=False)


@settings(deadline=None)
@given(with_attr=st.booleans(),
       with_orig_corpus_opt=st.booleans(),
       pass_doc_attr_names=st.booleans(),
       pass_token_attr_names=st.booleans())
def test_load_corpus_from_tokens_hypothesis(corpora_en_serial_and_parallel_module, with_attr, with_orig_corpus_opt,
                                            pass_doc_attr_names, pass_token_attr_names):
    for corp in corpora_en_serial_and_parallel_module:
        if len(corp) > 0:
            doc_attrs = {'empty': 'yes', 'small1': 'yes', 'small2': 'yes'}
        else:
            doc_attrs = {}
        c.set_document_attr(corp, 'docattr_test', doc_attrs, default='no')
        c.set_token_attr(corp, 'tokenattr_test', {'the': True}, default=False)
        tokens = c.doc_tokens(corp, with_attr=with_attr)

        kwargs = {}
        if with_orig_corpus_opt:
            kwargs['spacy_instance'] = corp.nlp
            kwargs['max_workers'] = corp.max_workers
        else:
            kwargs['language'] = corp.language

        if pass_doc_attr_names:
            kwargs['doc_attr_names'] = ['docattr_test']
        if pass_token_attr_names:
            kwargs['token_attr_names'] = ['tokenattr_test']

        corp2 = c.load_corpus_from_tokens(tokens, **kwargs)
        assert len(corp) == len(corp2)
        assert corp2.language == 'en'

        # check if tokens are the same
        assert c.doc_tokens(corp) == c.doc_tokens(corp2)
        # check if token dataframes are the same
        corp_table = c.tokens_table(corp, with_attr=with_attr)
        corp2_table = c.tokens_table(corp2, with_attr=with_attr)
        cols = sorted(corp_table.columns.tolist())  # order of columns could be different
        assert cols == sorted(corp2_table.columns.tolist())
        assert _dataframes_equal(corp_table[cols], corp2_table[cols])

        if with_orig_corpus_opt:
            assert corp.nlp is corp2.nlp
            assert corp.max_workers == corp2.max_workers
        else:
            assert corp.nlp is not corp2.nlp


@pytest.mark.parametrize('with_orig_corpus_opt', (False, True))
def test_load_corpus_from_tokens_table(corpora_en_serial_and_parallel_module, with_orig_corpus_opt):
    for corp in corpora_en_serial_and_parallel_module:
        if len(corp) > 0:
            doc_attrs = {'empty': 'yes', 'small1': 'yes', 'small2': 'yes'}
        else:
            doc_attrs = {}
        c.set_document_attr(corp, 'docattr_test', doc_attrs, default='no')
        c.set_token_attr(corp, 'tokenattr_test', {'the': True}, default=False)
        tokenstab = c.tokens_table(corp, with_attr=True)

        kwargs = {}
        if with_orig_corpus_opt:
            kwargs['spacy_instance'] = corp.nlp
            kwargs['max_workers'] = corp.max_workers
        else:
            kwargs['language'] = corp.language

        corp2 = c.load_corpus_from_tokens_table(tokenstab, **kwargs)
        if len(corp) > 0:
            assert len(corp) - 1 == len(corp2)   # empty doc. not in result
        assert corp2.language == 'en'

        # check if tokens are the same
        assert c.doc_tokens(corp, only_non_empty=True) == c.doc_tokens(corp2)
        # check if token dataframes are the same
        assert _dataframes_equal(c.tokens_table(corp, with_attr=True), tokenstab)

        if with_orig_corpus_opt:
            assert corp.nlp is corp2.nlp
            assert corp.max_workers == corp2.max_workers
        else:
            assert corp.nlp is not corp2.nlp


@pytest.mark.parametrize('deepcopy_attrs', (False, True))
def test_serialize_deserialize_corpus(corpora_en_serial_and_parallel_module, deepcopy_attrs):
    for corp in corpora_en_serial_and_parallel_module:
        ser_corp = c.serialize_corpus(corp, deepcopy_attrs=deepcopy_attrs)
        assert isinstance(ser_corp, dict)
        corp2 = c.deserialize_corpus(ser_corp)
        assert isinstance(corp2, c.Corpus)

        _check_copies(corp, corp2, same_nlp_instance=False)


#%% helper functions


def _check_copies(corp_a, corp_b, same_nlp_instance):
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

    if same_nlp_instance:
        assert corp_a.nlp is corp_b.nlp
    else:
        assert corp_a.nlp is not corp_b.nlp
        assert corp_a.nlp.meta == corp_b.nlp.meta

    # check if token dataframes are the same
    assert _dataframes_equal(c.tokens_table(corp_a), c.tokens_table(corp_b))


def _dataframes_equal(df1, df2):
    return df1.shape == df2.shape and (df1 == df2).all(axis=1).sum() == len(df1)
