"""
Tests for tmtoolkit.corpus module.

Please see the special notes under "tests setup".

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""
import os.path
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

DATADIR = os.path.join('tests', 'data')
DATADIR_GUTENB = os.path.join(DATADIR, 'gutenberg')
DATADIR_WERTHER = os.path.join(DATADIR_GUTENB, 'werther')

textdata_en = textdata_sm['en']
textdata_de = textdata_sm['de']


#%% tests setup

# note: scope='module' means that fixture is created once per test module (not per test function run); this saves time,
# especially when using hypthesis, which runs many test cases; it's basically impractical to use hypothesis in
# conjunction with a per-test-function-run instance of a Corpus or a SpaCy instance, because it's too slow; hence,
# all tests that may modify an instance in-place (e.g. tests for filter functions) cannot be implemented with hypothesis

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


def test_datadirs():
    assert os.path.exists(DATADIR)
    assert os.path.exists(DATADIR_GUTENB)
    assert os.path.exists(DATADIR_WERTHER)


def test_fixtures_n_docs_and_doc_labels(corpus_en, corpus_de):
    assert corpus_en.n_docs == len(textdata_en)
    assert corpus_de.n_docs == len(textdata_de)

    assert set(corpus_en.doc_labels) == set(textdata_en.keys())
    assert set(corpus_de.doc_labels) == set(textdata_de.keys())


#%% test init


def test_corpus_no_lang_given():
    with pytest.raises(ValueError) as exc:
        c.Corpus({})

    assert str(exc.value) == 'either `language`, `language_model` or `spacy_instance` must be given'


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
    assert str(exc.value) == 'either `language`, `language_model` or `spacy_instance` must be given'

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
        elif args['select'] == 'empty' and args['only_non_empty']:
            with pytest.raises(ValueError) as exc:
                c.doc_tokens(corp, **args)
            assert str(exc.value).endswith('is empty but only non-empty documents should be retrieved')
        else:
            res = c.doc_tokens(corp, **args)

            if isinstance(args['select'], str):
                if args['as_tables']:
                    assert isinstance(res, pd.DataFrame)
                elif args['with_attr'] or args['with_mask'] or args['with_spacy_tokens']:
                    assert isinstance(res, dict)
                elif args['as_arrays']:
                    assert isinstance(res, np.ndarray)
                else:
                    assert isinstance(res, list)

                # wrap in dict for rest of test
                tmp = {args['select']: res}
                res = tmp
            else:
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


@pytest.mark.parametrize('collapse, set_override_text_collapse', [
    (None, False),
    (None, True),
    (' ', False),
    ('__', False),
    ('__', True),
])
def test_doc_texts(corpora_en_serial_and_parallel, collapse, set_override_text_collapse):
    # using corpora_en_serial_and_parallel instead of corpora_en_serial_and_parallel_module here since we're modifying
    # the Corpus objects using the override_text_collapse property

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

    for corp in corpora_en_serial_and_parallel:
        collapse_expect = collapse

        if set_override_text_collapse:
            assert corp.override_text_collapse is None
            corp.override_text_collapse = ' '
            if collapse is None:
                collapse_expect = ' '

        res = c.doc_texts(corp, collapse=collapse)
        assert isinstance(res, dict)
        assert set(res.keys()) == set(corp.keys())

        for lbl, txt in res.items():
            assert isinstance(txt, str)
            if collapse_expect is None:
                assert txt == textdata_en[lbl]
            else:
                if lbl in expected[collapse_expect]:
                    assert txt == expected[collapse_expect][lbl]


@pytest.mark.parametrize('proportions', [False, True])
def test_doc_frequencies(corpora_en_serial_and_parallel_module, proportions):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.doc_frequencies(corp, proportions=proportions)
        assert isinstance(res, dict)
        assert set(res.keys()) == c.vocabulary(corp)

        if len(corp) > 0:
            if proportions:
                assert all([0 < v <= 1 for v in res.values()])
                assert np.isclose(res['the'], 5/9)
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
                    assert set(c.Corpus.STD_TOKEN_ATTRS) <= set(cols)
                elif isinstance(args['with_attr'], str):
                    assert args['with_attr'] in cols
                elif isinstance(args['with_attr'], list):
                    assert set(args['with_attr']) <= set(cols)

                if args['with_mask']:
                    assert {'doc_mask', 'mask'} <= set(cols)


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
       glue=st.one_of(st.none(), st.text(string.printable, max_size=1)))
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
       join_str=st.text(string.printable, max_size=1))
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
       context_size=st.one_of(st.integers(-1, 3), st.tuples(st.integers(-1, 2), st.integers(-1, 2))),
       by_attr=st.sampled_from([None, 'nonexistent', 'pos', 'lemma']),
       inverse=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos', 'mask', ['pos', 'mask']])),
       as_tables=st.booleans(),
       only_non_empty=st.booleans(),
       glue=st.one_of(st.none(), st.text(string.printable, max_size=1)),
       highlight_keyword=st.one_of(st.none(), st.text(string.printable, max_size=1)))
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
       context_size=st.one_of(st.integers(-1, 3), st.tuples(st.integers(-1, 2), st.integers(-1, 2))),
       by_attr=st.sampled_from([None, 'nonexistent', 'pos', 'lemma']),
       inverse=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos', 'mask', ['pos', 'mask']])),
       glue=st.one_of(st.none(), st.text(string.printable, max_size=1)),
       highlight_keyword=st.one_of(st.none(), st.text(string.printable, max_size=1)))
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
            assert doclabels <= set(corp.keys())
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


@pytest.mark.parametrize('testtype, files, doc_label_fmt, inplace', [
    (1, os.path.join(DATADIR_GUTENB, 'kafka_verwandlung.txt'), '{path}-{basename}', True),
    (1, os.path.join(DATADIR_GUTENB, 'kafka_verwandlung.txt'), '{path}-{basename}', False),
    (2, {'testfile': os.path.join(DATADIR_GUTENB, 'kafka_verwandlung.txt')}, '{path}-{basename}', True),
    (3, [os.path.join(DATADIR_WERTHER, 'goethe_werther1.txt'), os.path.join(DATADIR_WERTHER, 'goethe_werther2.txt')],
     '{basename}', True),
    (4, [os.path.join(DATADIR_WERTHER, 'goethe_werther1.txt'), os.path.join(DATADIR_WERTHER, 'goethe_werther1.txt')],
     '{basename}', True),
])
def test_corpus_add_files_and_from_files(corpora_en_serial_and_parallel, testtype, files, doc_label_fmt, inplace):
    # make it a bit quicker by reading only 100 chars
    common_kwargs = dict(doc_label_fmt=doc_label_fmt, read_size=100, force_unix_linebreaks=False)

    ### test Corpus.from_files ###
    kwargs = dict(language='de', max_workers=1, **common_kwargs)               # Corpus constructor args
    if testtype == 4:
        with pytest.raises(ValueError) as exc:
            c.Corpus.from_files(files, **kwargs)
        assert str(exc.value).startswith('duplicate document label')
    else:
        corp = c.Corpus.from_files(files, **kwargs)
        assert isinstance(corp, c.Corpus)
        assert corp.language == 'de'
        assert corp.max_workers == 1

        doc_lbls = c.doc_labels(corp)

        if testtype == 1:
            assert len(doc_lbls) == 1
            assert 'kafka_verwandlung' in doc_lbls[0]
        elif testtype == 2:
            assert doc_lbls == ['testfile']
        elif testtype == 3:
            assert set(doc_lbls) == {'goethe_werther1', 'goethe_werther2'}
        else:
            raise ValueError(f'unknown testtype {testtype}')

    ### test corpus_add_files ###
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}
    kwargs = dict(inplace=inplace, **common_kwargs)
    for corp in corpora_en_serial_and_parallel:
        if testtype == 4:
            with pytest.raises(ValueError) as exc:
                c.corpus_add_files(corp, files, **kwargs)
            assert str(exc.value).startswith('duplicate document label')
        else:
            res = c.corpus_add_files(corp, files, **kwargs)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            if testtype == 1:
                assert any('kafka_verwandlung' in lbl for lbl in c.doc_labels(res))
            elif testtype == 2:
                assert 'testfile' in c.doc_labels(res)
            elif testtype == 3:
                assert 'goethe_werther1' in c.doc_labels(res)
                assert 'goethe_werther2' in c.doc_labels(res)
            else:
                raise ValueError(f'unknown testtype {testtype}')


@pytest.mark.parametrize('testtype, folder, valid_extensions, inplace', [
    (1, 'nonexistent', ('txt',), True),
    (2, DATADIR_GUTENB, ('txt',), True),
    (2, DATADIR_GUTENB, ('txt',), False),
    (2, DATADIR_GUTENB, ('txt', 'foo',), True),
    (3, DATADIR_GUTENB, ('foo',), True),
    (4, DATADIR_WERTHER, ('txt',), True),
])
def test_corpus_add_folder_and_from_folder(corpora_en_serial_and_parallel, testtype, folder, valid_extensions, inplace):
    if testtype == 1:
        expected_doclbls = None
    elif testtype in {2, 3}:
        expected_doclbls = {'kafka_verwandlung', 'werther-goethe_werther1', 'werther-goethe_werther2'}
    elif testtype == 4:
        expected_doclbls = {'goethe_werther1', 'goethe_werther2'}
    else:
        raise ValueError(f'unknown testtype {testtype}')

    # make it a bit quicker by reading only 100 chars
    common_kwargs = dict(valid_extensions=valid_extensions, read_size=100, force_unix_linebreaks=False)

    ### test Corpus.from_folder ###
    kwargs = dict(language='de', max_workers=1, **common_kwargs)               # Corpus constructor args

    if testtype == 1:
        with pytest.raises(IOError) as exc:
            c.Corpus.from_folder(folder, **kwargs)
        assert str(exc.value).startswith('path does not exist')
    else:
        corp = c.Corpus.from_folder(folder, **kwargs)
        assert isinstance(corp, c.Corpus)
        assert corp.language == 'de'
        assert corp.max_workers == 1

        doclbls = set(c.doc_labels(corp))

        if testtype in {2, 4}:
            assert doclbls == expected_doclbls
        elif testtype == 3:
            assert expected_doclbls & doclbls == set()

    ### test corpus_add_folder ###
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}
    kwargs = dict(inplace=inplace, **common_kwargs)
    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0

        if testtype == 1:
            with pytest.raises(IOError) as exc:
                c.corpus_add_folder(corp, folder, **kwargs)
            assert str(exc.value).startswith('path does not exist')
        else:
            res = c.corpus_add_folder(corp, folder, **kwargs)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            doclbls = set(c.doc_labels(res))

            if testtype == 2:
                if emptycorp:
                    assert expected_doclbls == doclbls
                else:
                    assert expected_doclbls < doclbls
            elif testtype == 3:
                assert expected_doclbls & doclbls == set()
            else:  # testtype == 4
                if emptycorp:
                    assert expected_doclbls == doclbls
                else:
                    assert expected_doclbls < doclbls


@pytest.mark.parametrize('testtype, files, id_column, text_column, prepend_columns, inplace', [
    (1, 'invalid_ext.foo', 0, 1, None, True),
    (2, os.path.join(DATADIR, '100NewsArticles.csv'), 'article_id', 'text', None, True),
    (2, os.path.join(DATADIR, '100NewsArticles.csv'), 'article_id', 'text', None, False),
    (2, os.path.join(DATADIR, '100NewsArticles.xlsx'), 'article_id', 'text', None, True),
    (3, os.path.join(DATADIR, '100NewsArticles.xlsx'), 'article_id', 'text', ['title', 'subtitle'], True),
    (4, [os.path.join(DATADIR, '100NewsArticles.csv'), os.path.join(DATADIR, '3ExampleDocs.xlsx')],
     'article_id', 'text', None, True),
])
def test_corpus_add_tabular_and_from_tabular(corpora_en_serial_and_parallel, testtype, files, id_column, text_column,
                                             prepend_columns, inplace):
    common_kwargs = dict(files=files, id_column=id_column, text_column=text_column, prepend_columns=prepend_columns)
    if testtype == 1:
        expected_doclbls = None
    elif testtype in {2, 3}:
        expected_doclbls = set(f'100NewsArticles-{i}' for i in range(1, 101))
    elif testtype == 4:
        expected_doclbls = set(f'100NewsArticles-{i}' for i in range(1, 101))
        expected_doclbls.update(f'3ExampleDocs-example{i}' for i in range(1, 4))
    else:
        raise ValueError(f'unknown testtype {testtype}')

    ### test Corpus.from_tabular ###
    kwargs = dict(language='de', max_workers=1, **common_kwargs)               # Corpus constructor args

    if testtype == 1:
        with pytest.raises(ValueError) as exc:
            c.Corpus.from_tabular(**kwargs)
        assert str(exc.value) == 'only file extensions ".csv", ".xls" and ".xlsx" are supported'
    else:
        corp = c.Corpus.from_tabular(**kwargs)
        assert isinstance(corp, c.Corpus)
        assert corp.language == 'de'
        assert corp.max_workers == 1

        if testtype in {2, 3}:
            assert len(corp) == 100
        else:  # testtype == 4
            assert len(corp) == 103

    ### test corpus_add_tabular ###
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}
    kwargs = dict(inplace=inplace, **common_kwargs)
    for corp in corpora_en_serial_and_parallel:
        n_docs_before = len(corp)

        if testtype == 1:
            with pytest.raises(ValueError) as exc:
                c.corpus_add_tabular(corp, **kwargs)
            assert str(exc.value) == 'only file extensions ".csv", ".xls" and ".xlsx" are supported'
        else:
            res = c.corpus_add_tabular(corp, **kwargs)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            doclbls = set(c.doc_labels(res))
            doctxts = c.doc_texts(res)

            if testtype in {2, 3}:
                assert len(res) == n_docs_before + 100
            else:  # testtype == 4
                assert len(res) == n_docs_before + 103

            if n_docs_before == 0:
                assert expected_doclbls == doclbls
            else:
                assert expected_doclbls <= doclbls

            if testtype in {2, 4}:
                assert doctxts['100NewsArticles-23'].startswith('The limited scope of')
                if testtype == 4:
                    assert doctxts['3ExampleDocs-example2'] == 'Second example document.'
            else:  # testtype == 3
                assert doctxts['100NewsArticles-23'].startswith('A vote for DeVos is a vote for resegregation\n\n'
                                                                'Felicia Wong is President and CEO of the Roosevelt '
                                                                'Institute, an economic and social policy think tank '
                                                                'working to re-imagine the rules so they work for all '
                                                                'Americans, and co-author of the forthcoming book '
                                                                '"Rewrite the Racial Rules: Building an Inclusive '
                                                                'American Economy." Randi Weingarten, President of the '
                                                                'American Federation of Teachers, is on Roosevelt\'s '
                                                                'board. The views expressed in this commentary are her '
                                                                'own.\n\nThe limited scope of')


@pytest.mark.parametrize('inplace', (True, False))
def test_corpus_add_zip_and_from_zip(corpora_en_serial_and_parallel, inplace):
    add_tabular_opts = dict(id_column='article_id', text_column='text')

    ### test Corpus.from_zip ###
    corp = c.Corpus.from_zip(os.path.join(DATADIR, 'zipdata.zip'), language='de', max_workers=1,
                             add_tabular_opts=add_tabular_opts)
    assert isinstance(corp, c.Corpus)
    assert corp.language == 'de'
    assert corp.max_workers == 1
    assert len(corp) == 101

    ### test corpus_add_zip ###
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}
    for corp in corpora_en_serial_and_parallel:
        n_docs_before = len(corp)

        res = c.corpus_add_zip(corp, os.path.join(DATADIR, 'zipdata.zip'), add_tabular_opts=add_tabular_opts,
                               inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del corp

        doclbls = c.doc_labels(res)

        assert len(res) == n_docs_before + 101

        assert sum(dl.startswith('100NewsArticles-') for dl in doclbls) == 100
        assert sum(dl == 'german-goethe_werther1' for dl in doclbls) == 1


@pytest.mark.parametrize('attrname, data, default, inplace', [
    ['is_small', {'empty': True, 'small1': True, 'small2': True}, False, True],
    ['is_small', {'empty': True, 'small1': True, 'small2': True}, False, False],
    ['is_small', {}, False, True],
    ['is_small', {}, False, False],
    ['is_empty', {'empty': 'yes'}, 'no', True],
    ['is_empty', {'empty': 'yes'}, 'no', False],
])
def test_set_remove_document_attr(corpora_en_serial_and_parallel_module, attrname, data, default, inplace):
    dont_check_attrs = {'doc_attrs', 'doc_attrs_defaults'}

    for corp in corpora_en_serial_and_parallel_module:
        if len(corp) > 0 or len(data) == 0:
            res = c.set_document_attr(corp, attrname=attrname, data=data, default=default, inplace=inplace)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            assert attrname in res.doc_attrs
            assert res.doc_attrs_defaults[attrname] == default
            assert Doc.has_extension(attrname)

            tok = c.doc_tokens(res, with_attr=attrname)

            for lbl, d in res.spacydocs.items():
                attrval = getattr(d._, attrname)
                tok_attrval = tok[lbl][attrname]
                if attrname == 'is_small':
                    if lbl in {'empty', 'small1', 'small2'} and len(data) > 0:
                        assert attrval is True
                        assert tok_attrval is True
                    else:
                        # attrval is None since a default value is corpus specific and can only be retrieved via
                        # doc_tokens() and the like
                        assert attrval is None
                        assert tok_attrval is False
                elif attrname == 'is_empty':
                    if lbl == 'empty':
                        assert attrval == 'yes'
                        assert tok_attrval == 'yes'
                    else:
                        # attrval is None since a default value is corpus specific and can only be retrieved via
                        # doc_tokens() and the like
                        assert attrval is None
                        assert tok_attrval == 'no'

            res2 = c.remove_document_attr(res, attrname, inplace=inplace)
            res2 = _check_corpus_inplace_modif(res, res2, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del res

            assert attrname not in res2.doc_attrs
            assert attrname not in res2.doc_attrs_defaults.keys()
            assert Doc.has_extension(attrname)   # is always retained

            if len(res2) > 0:
                with pytest.raises(AttributeError):   # this attribute doesn't exist anymore
                    c.doc_tokens(res2, with_attr=attrname)
        else:
            with pytest.raises(ValueError) as exc:
                c.set_document_attr(corp, attrname=attrname, data=data, default=default, inplace=inplace)
            assert 'does not exist in Corpus object `docs`' in str(exc.value)


@pytest.mark.parametrize('attrname, data, default, per_token_occurrence, inplace', [
    ['the_or_a', {'the': True, 'a': True}, False, True, True],
    ['the_or_a', {'the': True, 'a': True}, False, True, False],
    ['the_or_a', {}, False, True, True],
    ['the_or_a', {}, False, True, False],
    ['foobar_fail', {'small1': 'failure'}, '-', False, False],
    ['foobar', {'small1': ['foo'], 'small2': ['foo', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar']}, '-', False, False],
    ['foobar', {'small1': ['foo'], 'small2': ['foo', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar']}, '-', False, True],
])
def test_set_remove_token_attr(corpora_en_serial_and_parallel_module, attrname, data, default, per_token_occurrence,
                               inplace):
    dont_check_attrs = {'token_attrs', 'custom_token_attrs_defaults'}
    args = dict(attrname=attrname, data=data, default=default,
                per_token_occurrence=per_token_occurrence, inplace=inplace)

    for corp in corpora_en_serial_and_parallel_module:
        if attrname == 'foobar_fail' and len(corp) > 0:
            with pytest.raises(ValueError) as exc:
                c.set_token_attr(corp, **args)
            assert str(exc.value) == 'token attributes for document "small1" are neither tuple, list nor ' \
                                     'NumPy array'
        else:
            res = c.set_token_attr(corp, **args)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            assert attrname in res.token_attrs
            assert res.custom_token_attrs_defaults[attrname] == default

            tok = c.doc_tokens(res, with_attr=attrname)

            for lbl, d in res.spacydocs.items():
                assert attrname in d.user_data
                assert isinstance(d.user_data[attrname], np.ndarray)
                assert attrname in tok[lbl]
                assert d.user_data[attrname].tolist() == tok[lbl][attrname]
                assert len(tok[lbl]['token']) == len(tok[lbl][attrname])

                if per_token_occurrence:
                    if attrname == 'the_or_a':
                        if len(data) > 0:
                            for a, t in zip(tok[lbl][attrname], tok[lbl]['token']):
                                assert (t in {'the', 'a'}) == a
                        else:
                            assert all([a == default for a in tok[lbl][attrname]])
                else:
                    if attrname == 'foobar':
                        if lbl in {'small1', 'small2'}:
                            assert tok[lbl][attrname] == data[lbl]
                        else:
                            assert tok[lbl][attrname] == [default] * len(tok[lbl]['token'])

            res2 = c.remove_token_attr(res, attrname, inplace=inplace)
            res2 = _check_corpus_inplace_modif(res, res2, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del res

            assert attrname not in res2.token_attrs
            assert attrname not in res2.custom_token_attrs_defaults.keys()

            for d in res2.spacydocs_ignore_filter.values():
                assert attrname not in d.user_data

            if len(res2) > 0:
                with pytest.raises(AttributeError):   # this attribute doesn't exist anymore
                    c.doc_tokens(res2, with_attr=attrname)


@pytest.mark.parametrize('testcase, func, inplace', [
    ('identity', lambda x: x, True),
    ('identity', lambda x: x, False),
    ('upper', lambda x: x.upper(), True),
    ('upper', lambda x: x.upper(), False),
    ('lower', lambda x: x.lower(), True),
    ('lower', lambda x: x.lower(), False),
])
def test_transform_tokens_upper_lower(corpora_en_serial_and_parallel_module, testcase, func, inplace):
    dont_check_attrs = {'tokens_processed', 'is_processed'}

    for corp in corpora_en_serial_and_parallel_module:
        orig_tokens = c.doc_tokens(corp)

        if testcase == 'upper':
            trans_tokens = c.doc_tokens(c.to_uppercase(corp, inplace=False))
            expected = {lbl: [t.upper() for t in tok] for lbl, tok in orig_tokens.items()}
        elif testcase == 'lower':
            trans_tokens = c.doc_tokens(c.to_lowercase(corp, inplace=False))
            expected = {lbl: [t.lower() for t in tok] for lbl, tok in orig_tokens.items()}
        else:
            trans_tokens = None
            expected = None

        res = c.transform_tokens(corp, func, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del corp

        assert res.is_processed

        if testcase == 'identity':
            assert c.doc_tokens(res) == orig_tokens
        else:
            assert c.doc_tokens(res) == trans_tokens == expected


@pytest.mark.parametrize('testcase, chars, inplace', [
    ('nochars', [], True),
    ('nochars', [], False),
    ('fewchars', ['.', ','], True),
    ('fewchars', ['.', ','], False),
    ('punct', list(string.punctuation) + [' ', '\r', '\n', '\t'], True),
])
def test_remove_chars_or_punctuation(corpora_en_serial_and_parallel, testcase, chars, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_processed', 'is_processed'}

    for corp in corpora_en_serial_and_parallel:
        orig_vocab = c.vocabulary(corp)

        if testcase == 'punct':
            no_punct_vocab = c.vocabulary(c.remove_punctuation(corp, inplace=False))
        else:
            no_punct_vocab = None

        res = c.remove_chars(corp, chars=chars, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del corp

        assert res.is_processed
        vocab = c.vocabulary(res)

        if testcase == 'nochars':
            assert vocab == orig_vocab
        else:
            for t in vocab:
                assert not any([chr in t for chr in chars])

            if testcase == 'punct':
                assert vocab == no_punct_vocab


@pytest.mark.parametrize('inplace', [True, False])
def test_normalize_unicode(corpora_en_serial_and_parallel, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_processed', 'is_processed'}

    for corp in corpora_en_serial_and_parallel:
        orig_vocab = c.vocabulary(corp)
        orig_tok = c.doc_tokens(corp)

        res = c.normalize_unicode(corp, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del corp

        assert res.is_processed
        vocab = c.vocabulary(res)

        assert len(vocab) <= len(orig_vocab)

        if len(res) > 0:
            res_tok = c.doc_tokens(res)
            assert orig_tok['unicode1'][-1] != orig_tok['unicode1'][-3]
            assert res_tok['unicode1'][-1] == res_tok['unicode1'][-3]


@pytest.mark.parametrize('method, inplace', [
    ('icu', True),
    ('icu', False),
    ('ascii', True),
])
def test_simplify_unicode(corpora_en_serial_and_parallel, method, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_processed', 'is_processed'}

    for corp in corpora_en_serial_and_parallel:
        orig_vocab = c.vocabulary(corp)

        if method == 'icu' and not find_spec('PyICU'):
            with pytest.raises(RuntimeError) as exc:
                c.simplify_unicode(corp, method=method, inplace=inplace)
            assert str(exc.value) == 'package PyICU (https://pypi.org/project/PyICU/) must be installed to use this ' \
                                     'method'
        else:
            res = c.simplify_unicode(corp, method=method, inplace=inplace)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            assert res.is_processed
            vocab = c.vocabulary(res)

            assert len(vocab) <= len(orig_vocab)

            if len(res) > 0:
                res_tok = c.doc_tokens(res)
                if method == 'icu':
                    assert res_tok['unicode1'][-3:] == ['C', 'and', 'C']
                    assert res_tok['unicode2'][-5:] == ['C', 'C', 'e', 'ω', 'C']
                else:
                    assert res_tok['unicode1'][-3:] == ['C', 'and', 'C']
                    assert res_tok['unicode2'][-5:] == ['C', 'C', 'e', '', 'C']


@pytest.mark.parametrize('inplace', [True, False])
def test_lemmatize(corpora_en_serial_and_parallel, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_processed', 'is_processed'}

    for corp in corpora_en_serial_and_parallel:
        orig_lemmata = {lbl: tok['lemma'] for lbl, tok in c.doc_tokens(corp, with_attr='lemma').items()}
        res = c.lemmatize(corp, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del corp

        assert res.is_processed

        tok = c.doc_tokens(res)
        assert orig_lemmata == tok


@pytest.mark.parametrize('testcase, patterns, glue, match_type, return_joint_tokens, inplace', [
        (1, 'fail', '_', 'exact', False, True),
        (2, ['fail'], '_', 'exact', False, True),
        (3, ['is', 'a'], '_', 'exact', False, True),
        (4, ['is', 'a'], '_', 'exact', False, False),
        (5, ['is', 'a'], '_', 'exact', True, True),
        (6, ['is', 'a'], '_', 'exact', True, False),
        (7, ['on', 'the', 'law'], '_', 'exact', False, True),
        (8, ['Disney', '*'], '//', 'glob', False, True),
])
def test_join_collocations_by_patterns(corpora_en_serial_and_parallel, testcase, patterns, glue, match_type,
                                       return_joint_tokens, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_processed', 'is_processed'}
    args = dict(patterns=patterns, glue=glue, match_type=match_type, return_joint_tokens=return_joint_tokens,
                inplace=inplace)

    for corp in corpora_en_serial_and_parallel:
        if not isinstance(patterns, (list, tuple)) or len(patterns) < 2:
            with pytest.raises(ValueError) as exc:
                c.join_collocations_by_patterns(corp, **args)
            assert str(exc.value) == '`patterns` must be a list or tuple containing at least two elements'
        else:
            res = c.join_collocations_by_patterns(corp, **args)
            if return_joint_tokens:
                if inplace:
                    joint_colloc = res
                    res = None
                else:
                    res, joint_colloc = res
            else:
                joint_colloc = None

            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            assert res.is_processed

            tok = c.doc_tokens(res)

            if joint_colloc:
                assert isinstance(joint_colloc, set)

            if len(res) > 0:
                vocab = c.vocabulary(res)

                if joint_colloc:
                    assert len(joint_colloc) > 0

                if testcase in {3, 4, 5, 6}:
                    assert 'is_a' in vocab
                    assert tok['small2'][:2] == ['This', 'is_a']

                    if return_joint_tokens:
                        assert joint_colloc == {'is_a'}
                elif testcase == 7:
                    assert 'on_the_law' in vocab

                    if return_joint_tokens:
                        assert joint_colloc == {'on_the_law'}
                elif testcase == 8:
                    assert 'Disney//Parks' in vocab
                    assert 'Disney//park' in vocab
                    assert 'Disney//vacation' in vocab
                    assert 'Disney//World' in vocab
                    assert 'Disney//California' in vocab
                    assert 'Disney//will' in vocab
                    assert 'Disney//is' in vocab
                else:
                    raise RuntimeError('unknown testcase')
            else:
                assert tok == {}
                if joint_colloc:
                    assert joint_colloc == set()


@settings(deadline=None)
@given(threshold=st.integers(min_value=2, max_value=10),
       glue=st.text(string.printable, max_size=1),
       min_count=st.integers(min_value=0),
       embed_tokens_min_docfreq=st.one_of(st.none(), st.integers(min_value=1),
                                          st.floats(min_value=0, max_value=1, allow_nan=False)),
       pass_embed_tokens=st.integers(min_value=0, max_value=2),
       return_joint_tokens=st.booleans())
def test_join_collocations_by_statistic(corpora_en_serial_and_parallel_module, threshold, glue, min_count,
                                        embed_tokens_min_docfreq, pass_embed_tokens, return_joint_tokens):
    # restricting statistic to simple counts, otherwise the test takes too long
    args = dict(threshold=threshold, min_count=min_count, embed_tokens_min_docfreq=embed_tokens_min_docfreq,
                glue=glue, statistic=tokenseq.simple_collocation_counts)

    for corp in corpora_en_serial_and_parallel_module:
        if pass_embed_tokens > 0:
            vocab = list(c.vocabulary(corp))
            args['embed_tokens_set'] = random.choices(vocab, k=min(pass_embed_tokens, len(vocab)))
        else:
            args['embed_tokens_set'] = None

        colloc = c.corpus_collocations(corp, **args, return_statistic=False, rank=None, as_table=False)
        res = c.join_collocations_by_statistic(corp, **args, return_joint_tokens=return_joint_tokens,
                                               inplace=False)

        if return_joint_tokens:
            assert isinstance(res, tuple)
            assert len(res) == 2
            res, joint_tokens = res
        else:
            joint_tokens = None

        assert isinstance(res, c.Corpus)
        assert res is not corp
        assert res.is_processed

        assert all([glue in t for t in colloc])

        # TODO: once we support sentences in corpus_collocations, this should work
        #vocab = c.vocabulary(res)
        # assert set(colloc) <= vocab
        # if return_joint_tokens:
        #     assert joint_tokens == set(colloc)


@pytest.mark.parametrize('which, inplace', [
    ('fail', True),
    ('all', True),
    ('all', False),
    ('documents', True),
    ('tokens', True),
])
def test_reset_filter(corpora_en_serial_and_parallel, which, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'docs_filtered', 'tokens_filtered', 'is_filtered', 'n_docs', 'n_docs_masked', 'doc_labels'}

    for corp in corpora_en_serial_and_parallel:
        if which == 'fail':
            with pytest.raises(ValueError) as exc:
                c.reset_filter(corp, which=which, inplace=inplace)
            assert str(exc.value).startswith('`which` must be one of: ')
        else:
            orig_doclables = set(c.doc_labels(corp))
            orig_tok = c.doc_tokens(corp)
            orig_vocab = c.vocabulary(corp)

            if which in {'all', 'documents'}:
                c.filter_documents_by_label(corp, 'small*', match_type='glob', inplace=True)
                if len(corp) > 0:
                    assert corp.docs_filtered
                    assert set(c.doc_labels(corp)) == {'small1', 'small2'}

            if which in {'all', 'tokens'}:
                c.filter_tokens(corp, '[aeiou]', match_type='regex', inverse=True, inplace=True)
                assert corp.tokens_filtered

                if len(corp) > 0:
                    assert c.vocabulary(corp) <= orig_vocab
                    assert c.doc_tokens(corp) != orig_tok

            if len(corp) > 0:
                assert corp.is_filtered

            res = c.reset_filter(corp, which=which, inplace=inplace)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            if which == 'all':
                assert not res.is_filtered

            if which in {'all', 'documents'}:
                assert not res.docs_filtered
                assert set(c.doc_labels(res)) == orig_doclables
            if which in {'all', 'tokens'}:
                assert not res.tokens_filtered
                assert c.vocabulary(res) == orig_vocab
                assert c.doc_tokens(res) == orig_tok


@pytest.mark.parametrize('replace, inverse, inplace', [
    (False, False, True),
    (False, False, False),
    (True, False, True),
    (False, True, False),
    (True, True, False),
])
def test_filter_tokens_by_mask(corpora_en_serial_and_parallel, replace, inverse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_filtered', 'is_filtered'}

    mask1 = {'small2': [True, False, False, True, True, True, False]}
    mask2a = {'small2': [False, False, False, True, False, True, False]}
    mask2b = {'small2': [False, True, False, True]}
    mask2b_inv = {'small2': [True, False, True]}

    for corp in corpora_en_serial_and_parallel:
        if len(corp) == 0:
            with pytest.raises(ValueError) as exc:
                c.filter_tokens_by_mask(corp, mask=mask1, replace=replace, inverse=inverse, inplace=inplace)
            assert 'does not exist in Corpus object `docs` or is masked' in str(exc.value)

            with pytest.raises(ValueError) as exc:
                c.remove_tokens_by_mask(corp, mask=mask1, replace=replace, inplace=False)
            assert 'does not exist in Corpus object `docs` or is masked' in str(exc.value)
        else:
            res = c.filter_tokens_by_mask(corp, mask=mask1, replace=replace, inverse=inverse, inplace=inplace)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)

            assert res.is_filtered
            assert res.tokens_filtered

            tok = c.doc_tokens(res, select='small2')
            if inverse:
                assert tok == ['is', 'a', '.']
            else:
                assert tok == ['This', 'small', 'example', 'document']

            if inverse:
                assert not inplace
                res_inv = c.remove_tokens_by_mask(corp, mask=mask1, replace=replace, inplace=False)
                assert res_inv.is_filtered
                assert res_inv.tokens_filtered
                assert c.doc_tokens(res_inv, select='small2') == tok

            with pytest.raises(ValueError) as exc:
                c.filter_tokens_by_mask(res, mask=mask2b if replace else mask2a,
                                        replace=replace, inverse=inverse, inplace=inplace)
            assert str(exc.value).startswith('length of provided mask for document ')

            res2 = c.filter_tokens_by_mask(res, mask=mask2a if replace else (mask2b_inv if inverse else mask2b),
                                           replace=replace, inverse=inverse, inplace=inplace)
            res2 = _check_corpus_inplace_modif(res, res2, dont_check_attrs=dont_check_attrs, inplace=inplace)

            assert res2.is_filtered
            assert res2.tokens_filtered

            tok = c.doc_tokens(res2, select='small2')
            if inverse:
                if replace:
                    assert tok == ['This', 'is', 'a', 'example', '.']
                else:
                    assert tok == ['a']
            else:
                assert tok == ['small', 'document']


@pytest.mark.parametrize('testtype, search_tokens, by_attr, match_type, ignore_case, glob_method, inverse, inplace', [
    (1, 'the', None, 'exact', False, 'match', False, True),
    (1, 'the', None, 'exact', False, 'match', False, False),
    (2, 'the', None, 'exact', False, 'match', True, True),
    (3, 'the', None, 'exact', True, 'match', False, True),
    (3, ['the', 'The'], None, 'exact', False, 'match', False, True),
    (4, ' ', 'whitespace', 'exact', False, 'match', False, True),
    (5, 'Dis*', None, 'glob', False, 'match', False, True),
    (6, '*y*', None, 'glob', False, 'search', False, True),
    (5, '^Dis.*', None, 'regex', False, 'match', False, True),
    (7, True, 'is_the', 'exact', False, 'match', False, True),
])
def test_filter_tokens(corpora_en_serial_and_parallel, testtype, search_tokens, by_attr, match_type, ignore_case,
                       glob_method, inverse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_filtered', 'is_filtered'}

    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0

        if testtype == 7:
            c.set_token_attr(corp, 'is_the', {'the': True}, default=False)

        res = c.filter_tokens(corp, search_tokens, by_attr=by_attr, match_type=match_type, ignore_case=ignore_case,
                              glob_method=glob_method, inverse=inverse, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)

        vocab = c.vocabulary(res)

        if inverse:
            res_inv = c.remove_tokens(corp, search_tokens, by_attr=by_attr, match_type=match_type,
                                      ignore_case=ignore_case, glob_method=glob_method, inplace=False)
            vocab_inv = c.vocabulary(res_inv)
        else:
            vocab_inv = None

        if emptycorp:
            assert vocab == set()
        else:
            if testtype == 1:
                assert vocab == {'the'}
            elif testtype == 2:
                assert 'the' not in vocab
                assert vocab == vocab_inv
            elif testtype == 3:
                assert vocab == {'the', 'The'}
            elif testtype == 4:
                tokens_ws = c.doc_tokens(res, with_attr='whitespace')
                assert all([t is True for tok in tokens_ws.values() for t in tok['whitespace']])
            elif testtype == 5:
                assert all([t.startswith('Dis') for t in vocab])
            elif testtype == 6:
                assert all(['y' in t for t in vocab])
            elif testtype == 7:
                assert vocab == {'the'}
            else:
                raise ValueError(f'unknown testtype {testtype}')


@pytest.mark.parametrize('testtype, search_pos, simplify_pos, inverse, inplace', [
    (1, 'N', True, False, True),
    (1, 'N', True, False, False),
    (2, ['N', 'V'], True, False, True),
    (3, 'NOUN', False, False, True),
    (1, ['NOUN', 'PROPN'], False, False, True),
    (4, 'N', True, True, True),
])
def test_filter_for_pos(corpora_en_serial_and_parallel, testtype, search_pos, simplify_pos, inverse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_filtered', 'is_filtered'}

    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0
        res = c.filter_for_pos(corp, search_pos=search_pos, simplify_pos=simplify_pos, inverse=inverse, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)

        pos_flat = flatten_list([tok['pos'] for tok in c.doc_tokens(res, with_attr='pos').values()])

        if emptycorp:
            assert pos_flat == []
        else:
            pos_unique = set(pos_flat)

            if testtype == 1:
                assert pos_unique == {'PROPN', 'NOUN'}
            elif testtype == 2:
                assert pos_unique == {'PROPN', 'NOUN', 'VERB'}
            elif testtype == 3:
                assert pos_unique == {'NOUN'}
            elif testtype == 4:
                assert pos_unique & {'PROPN', 'NOUN'} == set()
            else:
                raise ValueError(f'unknown testtype {testtype}')


@pytest.mark.parametrize('testtype, which, df_threshold, proportions, return_filtered_tokens, inverse, inplace', [
    (1, 'common', 0.5, True, False, False, True),
    (1, 'common', 0.5, True, False, False, False),
    (1, '>=', 0.5, True, False, False, True),
    (1, '<', 0.5, True, False, True, True),
    (2, 'uncommon', 3, False, False, False, True),
    (2, 'uncommon', 3, False, True, False, True),
    (3, 'common', 0.7, True, False, True, True),
    (4, 'uncommon', 0.3, True, False, True, True),
])
def test_filter_tokens_by_doc_frequency(corpora_en_serial_and_parallel, testtype, which, df_threshold, proportions,
                                        return_filtered_tokens, inverse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_filtered', 'is_filtered'}

    for corp in corpora_en_serial_and_parallel:
        if testtype == 3:
            res_remove = c.remove_common_tokens(corp, df_threshold=df_threshold, proportions=proportions,
                                                inplace=False)
        elif testtype == 4:
            res_remove = c.remove_uncommon_tokens(corp, df_threshold=df_threshold, proportions=proportions,
                                                  inplace=False)
        else:
            res_remove = None

        if res_remove is not None:
            vocab_remove = c.vocabulary(res_remove)
        else:
            vocab_remove = None

        doc_freq = c.doc_frequencies(corp, proportions=proportions)
        res = c.filter_tokens_by_doc_frequency(corp, which=which, df_threshold=df_threshold, proportions=proportions,
                                               return_filtered_tokens=return_filtered_tokens, inverse=inverse,
                                               inplace=inplace)

        if return_filtered_tokens:
            if inplace:
                filt_tok = res
                res = None
            else:
                assert isinstance(res, tuple)
                assert len(res) == 2
                res, filt_tok = res
        else:
            filt_tok = None

        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        vocab = c.vocabulary(res)

        if testtype == 1:
            retained = {t for t, df in doc_freq.items() if df >= df_threshold}
        elif testtype == 2:
            retained = {t for t, df in doc_freq.items() if df <= df_threshold}
        elif testtype == 3:
            retained = {t for t, df in doc_freq.items() if df < df_threshold}
        elif testtype == 4:
            retained = {t for t, df in doc_freq.items() if df > df_threshold}
        else:
            raise ValueError(f'unknown testtype {testtype}')

        assert vocab == retained

        if testtype in {3, 4}:
            assert vocab_remove is not None
            assert vocab_remove == retained

        if return_filtered_tokens:
            assert isinstance(filt_tok, set)
            assert filt_tok == retained


@pytest.mark.parametrize('testtype, search_tokens, by_attr, matches_threshold, match_type, ignore_case, glob_method, '
                         'inverse_result, inverse_matches, inplace', [
    (1, 'the', None, 1, 'exact', False, 'match', False, False, True),
    (1, 'the', None, 1, 'exact', False, 'match', False, False, False),
    (2, 'the', None, 1, 'exact', False, 'match', True, False, True),
    (3, 'the', None, 2, 'exact', False, 'match', False, True, True),
    (4, 'Dis*', None, 1, 'glob', False, 'match', False, False, True),
    (4, '^Dis.*', None, 1, 'regex', False, 'match', False, False, True),
    (5, ['example', 'document'], None, 2, 'exact', False, 'match', False, False, True),
    (1, True, 'is_the', 1, 'exact', False, 'match', False, False, True),
])
def test_filter_documents(corpora_en_serial_and_parallel, testtype, search_tokens, by_attr, matches_threshold,
                          match_type, ignore_case, glob_method, inverse_result, inverse_matches, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'docs_filtered', 'tokens_filtered', 'is_filtered', 'doc_labels', 'n_docs', 'n_docs_masked'}

    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0

        if by_attr == 'is_the':
            c.set_token_attr(corp, by_attr, {'the': True}, default=False)

        doctok_before = c.doc_tokens(corp)

        if inverse_result:
            res_rem = c.remove_documents(corp, search_tokens=search_tokens, by_attr=by_attr,
                                         matches_threshold=matches_threshold, match_type=match_type,
                                         ignore_case=ignore_case, glob_method=glob_method,
                                         inverse_matches=inverse_matches, inplace=False)
            doctok_rem = c.doc_tokens(res_rem)
        else:
            doctok_rem = None

        res = c.filter_documents(corp, search_tokens=search_tokens, by_attr=by_attr,
                                 matches_threshold=matches_threshold, match_type=match_type, ignore_case=ignore_case,
                                 glob_method=glob_method, inverse_result=inverse_result,
                                 inverse_matches=inverse_matches, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        doctok = c.doc_tokens(res)
        removed_docs = set(doctok_before.keys()) - set(doctok.keys())

        if inverse_result:
            assert doctok_rem == doctok

        if emptycorp:
            assert len(removed_docs) == 0
        else:
            assert len(removed_docs) > 0

            if testtype == 1:
                assert len(removed_docs) == 4
                for tok in doctok.values():
                    assert 'the' in tok
                for lbl in removed_docs:
                    assert 'the' not in doctok_before[lbl]
            elif testtype == 2:
                assert len(removed_docs) == len(doctok_before) - 4
                for tok in doctok.values():
                    assert 'the' not in tok
                for lbl in removed_docs:
                    assert 'the' in doctok_before[lbl]
            elif testtype == 3:
                for tok in doctok.values():
                    assert sum(t != 'the' for t in tok) >= 2
                for lbl in removed_docs:
                    assert sum(t != 'the' for t in doctok_before[lbl]) < 2
            elif testtype == 4:
                for tok in doctok.values():
                    assert any([t.startswith('Dis') for t in tok])
                for lbl in removed_docs:
                    assert all([not t.startswith('Dis') for t in doctok_before[lbl]])
            elif testtype == 5:
                for tok in doctok.values():
                    assert 'example' in tok
                    assert 'document' in tok
            else:
                raise ValueError(f'unknown testtype {testtype}')


@pytest.mark.parametrize('testtype, search_tokens, by_attr, match_type, ignore_case, glob_method, inverse, inplace', [
    (1, 'empty', 'label', 'exact', False, 'match', False, True),
    (1, 'empty', 'label', 'exact', False, 'match', False, False),
    (2, 'empty', 'label', 'exact', False, 'match', True, True),
    (3, ['small1', 'small2'], 'label', 'exact', False, 'match', False, True),
    (3, ['small1', 'small2', 'nonexistent'], 'label', 'exact', False, 'match', False, True),
    (3, 'small*', 'label', 'glob', False, 'match', False, True),
    (3, '^small.*', 'label', 'regex', False, 'match', False, True),
    (3, True, 'is_small', 'exact', False, 'match',  False, True),
])
def test_filter_documents_by_docattr(corpora_en_serial_and_parallel, testtype, search_tokens, by_attr, match_type,
                                     ignore_case, glob_method, inverse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'docs_filtered', 'tokens_filtered', 'is_filtered', 'doc_labels', 'n_docs', 'n_docs_masked'}

    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0
        doclabels_before = set(c.doc_labels(corp))

        if by_attr == 'is_small':
            if emptycorp:
                c.set_document_attr(corp, by_attr, {}, default=False)
            else:
                c.set_document_attr(corp, by_attr, {'small1': True, 'small2': True}, default=False)
            doclabels_by_label = None
        else:  # by document label
            res_by_label = c.filter_documents_by_label(corp, search_tokens=search_tokens, match_type=match_type,
                                                       ignore_case=ignore_case, glob_method=glob_method,
                                                       inverse=inverse, inplace=False)
            doclabels_by_label = set(c.doc_labels(res_by_label))

        doclabels_by_label_rem = None

        if inverse:
            res_rem = c.remove_documents_by_docattr(corp, search_tokens=search_tokens, by_attr=by_attr,
                                                    match_type=match_type, ignore_case=ignore_case,
                                                    glob_method=glob_method, inplace=False)
            doclabels_rem = set(c.doc_labels(res_rem))

            if by_attr == 'label':
                res_by_label_rem = c.remove_documents_by_label(corp, search_tokens=search_tokens, match_type=match_type,
                                                               ignore_case=ignore_case, glob_method=glob_method,
                                                               inplace=False)
                doclabels_by_label_rem = set(c.doc_labels(res_by_label_rem))
        else:
            doclabels_rem = None

        res = c.filter_documents_by_docattr(corp, search_tokens=search_tokens, by_attr=by_attr, match_type=match_type,
                                            ignore_case=ignore_case, glob_method=glob_method, inverse=inverse,
                                            inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)

        doclabels = set(c.doc_labels(res))

        if doclabels_by_label is not None:
            assert doclabels == doclabels_by_label
        if doclabels_by_label_rem is not None:
            assert doclabels_rem == doclabels_by_label_rem

        removed_docs = doclabels_before - doclabels

        if emptycorp:
            assert len(removed_docs) == 0
        else:
            assert len(removed_docs) > 0

        if inverse:
            assert doclabels == doclabels_rem

        if emptycorp:
            assert doclabels == set()
        else:
            if testtype == 1:
                assert doclabels == {'empty'}
            elif testtype == 2:
                assert doclabels == doclabels_before - {'empty'}
            elif testtype == 3:
                assert doclabels == {'small1', 'small2'}
            else:
                raise ValueError(f'unknown testtype {testtype}')


@pytest.mark.parametrize('testtype, relation, threshold, inverse, inplace', [
    (1, '>', 0, False, True),
    (1, '>', 0, False, False),
    (1, '>=', 1, False, True),
    (2, '>=', 8, False, True),
    (3, '==', 0, False, True),
    (3, '>', 0, True, True),
])
def test_filter_documents_by_length(corpora_en_serial_and_parallel, testtype, relation, threshold, inverse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'docs_filtered', 'tokens_filtered', 'is_filtered', 'doc_labels', 'n_docs', 'n_docs_masked'}

    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0
        doclabels_before = set(c.doc_labels(corp))

        if inverse:
            res_rem = c.remove_documents_by_length(corp, relation=relation, threshold=threshold, inplace=False)
            doclabels_rem = set(c.doc_labels(res_rem))
        else:
            doclabels_rem = None

        res = c.filter_documents_by_length(corp, relation=relation, threshold=threshold, inverse=inverse,
                                           inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        doclabels = set(c.doc_labels(res))

        if inverse:
            assert doclabels_rem == doclabels

        if emptycorp:
            assert doclabels == set()
        else:
            if testtype == 1:
                assert doclabels == doclabels_before - {'empty'}
            elif testtype == 2:
                assert doclabels == doclabels_before - {'empty', 'small1', 'small2'}
            elif testtype == 3:
                assert doclabels == {'empty'}
            else:
                raise ValueError(f'unknown testtype {testtype}')


@pytest.mark.parametrize('remove_punct, remove_stopwords, remove_empty, remove_shorter_than, remove_longer_than, '
                         'remove_numbers, inplace', [
    (True, False, False, None, None, False, True),
    (True, False, False, None, None, False, False),
    (False, True, False, None, None, False, True),
    (False, ['the', 'a'], False, None, None, False, True),
    (False, False, True, None, None, False, True),
    (False, False, False, 5, None, False, True),
    (False, False, False, None, 5, False, True),
    (False, False, False, 5, 10, False, True),
    (False, False, False, None, None, True, True),
])
def test_filter_clean_tokens(corpora_en_serial_and_parallel, remove_punct, remove_stopwords, remove_empty,
                             remove_shorter_than, remove_longer_than, remove_numbers, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_filtered', 'is_filtered'}

    for corp in corpora_en_serial_and_parallel:
        vocab_before = c.vocabulary(corp)
        res = c.filter_clean_tokens(corp,
                                    remove_punct=remove_punct,
                                    remove_stopwords=remove_stopwords,
                                    remove_empty=remove_empty,
                                    remove_shorter_than=remove_shorter_than,
                                    remove_longer_than=remove_longer_than,
                                    remove_numbers=remove_numbers,
                                    inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)

        vocab = c.vocabulary(res)
        assert len(vocab) <= len(vocab_before)

        if remove_punct:
            for tok in c.doc_tokens(res, with_attr='is_punct').values():
                assert all([attrval is False for attrval in tok['is_punct']])

        if remove_stopwords is True:
            for tok in c.doc_tokens(res, with_attr='is_stop').values():
                assert all([attrval is False for attrval in tok['is_stop']])
        elif isinstance(remove_stopwords, list):
            assert set(remove_stopwords) & vocab == set()

        if remove_empty:
            assert '' not in vocab

        if remove_shorter_than is not None:
            assert all([len(t) >= remove_shorter_than for t in vocab])

        if remove_longer_than is not None:
            assert all([len(t) <= remove_longer_than for t in vocab])

        if remove_numbers:
            for tok in c.doc_tokens(res, with_attr='like_num').values():
                assert all([attrval is False for attrval in tok['like_num']])


@pytest.mark.parametrize('testtype, search_tokens, context_size, by_attr, match_type, ignore_case, glob_method, '
                         'inverse, inplace', [
    (1, 'the', 1, None, 'exact', False, 'match', False, True),
    (1, 'the', 1, None, 'exact', False, 'match', False, False),
    (2, 'example', 2, None, 'exact', False, 'match', False, True),
    (3, 'example', (2, 1), None, 'exact', False, 'match', False, True),
    (1, True, 1, 'is_the', 'exact', False, 'match', False, True),
    (4, 'Dis*', 1, None, 'glob', False, 'match', False, True),
    (4, '^Dis.*', 1, None, 'regex', False, 'match', False, True),
    (5, 'the', 1, None, 'exact', False, 'match', True, True),
])
def test_filter_tokens_with_kwic(corpora_en_serial_and_parallel, testtype, search_tokens, context_size, by_attr,
                                 match_type, ignore_case, glob_method, inverse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'tokens_filtered', 'is_filtered'}

    for corp in corpora_en_serial_and_parallel:
        doctok_before = c.doc_tokens(corp)

        if by_attr == 'is_the':
            c.set_token_attr(corp, by_attr, {'the': True}, default=False)

        res = c.filter_tokens_with_kwic(corp, search_tokens=search_tokens, context_size=context_size, by_attr=by_attr,
                                        match_type=match_type, ignore_case=ignore_case, glob_method=glob_method,
                                        inverse=inverse, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)

        doctok = c.doc_tokens(res)

        if testtype == 1:
            for lbl, tok in doctok.items():
                tok_before = doctok_before[lbl]
                if 'the' in tok_before:
                    assert 'the' in tok
                if lbl == 'NewsArticles-1':
                    assert tok[:3] == ['over', 'the', 'weekend']
        elif testtype == 2:
            for lbl, tok in doctok.items():
                tok_before = doctok_before[lbl]
                if 'example' in tok_before:
                    assert 'example' in tok
                if lbl == 'small2':
                    assert tok == ['a', 'small', 'example', 'document', '.']
        elif testtype == 3:
            for lbl, tok in doctok.items():
                tok_before = doctok_before[lbl]
                if 'example' in tok_before:
                    assert 'example' in tok
                if lbl == 'small2':
                    assert tok == ['a', 'small', 'example', 'document']
        elif testtype == 4:
            for lbl, tok in doctok.items():
                tok_before = doctok_before[lbl]
                if 'Disney' in tok_before:
                    assert 'Disney' in tok
        elif testtype == 5:
            for lbl, tok in doctok.items():
                tok_before = doctok_before[lbl]
                if 'the' in tok_before:
                    assert 'the' not in tok
        else:
            raise ValueError(f'unknown testtype {testtype}')


@pytest.mark.parametrize('testtype, which, override_text_collapse, inplace', [
    (1, 'all', True, True),
    (1, 'all', False, True),
    (1, 'all', '_', True),
    (1, 'all', True, False),
    (2, 'all', True, True),
    (2, 'tokens', True, True),
    (2, 'tokens', False, True),
    (3, 'all', True, True),
    (3, 'documents', True, True),
    (3, 'documents', False, True),
])
def test_compact(corpora_en_serial_and_parallel, testtype, which, override_text_collapse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'docs_filtered', 'tokens_filtered', 'is_filtered', 'doc_labels', 'n_docs', 'n_docs_masked',
                        'workers_docs'}

    for corp in corpora_en_serial_and_parallel:
        assert corp.override_text_collapse is None

        if len(corp) > 0:
            doclabels_before = set(c.doc_labels(corp))

            if testtype <=2:
                c.filter_tokens_by_mask(corp, {'small2': [True, False, False, True, True, True, False]})

            if testtype != 2:
                c.filter_documents_by_mask(corp, {'small1': False})
                assert corp.n_docs_masked == 1

            res = c.compact(corp, which=which, override_text_collapse=override_text_collapse, inplace=inplace)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            doclabels = set(c.doc_labels(res))
            doctok = c.doc_tokens(res)

            assert res.n_docs_masked == 0

            if override_text_collapse is True:
                if testtype <= 2 and which in {'all', 'tokens'}:
                    assert res.override_text_collapse == ' '
                else:
                    assert res.override_text_collapse is None
            elif isinstance(override_text_collapse, str):
                assert res.override_text_collapse == override_text_collapse
            else:  # is False
                assert res.override_text_collapse is None

            c.reset_filter(res)   # shouldn't have any effect

            if testtype == 1 or testtype == 2:
                assert doctok['small2'] == ['This', 'small', 'example', 'document']

            if testtype == 1 or testtype == 3:
                assert doclabels == doclabels_before - {'small1'}


@pytest.mark.parametrize('n, join_str, inplace', [
    (2, ' ', True),
    (2, ' ', False),
    (2, '_', True),
    (3, '//', True),
    (1, ' ', True),
])
def test_ngramify(corpora_en_serial_and_parallel, n, join_str, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'uses_unigrams', 'ngrams', 'ngrams_join_str'}

    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0
        vocab_before = c.vocabulary(corp)
        doctok_before = c.doc_tokens(corp)

        if n > 1:
            ngrams = c.ngrams(corp, n=n, join=True, join_str=join_str)
        else:
            ngrams = doctok_before

        res = c.ngramify(corp, n=n, join_str=join_str, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)

        assert res.uses_unigrams == (n == 1)
        assert res.ngrams == n
        assert res.ngrams_join_str == join_str
        assert c.doc_tokens(res) == ngrams
        assert c.doc_tokens(res, force_unigrams=True) == doctok_before

        if n > 1 and not emptycorp:
            assert c.vocabulary(res, force_unigrams=False) != vocab_before
        else:
            assert c.vocabulary(res, force_unigrams=False) == vocab_before

        assert c.vocabulary(res, force_unigrams=True) == vocab_before


#%% other functions

@pytest.mark.parametrize('with_paths', [False, True])
def test_builtin_corpora_info(with_paths):
    corpinfo = c.builtin_corpora_info(with_paths=with_paths)
    if with_paths:
        assert isinstance(corpinfo, dict)
        corpnames = list(corpinfo.keys())
        for name, path in corpinfo.items():
            namecomponents = name.split('-')
            assert path.endswith(f'/data/{namecomponents[0]}/{"-".join(namecomponents[1:])}.zip')
    else:
        assert isinstance(corpinfo, list)
        corpnames = corpinfo

    assert set(corpnames) == set(c.Corpus._BUILTIN_CORPORA_LOAD_KWARGS.keys())

    # TODO: instantiate all corpora


#%% workflow examples tests


def test_corpus_workflow_example1(corpora_en_serial_and_parallel):
    for corp_orig in corpora_en_serial_and_parallel:
        emptycorp = len(corp_orig) == 0

        c.set_token_attr(corp_orig, 'tokbar', {'a': True}, default=False)

        if emptycorp:
            c.set_document_attr(corp_orig, 'docfoo', {}, default='no')
        else:
            c.set_document_attr(corp_orig, 'docfoo', {'small1': 'yes', 'NewsArticles-1': 'yes'}, default='no')

            toktbl = c.tokens_table(corp_orig)

            assert np.all(toktbl.loc[toktbl.doc.isin({'small1', 'NewsArticles-1'}), 'docfoo'] == 'yes')
            assert np.all(toktbl.loc[~toktbl.doc.isin({'small1', 'NewsArticles-1'}), 'docfoo'] == 'no')

            assert np.all(toktbl.loc[toktbl.token == 'a', 'tokbar'])
            assert np.all(~toktbl.loc[toktbl.token != 'a', 'tokbar'])

        assert 'tokbar' in corp_orig.token_attrs
        assert 'docfoo' in corp_orig.doc_attrs

        corp = c.filter_documents_by_length(corp_orig, '>', 10, inplace=False)

        if emptycorp:
            assert len(corp) == corp.n_docs_masked == 0
        else:
            assert len(corp) == 6
            assert corp.n_docs_masked == 3

        c.compact(corp)

        if emptycorp:
            assert len(corp) == corp.n_docs_masked == 0
        else:
            assert len(corp) == 6
            assert corp.n_docs_masked == 0

        c.lemmatize(corp)

        toktbl = c.tokens_table(corp)

        if not emptycorp:
            assert np.all(toktbl.token == toktbl.lemma)

            assert np.all(toktbl.loc[toktbl.doc.isin({'small1', 'NewsArticles-1'}), 'docfoo'] == 'yes')
            assert np.all(toktbl.loc[~toktbl.doc.isin({'small1', 'NewsArticles-1'}), 'docfoo'] == 'no')

        assert 'tokbar' in corp.token_attrs
        assert 'docfoo' in corp.doc_attrs

        c.to_lowercase(corp)

        toktbl = c.tokens_table(corp)

        if not emptycorp:
            assert np.all(toktbl.token == toktbl.lemma.str.lower())

        c.filter_clean_tokens(corp, remove_shorter_than=2)

        # shouldn't really do that with so few docs, but this is just for testing
        c.remove_common_tokens(corp, df_threshold=6, proportions=False)
        c.remove_uncommon_tokens(corp, df_threshold=1, proportions=False)

        if not emptycorp:
            assert np.all(toktbl.loc[toktbl.doc.isin({'small1', 'NewsArticles-1'}), 'docfoo'] == 'yes')
            assert np.all(toktbl.loc[~toktbl.doc.isin({'small1', 'NewsArticles-1'}), 'docfoo'] == 'no')

        assert 'tokbar' in corp.token_attrs
        assert 'docfoo' in corp.doc_attrs

        corp_final = c.compact(corp, inplace=False)

        dtm_final = c.dtm(corp_final)
        assert np.all(dtm_final.todense() == c.dtm(corp).todense())
        assert dtm_final.shape == (len(corp_final), c.vocabulary_size(corp_final))


#%% helper functions


def _check_corpus_inplace_modif(corp_a, corp_b, inplace, check_attrs=None, dont_check_attrs=None):
    if inplace:
        assert corp_b is None

        return corp_a
    else:
        assert isinstance(corp_b, c.Corpus)
        assert corp_a is not corp_b
        _check_copies_attrs(corp_a, corp_b, check_attrs=check_attrs, dont_check_attrs=dont_check_attrs)

        return corp_b


def _check_copies(corp_a, corp_b, same_nlp_instance):
    _check_copies_attrs(corp_a, corp_b, same_nlp_instance=same_nlp_instance)

    # check if tokens are the same
    tok_a = c.doc_tokens(corp_a)
    tok_b = c.doc_tokens(corp_b)
    assert tok_a == tok_b

    # check if token dataframes are the same
    assert _dataframes_equal(c.tokens_table(corp_a), c.tokens_table(corp_b))


def _check_copies_attrs(corp_a, corp_b, check_attrs=None, dont_check_attrs=None, same_nlp_instance=True):
    attrs_a = dir(corp_a)
    attrs_b = dir(corp_b)

    # check if simple attributes are the same
    if check_attrs is None:
        check_attrs = {'docs_filtered', 'tokens_filtered', 'is_filtered', 'tokens_processed',
                       'is_processed', 'uses_unigrams', 'token_attrs', 'custom_token_attrs_defaults', 'doc_attrs',
                       'doc_attrs_defaults', 'ngrams', 'ngrams_join_str', 'language', 'language_model',
                       'doc_labels', 'n_docs', 'n_docs_masked', 'ignore_doc_filter', 'workers_docs',
                       'max_workers'}

    if dont_check_attrs is not None:
        check_attrs.difference_update(dont_check_attrs)

    for attr in check_attrs:
        assert attr in attrs_a
        assert attr in attrs_b
        assert getattr(corp_a, attr) == getattr(corp_b, attr)

    if same_nlp_instance:
        assert corp_a.nlp is corp_b.nlp
    else:
        assert corp_a.nlp is not corp_b.nlp
        assert corp_a.nlp.meta == corp_b.nlp.meta


def _dataframes_equal(df1, df2):
    return df1.shape == df2.shape and (df1 == df2).all(axis=1).sum() == len(df1)
