"""
Tests for tmtoolkit.corpus module.

Please see the special notes under "tests setup".

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import math
import os.path
import random
import re
import string
import tempfile
import multiprocessing
from importlib.util import find_spec
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings

if any(find_spec(pkg) is None for pkg in ('spacy', 'bidict', 'loky')):
    pytest.skip("skipping tmtoolkit.corpus tests (required packages not installed)", allow_module_level=True)

import spacy
from spacy.tokens import Doc
from spacy.util import get_installed_models
from scipy.sparse import csr_matrix

from tmtoolkit import tokenseq
from tmtoolkit.utils import flatten_list
from tmtoolkit.corpus._common import LANGUAGE_LABELS, TOKENMAT_ATTRS, STD_TOKEN_ATTRS
TOKENMAT_ATTRS = TOKENMAT_ATTRS - {'whitespace', 'token', 'sent_start'}
from tmtoolkit import corpus as c
from ._testtools import strategy_str_str_dict_printable
from ._testtextdata import textdata_sm

DATADIR = os.path.join('tests', 'data')
DATADIR_GUTENB = os.path.join(DATADIR, 'gutenberg')
DATADIR_WERTHER = os.path.join(DATADIR_GUTENB, 'werther')

installed_lang = set(model[:2] for model in get_installed_models())
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
    with pytest.raises(ValueError, match='either `language`, `language_model` or `spacy_instance` must be given'):
        c.Corpus({})


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
    with pytest.raises(ValueError, match=r'is not supported'):
        c.Corpus(textdata_en, language='00')

    with pytest.raises(ValueError, match=r'`language` must be a two-letter ISO 639-1 language code'):
        c.Corpus(textdata_en, language='fail')

    with pytest.raises(ValueError, match=r'either `language`, `language_model` or `spacy_instance` must be given'):
        c.Corpus(textdata_en)

    corp = c.Corpus(textdata_en, language='en')
    assert corp.has_sents
    _check_corpus_docs(corp, has_sents=True)
    assert corp.language_model == 'en_core_web_sm'
    assert 'ner' not in corp.nlp.pipe_names
    assert 'senter' not in corp.nlp.pipe_names
    assert isinstance(corp._spacy_opts['config']['nlp'], dict)

    _check_copies(corp, copy(corp), same_nlp_instance=True)
    _check_copies(corp, deepcopy(corp), same_nlp_instance=False)

    corp = c.Corpus(textdata_en, language='en', load_features=[])
    assert not corp.has_sents
    assert corp.language_model == 'en_core_web_sm'
    assert 'senter' not in corp.nlp.pipe_names
    assert 'tagger' not in corp.nlp.pipe_names
    assert 'parser' not in corp.nlp.pipe_names
    assert 'lemmatizer' not in corp.nlp.pipe_names

    _check_copies(corp, copy(corp), same_nlp_instance=True)
    _check_copies(corp, deepcopy(corp), same_nlp_instance=False)

    corp = c.Corpus(textdata_en, language='en', load_features={'tok2vec', 'senter'})
    assert corp.has_sents
    assert corp.language_model == 'en_core_web_sm'
    _check_corpus_docs(corp, has_sents=True)
    assert 'senter' in corp.nlp.pipe_names
    assert 'tagger' not in corp.nlp.pipe_names
    assert 'parser' not in corp.nlp.pipe_names
    assert 'lemmatizer' not in corp.nlp.pipe_names

    _check_copies(corp, copy(corp), same_nlp_instance=True)
    _check_copies(corp, deepcopy(corp), same_nlp_instance=False)

    corp = c.Corpus(textdata_en, language='en', add_features={'ner'})
    assert corp.has_sents
    assert corp.language_model == 'en_core_web_sm'
    _check_corpus_docs(corp, has_sents=True)
    assert 'ner' in corp.nlp.pipe_names
    assert 'tagger' in corp.nlp.pipe_names
    assert 'parser' in corp.nlp.pipe_names
    assert 'lemmatizer' in corp.nlp.pipe_names

    _check_copies(corp, copy(corp), same_nlp_instance=True)
    _check_copies(corp, deepcopy(corp), same_nlp_instance=False)

    corp = c.Corpus(textdata_en, language='en', spacy_opts={'vocab': True})
    assert corp.has_sents
    _check_corpus_docs(corp, has_sents=True)
    assert corp._spacy_opts['vocab'] is True

    _check_copies(corp, copy(corp), same_nlp_instance=True)
    _check_copies(corp, deepcopy(corp), same_nlp_instance=False)

    custom_nlp = spacy.load("en_core_web_sm", exclude=['parser', 'ner'])
    corp = c.Corpus(textdata_en, spacy_instance=custom_nlp)
    assert corp.nlp is custom_nlp
    assert corp.language == 'en'
    assert corp.language_model == 'en_core_web_sm'
    assert 'parser' not in corp.nlp.pipe_names
    assert 'ner' not in corp.nlp.pipe_names

    _check_copies(corp, copy(corp), same_nlp_instance=True)
    _check_copies(corp, deepcopy(corp), same_nlp_instance=False)

    with pytest.raises(ValueError, match=r'all token attributes given in `spacy_token_attrs` must be valid SpaCy token '
                                         r'attribute names'):
        c.Corpus(textdata_en, language='en', spacy_token_attrs=('pos', 'lemma', 'ner'))

    with pytest.raises(ValueError, match=r'^the following SpaCy attributes are not available'):
        c.Corpus(textdata_en, language='en', spacy_token_attrs=('pos', 'lemma', 'ent_type'))

    corp = c.Corpus(textdata_en, language='en', spacy_token_attrs=('pos', 'lemma'))
    assert corp.has_sents
    assert corp.language_model == 'en_core_web_sm'
    _check_corpus_docs(corp, has_sents=True)
    assert corp.token_attrs == ('pos', 'lemma')

    _check_copies(corp, copy(corp), same_nlp_instance=True)
    _check_copies(corp, deepcopy(corp), same_nlp_instance=False)

    for n_workers in (1, 2):
        corp = c.Corpus(textdata_en, language='en', raw_preproc=c.strip_tags, max_workers=n_workers)
        assert corp.has_sents
        assert corp.language_model == 'en_core_web_sm'
        _check_corpus_docs(corp, has_sents=True)
        assert corp.raw_preproc == [c.strip_tags]
        assert c.doc_texts(corp, select='NewsArticles-3')['NewsArticles-3']\
            .startswith('BRICS wants to set up an alternative rating agency')

        _check_copies(corp, copy(corp), same_nlp_instance=True)
        _check_copies(corp, deepcopy(corp), same_nlp_instance=False)


@pytest.mark.skipif('en_core_web_md' not in spacy.util.get_installed_models(),
                    reason='language model "en_core_web_md" not installed')
def test_corpus_init_md_model_required():
    corp = c.Corpus(textdata_en, language='en', load_features={'vectors', 'tok2vec', 'tagger', 'morphologizer',
                                                               'parser', 'attribute_ruler', 'lemmatizer', 'ner'})
    assert corp.has_sents
    assert corp.language_model == 'en_core_web_md'
    _check_corpus_docs(corp, has_sents=True)
    assert 'ner' in corp.nlp.pipe_names

    _check_copies(corp, copy(corp), same_nlp_instance=True)
    _check_copies(corp, deepcopy(corp), same_nlp_instance=False)


@settings(deadline=None)
@given(docs=strategy_str_str_dict_printable(),
       punctuation=st.one_of(st.none(), st.lists(st.text(string.punctuation, min_size=1, max_size=1))),
       max_workers=st.one_of(st.none(),
                             st.integers(min_value=-2, max_value=2),
                             st.floats(allow_nan=False, allow_infinity=False)),
       workers_timeout=st.integers(0, 120))
def test_corpus_init_and_properties_hypothesis(spacy_instance_en_sm, docs, punctuation, max_workers, workers_timeout):
    args = dict(docs=docs, spacy_instance=spacy_instance_en_sm, punctuation=punctuation,
                max_workers=max_workers, workers_timeout=workers_timeout)

    if isinstance(max_workers, float) and not 0 <= max_workers <= 1:
        with pytest.raises(ValueError, match=re.escape(r'`max_workers` must be an integer, a float in [0, 1] or None')):
            c.Corpus(**args)
    else:
        corp = c.Corpus(**args)
        assert corp.nlp == spacy_instance_en_sm
        if punctuation is None:
            assert corp.punctuation == list(string.punctuation) + [' ', '\r', '\n', '\t']
        else:
            assert corp.punctuation == punctuation

        assert 0 < corp.max_workers <= max(2, multiprocessing.cpu_count())
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
        assert corp['_new_doc']['token'] == ['Foo', 'bar', '.']
        del corp['_new_doc']
        assert '_new_doc' not in corp
        assert len(corp) == len(docs)

        for (lbl, tok), lbl2, tok2 in zip(corp.items(), corp.keys(), corp.values()):
            assert lbl in docs.keys()
            assert isinstance(tok, c.Document)
            assert lbl == lbl2
            assert tok == tok2

        assert corp.uses_unigrams
        assert corp.token_attrs == corp._spacy_token_attrs
        assert corp.custom_token_attrs_defaults == {}
        assert corp.doc_attrs == ('label', 'has_sents')
        assert corp.doc_attrs_defaults == {'has_sents': False, 'label': ''}
        assert corp.ngrams == 1
        assert corp.ngrams_join_str == ' '
        assert corp.language == 'en'
        assert corp.language_model == 'en_core_web_sm'
        assert corp.doc_labels == list(docs.keys())
        assert corp.has_sents
        assert corp.n_docs == len(docs)

        if corp:
            lbl = random.choice(list(docs.keys()))
            assert isinstance(corp[lbl], c.Document)
            assert isinstance(corp.get(lbl), c.Document)
            assert corp[lbl]['token'] == corp.get(lbl)['token'] == c.doc_tokens(corp, select=lbl)
            assert corp.get('nonexistent', None) is None

            ind = random.randint(0, len(corp)-1)
            assert corp[ind] == corp[corp.doc_labels[ind]]
            assert corp[:ind] == [corp[lbl] for lbl in corp.doc_labels[:ind]]

            assert next(iter(corp)) == next(iter(corp.keys()))
            assert isinstance(next(iter(corp.values())), c.Document)


def test_corpus_init_otherlang_by_langcode():
    for langcode, docs in textdata_sm.items():
        if langcode in {'en', 'de'}: continue  # this is already tested

        if langcode not in installed_lang:
            with pytest.raises(RuntimeError):
                c.Corpus(docs, language=langcode)
        else:
            corp = c.Corpus(docs, language=langcode)

            assert set(corp.doc_labels) == set(docs.keys())
            assert corp.language == langcode
            assert corp.language_model.startswith(langcode)
            assert corp.max_workers == 1

            spdocs = c.spacydocs(corp)
            for d in spdocs.values():
                assert isinstance(d, Doc)


#%% test corpus properties and methods


def test_corpus_setitem_delitem(corpora_en_serial_and_parallel):
    for corp in corpora_en_serial_and_parallel:
        texts_before = c.doc_texts(corp)
        corp['added_doc1'] = ''
        corp['added_doc2'] = 'A new doc.'
        corp['added_doc3'] = corp.nlp('Another new doc.')

        with pytest.raises(ValueError, match=r'`doc` must be a string, a spaCy Doc object or a tmtoolkit Document '
                                             r'object'):
            corp['added_doc4'] = 1

        assert c.doc_texts(corp) == dict(**texts_before, **{
            'added_doc1': '',
            'added_doc2': 'A new doc.',
            'added_doc3': 'Another new doc.',
        })

        corp['added_doc1'] = 'Update!'

        assert c.doc_texts(corp) == dict(**texts_before, **{
            'added_doc1': 'Update!',
            'added_doc2': 'A new doc.',
            'added_doc3': 'Another new doc.',
        })

        for i in range(1, 5):
            if i < 4:
                del corp[f'added_doc{i}']
            else:
                with pytest.raises(KeyError):
                    del corp[f'added_doc{i}']

        assert c.doc_texts(corp) == texts_before


def test_corpus_iter_contains(corpora_en_serial_and_parallel):
    for corp in corpora_en_serial_and_parallel:
        doc_lbls_before = c.doc_labels(corp)
        assert sorted(corp) == doc_lbls_before

        c.remove_documents_by_label(corp, 'empty')
        assert set(corp) == set(doc_lbls_before) - {'empty'}
        assert 'empty' not in corp


def test_corpus_update(corpora_en_serial_and_parallel):
    for corp in corpora_en_serial_and_parallel:
        texts_before = c.doc_texts(corp)
        corp.update({})
        assert c.doc_texts(corp) == texts_before

        added1 = {'added_doc1': 'Added a new document.', 'added_doc2': 'Added another one.'}
        corp.update(added1)
        assert c.doc_texts(corp) == dict(**texts_before, **added1)

        added2 = {'added_doc2': corp.nlp('Updated as SpaCy document.'),
                  'added_doc3': corp.nlp('Added as SpaCy document.'),
                  'added_doc4': 'Added as raw text.'}
        corp.update(added2)
        assert c.doc_texts(corp) == dict(**texts_before, **{
            'added_doc1': 'Added a new document.',
            'added_doc2': 'Updated as SpaCy document.',
            'added_doc3': 'Added as SpaCy document.',
            'added_doc4': 'Added as raw text.'
        })

        with pytest.raises(ValueError, match=r'^one or more documents in `new_docs` are neither raw text documents'):
            corp.update({'error': 1})


#%% test corpus functions


@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       sentences=st.booleans(),
       only_non_empty=st.booleans(),
       tokens_as_hashes=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos',
                                                           list(TOKENMAT_ATTRS),
                                                           list(STD_TOKEN_ATTRS),
                                                           list(STD_TOKEN_ATTRS) + ['nonexistent']])),
       n_tokens=st.one_of(st.none(), st.integers(1, 10)),
       as_tables=st.booleans(),
       as_arrays=st.booleans())
def test_doc_tokens_hypothesis(corpora_en_serial_and_parallel_module, **args):
    for corp in list(corpora_en_serial_and_parallel_module):
        if args['select'] == 'nonexistent' or (args['select'] is not None and args['select'] != [] and len(corp) == 0):
            # selected document(s) don't exist
            with pytest.raises(KeyError):
                c.doc_tokens(corp, **args)
        elif args['select'] == 'empty' and args['only_non_empty']:
            # can't select empty document when `only_non_empty` is active
            with pytest.raises(ValueError, match=r'but only non-empty documents should be retrieved$'):
                c.doc_tokens(corp, **args)
        elif len(corp) > 0 and isinstance(args['with_attr'], list) and \
                any(a not in corp.token_attrs for a in args['with_attr']) \
                and args['select'] != []:
            # selected attribute(s) don't exist
            with pytest.raises(KeyError, match=r'^\'requested token attribute'):
                c.doc_tokens(corp, **args)
        else:
            res = c.doc_tokens(corp, **args)

            if isinstance(args['select'], str):
                # wrap in dict for rest of test
                res = {args['select']: res}
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
                    cols = [tuple(v.columns) for v in res.values() if len(v) > 0]
                    if len(cols) > 0:
                        assert len(set(cols)) == 1
                        attrs = next(iter(set(cols)))
                    else:
                        continue

                    for v in res.values():
                        if len(v) > 0:
                            assert np.issubdtype(v['token'].dtype,
                                                 np.uint64 if args['tokens_as_hashes'] else np.dtype('O'))
                            if args['sentences']:
                                assert np.issubdtype(v['sent'].dtype, 'int')
                                assert np.min(v['sent']) == 0

                    res_tokens = {}
                    for lbl, df in res.items():
                        res_tokens[lbl] = list(df['token'])
                else:  # no tables
                    if args['with_attr']:
                        assert all([isinstance(v, dict) for v in res.values()])

                        if args['sentences']:
                            assert all([isinstance(sents, list) for d in res.values()
                                        for sents in d.values()])

                        if args['as_arrays']:
                            if args['sentences']:
                                assert all([isinstance(s, np.ndarray) for d in res.values()
                                            for sents in d.values() for s in sents])
                            else:
                                assert all([isinstance(arr, np.ndarray) for d in res.values()
                                            for arr in d.values()])
                            res_tokens = {lbl: np.concatenate(d['token']).tolist() if args['sentences']
                                          else d['token'].tolist() for lbl, d in res.items()}
                        else:
                            res_tokens = {lbl: flatten_list(d['token']) if args['sentences'] else d['token']
                                          for lbl, d in res.items()}

                        cols = [tuple(v.keys()) for v in res.values()]
                        if len(cols) > 0:
                            assert len(set(cols)) == 1
                            attrs = next(iter(set(cols)))
                            assert set(attrs).isdisjoint({'has_sents', 'label'})
                        else:
                            attrs = None
                    else:   # no attributes
                        attrs = None
                        if args['sentences']:
                            assert all([isinstance(sents, list) for sents in res.values()])
                            res_tokens = {lbl: np.concatenate(sents).tolist() if args['as_arrays']
                                          else flatten_list(sents) for lbl, sents in res.items()}
                        else:
                            res_tokens = {lbl: d.tolist() for lbl, d in res.items()} if args['as_arrays'] else res

                for lbl, tok in res_tokens.items():
                    if args['n_tokens'] is None or args['n_tokens'] > len(tok):
                        assert len(tok) == len(corp[lbl])
                    else:
                        assert len(tok) == args['n_tokens']

                if args['tokens_as_hashes']:
                    assert all([isinstance(t, int) for tok in res_tokens.values() for t in tok])
                else:
                    assert all([isinstance(t, str) for tok in res_tokens.values() for t in tok])

                firstattrs = ['label'] if args['as_tables'] else []
                firstattrs.extend(['sent', 'token'] if args['sentences'] and args['as_tables'] else ['token'])

                if args['with_attr'] is True:
                    assert attrs == tuple(firstattrs + list(corp.spacy_token_attrs))
                elif args['with_attr'] is False:
                    if args['as_tables']:
                        assert attrs == tuple(firstattrs)
                    else:
                        assert attrs is None
                else:
                    if isinstance(args['with_attr'], str):
                        assert attrs == tuple(firstattrs + [args['with_attr']])
                    else:
                        assert attrs == tuple(firstattrs + args['with_attr'])


@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       as_table=st.sampled_from([False, True, 'length']))
def test_doc_lengths(corpora_en_serial_and_parallel_module, select, as_table):
    expected = {
        'empty': 0,
        'small1': 1,
        'small2': 7
    }
    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.doc_lengths(corp, select=select, as_table=as_table)
        else:
            res = c.doc_lengths(corp, select=select, as_table=as_table)

            if as_table is False:
                assert isinstance(res, dict)

                if select is None or len(corp) == 0:
                    assert set(res.keys()) == set(corp.keys())
                else:
                    assert set(res.keys()) == ({select} if isinstance(select, str) else set(select)) - {'nonexistent'}

                for lbl, n in res.items():
                    assert n >= 0
                    if lbl in expected:
                        assert n == expected[lbl]
                    else:
                        assert n >= len(textdata_en[lbl].split())
            else:
                assert isinstance(res, pd.DataFrame)

                if select is None or len(corp) == 0:
                    assert set(res['doc']) == set(corp.keys())
                else:
                    assert set(res['doc']) == ({select} if isinstance(select, str) else set(select)) - {'nonexistent'}


@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]))
def test_doc_token_lengths(corpora_en_serial_and_parallel_module, select):
    expected = {
        'empty': [],
        'small1': [3],
        'small2': [4, 2, 1, 5, 7, 8, 1]
    }

    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.doc_token_lengths(corp, select=select)
        else:
            res = c.doc_token_lengths(corp, select=select)

            assert isinstance(res, dict)

            if select is None or len(corp) == 0:
                assert set(res.keys()) == set(corp.keys())
            else:
                assert set(res.keys()) == ({select} if isinstance(select, str) else set(select)) - {'nonexistent'}

            for lbl, toklengths in res.items():
                assert isinstance(toklengths, list)
                assert all([n >= 0 for n in toklengths])
                if lbl in expected:
                    assert toklengths == expected[lbl]


@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       as_table=st.sampled_from([False, True, 'num_sents']))
def test_doc_num_sents(corpora_en_serial_and_parallel_module, select, as_table):
    expected = {
        'empty': 0,
        'small1': 1,
        'unicode1': 1,
        'unicode2': 1,
        'NewsArticles-2': 19,
    }

    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.doc_num_sents(corp, select=select, as_table=as_table)
        else:
            res = c.doc_num_sents(corp, select=select, as_table=as_table)

            if as_table is False:
                assert isinstance(res, dict)

                if select is None or len(corp) == 0:
                    assert set(res.keys()) == set(corp.keys())
                else:
                    assert set(res.keys()) == ({select} if isinstance(select, str) else set(select)) - {'nonexistent'}

                for lbl, n_sents in res.items():
                    assert isinstance(n_sents, int)
                    assert n_sents >= 0
                    if lbl in expected:
                        assert n_sents == expected[lbl]
            else:
                assert isinstance(res, pd.DataFrame)

                if select is None or len(corp) == 0:
                    assert set(res['doc']) == set(corp.keys())
                else:
                    assert set(res['doc']) == ({select} if isinstance(select, str) else set(select)) - {'nonexistent'}


@settings(deadline=None)
@given(apply_filter=st.booleans(),
       select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]))
def test_doc_sent_lengths(corpora_en_serial_and_parallel_module, apply_filter, select):
    for corp in corpora_en_serial_and_parallel_module:
        if apply_filter:
            corp = c.filter_clean_tokens(corp, inplace=False)

        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.doc_sent_lengths(corp, select=select)
        else:
            res = c.doc_sent_lengths(corp, select=select)

            assert isinstance(res, dict)

            if select is None or len(corp) == 0:
                assert set(res.keys()) == set(corp.keys())
            else:
                assert set(res.keys()) == ({select} if isinstance(select, str) else set(select)) - {'nonexistent'}

            num_sents = c.doc_num_sents(corp, select=select)
            num_tok = c.doc_lengths(corp, select=select)

            for lbl, s_lengths in res.items():
                if apply_filter:
                    assert all([l >= 0 for l in s_lengths])
                else:
                    assert all([l > 0 for l in s_lengths])
                assert len(s_lengths) == num_sents[lbl]
                assert sum(s_lengths) == num_tok[lbl]


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


@given(n=st.integers())
def test_doc_labels_sample(corpora_en_serial_and_parallel_module, n):
    for corp in corpora_en_serial_and_parallel_module:
        if 0 <= n <= len(corp):
            res = c.doc_labels_sample(corp, n=n)
            assert isinstance(res, set)
            assert len(res) == n
            assert res <= set(corp.keys())
        else:
            with pytest.raises(ValueError, match='Sample larger than population or is negative'):
                c.doc_labels_sample(corp, n=n)


@settings(deadline=None)
@given(collapse=st.sampled_from([None, ' ', '__']),
       select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       as_table=st.sampled_from([False, True, 'text']))
def test_doc_texts(corpora_en_serial_and_parallel_module, collapse, select, as_table):
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
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.doc_texts(corp, select=select, collapse=collapse, as_table=as_table)
        else:
            res = c.doc_texts(corp, select=select, collapse=collapse, as_table=as_table)

            if as_table is False:
                assert isinstance(res, dict)
                if select is None:
                    assert set(res.keys()) == set(corp.keys())
                else:
                    assert set(res.keys()) == ({select} if isinstance(select, str) else set(select))

                for lbl, txt in res.items():
                    assert isinstance(txt, str)
                    if collapse is None:
                        assert txt == textdata_en[lbl]
                    else:
                        if lbl in expected[collapse]:
                            assert txt == expected[collapse][lbl]
            else:
                assert isinstance(res, pd.DataFrame)
                if select is None:
                    assert set(res['doc']) == set(corp.keys())
                else:
                    assert set(res['doc']) == ({select} if isinstance(select, str) else set(select))



@settings(deadline=None)
@given(proportions=st.sampled_from([0, 1, 2]),
       select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       as_table=st.sampled_from([False, True, 'freq']))
def test_doc_frequencies(corpora_en_serial_and_parallel_module, proportions, select, as_table):
    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.doc_frequencies(corp, select=select, proportions=proportions, as_table=as_table)
        else:
            res = c.doc_frequencies(corp, select=select, proportions=proportions, as_table=as_table)

            if as_table is False:
                assert isinstance(res, dict)
                assert set(res.keys()) == c.vocabulary(corp, select=select, sort=False)

                if len(corp) > 0 and select not in ('empty', []):
                    if proportions == 1:
                        # proportions
                        assert all([0 < v <= 1 for v in res.values()])
                        if select is None:
                            assert np.isclose(res['the'], 5/9)
                    elif proportions == 2:
                        # log proportions
                        assert all([v <= 0 for v in res.values()])
                        assert all([0 < 10**v <= 1 for v in res.values()])
                        if select is None:
                            assert np.isclose(res['the'], math.log10(5/9))
                    else:
                        # counts
                        assert all([0 < v < len(corp) for v in res.values()])
                        assert any([v > 0 for v in res.values()])
                        if select is None:
                            assert res['the'] == 5
            else:
                assert isinstance(res, pd.DataFrame)
                assert set(res['token']) == c.vocabulary(corp, select=select, sort=False)


@pytest.mark.skipif('en_core_web_md' not in spacy.util.get_installed_models(),
                    reason='language model "en_core_web_md" not installed')
@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       omit_empty=st.booleans())
def test_doc_vectors(corpora_en_serial_and_parallel_also_w_vectors_module, select, omit_empty):
    for i_corp, corp in enumerate(corpora_en_serial_and_parallel_also_w_vectors_module):
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.doc_vectors(corp, omit_empty=omit_empty, select=select)
        else:
            if i_corp < 2:
                with pytest.raises(RuntimeError):
                    c.doc_vectors(corp, omit_empty=omit_empty, select=select)
            else:
                res = c.doc_vectors(corp, omit_empty=omit_empty, select=select)
                assert isinstance(res, dict)

                if select is None:
                    expected_lbls = set(corp.keys())
                else:
                    expected_lbls = ({select} if isinstance(select, str) else set(select)) - {'nonexistent'}

                if omit_empty:
                    assert set(res.keys()) == expected_lbls - {'empty'}
                else:
                    assert set(res.keys()) == expected_lbls

                for vec in res.values():
                    assert isinstance(vec, np.ndarray)
                    assert len(vec) > 0


@pytest.mark.skipif('en_core_web_md' not in spacy.util.get_installed_models(),
                    reason='language model "en_core_web_md" not installed')
@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       omit_oov=st.booleans())
def test_token_vectors(corpora_en_serial_and_parallel_also_w_vectors_module, select, omit_oov):
    for i_corp, corp in enumerate(corpora_en_serial_and_parallel_also_w_vectors_module):
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.token_vectors(corp, omit_oov=omit_oov, select=select)
        else:
            if i_corp < 2:
                with pytest.raises(RuntimeError):
                    c.token_vectors(corp, omit_oov=omit_oov, select=select)
            else:
                res = c.token_vectors(corp, omit_oov=omit_oov, select=select)

                assert isinstance(res, dict)

                if select is None:
                    assert set(res.keys()) == set(corp.keys())
                else:
                    assert set(res.keys()) == ({select} if isinstance(select, str) else set(select)) - {'nonexistent'}

                doc_length = c.doc_lengths(corp)
                spdocs = c.spacydocs(corp)

                for lbl, mat in res.items():
                    assert isinstance(mat, np.ndarray)

                    if omit_oov:
                        assert len(mat) == sum([not t.is_oov for t in spdocs[lbl]])
                    else:
                        assert len(mat) == doc_length[lbl]

                    if len(mat) > 0:
                        assert mat.ndim == 2


@pytest.mark.skipif('en_core_web_md' not in spacy.util.get_installed_models(),
                    reason='language model "en_core_web_md" not installed')
@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       collapse=st.sampled_from([None, ' ']))
def test_spacydocs(corpora_en_serial_and_parallel_also_w_vectors_module, select, collapse):
    for i_corp, corp in enumerate(corpora_en_serial_and_parallel_also_w_vectors_module):
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.spacydocs(corp, select=select, collapse=collapse)
        else:
            res = c.spacydocs(corp, select=select, collapse=collapse)

            assert isinstance(res, dict)

            if select is None:
                assert set(res.keys()) == set(corp.keys())
            else:
                assert set(res.keys()) == ({select} if isinstance(select, str) else set(select)) - {'nonexistent'}

            if collapse is None:
                texts = c.doc_texts(corp, select=select)
            else:
                texts = None

            for lbl, d in res.items():
                assert isinstance(d, Doc)
                if collapse is None:
                    assert d.text_with_ws == texts[lbl]


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       tokens_as_hashes=st.booleans(),
       force_unigrams=st.booleans(),
       sort=st.booleans(),
       convert_uint64hashes=st.booleans())
def test_vocabulary_hypothesis(corpora_en_serial_and_parallel_module, select, tokens_as_hashes, force_unigrams, sort,
                               convert_uint64hashes):
    kwargs = dict(select=select, tokens_as_hashes=tokens_as_hashes, force_unigrams=force_unigrams, sort=sort,
                  convert_uint64hashes=convert_uint64hashes)

    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.vocabulary(corp, **kwargs)
        else:
            res = c.vocabulary(corp, **kwargs)

            if sort:
                assert isinstance(res, list)
                assert sorted(res) == res
            else:
                assert isinstance(res, set)

            if len(corp) > 0:
                if select in ('empty', []):
                    assert len(res) == 0
                else:
                    assert len(res) > 0

                if select == 'small2' and not tokens_as_hashes:
                    assert set(res) == {'This', 'is', 'a', 'small', 'example', 'document', '.'}

                if select != 'empty':
                    corp_flat = c.corpus_tokens_flattened(corp, select=select, tokens_as_hashes=tokens_as_hashes)
                    assert all(t in corp_flat for t in res)

                if not convert_uint64hashes and tokens_as_hashes and select is None:
                    assert all([np.issubdtype(t.dtype, 'uint64') for t in res])
                else:
                    if tokens_as_hashes:
                        expect_type = int
                    else:
                        expect_type = str

                    assert all([isinstance(t, expect_type) for t in res])


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       proportions=st.sampled_from([0, 1, 2]),
       tokens_as_hashes=st.booleans(),
       force_unigrams=st.booleans(),
       convert_uint64hashes=st.booleans(),
       as_table=st.sampled_from([False, True, 'freq']))
def test_vocabulary_counts(corpora_en_serial_and_parallel_module, select, proportions, tokens_as_hashes, force_unigrams,
                           convert_uint64hashes, as_table):
    kwargs = dict(select=select, proportions=proportions, tokens_as_hashes=tokens_as_hashes,
                  force_unigrams=force_unigrams, convert_uint64hashes=convert_uint64hashes, as_table=as_table)

    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.vocabulary_counts(corp, **kwargs)
        else:
            res = c.vocabulary_counts(corp, **kwargs)
            vocab = c.vocabulary(corp, select=select, tokens_as_hashes=tokens_as_hashes,
                                 force_unigrams=force_unigrams, sort=False)

            if as_table is False:
                assert isinstance(res, dict)

                if len(corp) > 0:
                    if select in ('empty', []):
                        assert len(res) == 0
                    else:
                        assert len(res) > 0

                    if not convert_uint64hashes and tokens_as_hashes:
                        assert all([np.issubdtype(t.dtype, 'uint64') for t in res.keys()])
                    else:
                        if tokens_as_hashes:
                            expect_type = int
                        else:
                            expect_type = str

                        assert all([isinstance(t, expect_type) for t in res.keys()])

                    if select != 'empty':
                        corp_flat = c.corpus_tokens_flattened(corp, select=select, tokens_as_hashes=tokens_as_hashes)
                        assert all(t in corp_flat for t in res.keys())

                    if proportions == 0:
                        assert all([n > 0 for n in res.values()])
                    elif proportions == 1:
                        assert all([(0 < n <= 1) for n in res.values()])
                    else:   # proportions == 2 (log10)
                        assert all([(n <= 0) and (0 < 10**n <= 1) for n in res.values()])

                    assert vocab == set(res.keys())
            else:
                assert isinstance(res, pd.DataFrame)
                assert set(res['token']) == vocab


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       force_unigrams=st.booleans())
def test_vocabulary_size(corpora_en_serial_and_parallel_module, select, force_unigrams):
    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.vocabulary_size(corp, select=select, force_unigrams=force_unigrams)
        else:
            res = c.vocabulary_size(corp, select=select, force_unigrams=force_unigrams)

            assert isinstance(res, int)
            if len(corp) > 0 and select not in ('empty', []):
                assert res > 0
                if select != 'empty':
                    corp_flat = c.corpus_tokens_flattened(corp, select=select, force_unigrams=force_unigrams)
                    assert res <= len(corp_flat)
            else:
                assert res == 0


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       sentences=st.booleans(),
       tokens_as_hashes=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos',
                                                           list(TOKENMAT_ATTRS),
                                                           list(STD_TOKEN_ATTRS),
                                                           list(STD_TOKEN_ATTRS) + ['nonexistent']])))
def test_tokens_table_hypothesis(corpora_en_serial_and_parallel_module, **args):
    for corp in corpora_en_serial_and_parallel_module:
        if args['select'] == 'nonexistent' or (args['select'] is not None and args['select'] != [] and len(corp) == 0):
            with pytest.raises(KeyError):
                c.tokens_table(corp, **args)
        elif len(corp) > 0 and isinstance(args['with_attr'], list) and \
                any(a not in corp.token_attrs for a in args['with_attr']) \
                and args['select'] not in ([], 'empty'):
            # selected attribute(s) don't exist
            with pytest.raises(KeyError):
                c.tokens_table(corp, **args)
        elif args['select'] == 'empty':
            # can't select empty document when `only_non_empty` is active
            with pytest.raises(ValueError, match=r'but only non-empty documents should be retrieved$'):
                c.tokens_table(corp, **args)
        else:
            res = c.tokens_table(corp, **args)
            assert isinstance(res, pd.DataFrame)

            cols = res.columns.tolist()
            if args['sentences']:
                assert cols[:3] == ['doc', 'sent', 'position']
                assert np.all(res.sent >= 0)
                assert np.all(res.sent <= np.max(res.position))
            else:
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
                    assert set(corp.token_attrs) <= set(cols)
                elif isinstance(args['with_attr'], str):
                    assert args['with_attr'] in cols
                elif isinstance(args['with_attr'], list):
                    assert set(args['with_attr']) <= set(cols)


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       sentences=st.booleans(),
       tokens_as_hashes=st.booleans(),
       as_array=st.booleans())
def test_corpus_tokens_flattened(corpora_en_serial_and_parallel_module, select, sentences, tokens_as_hashes, as_array):
    def _check_tokens(tok, vocab):
        if as_array:
            assert isinstance(tok, np.ndarray)
            expected_tok_type = np.uint64 if tokens_as_hashes else str
            assert all([isinstance(t, expected_tok_type)for t in tok])
        else:
            assert isinstance(tok, list)
            expected_tok_type = int if tokens_as_hashes else str
            assert all([isinstance(t, expected_tok_type) for t in tok])

        assert set(tok) <= vocab

    kwargs = dict(select=select, sentences=sentences, tokens_as_hashes=tokens_as_hashes, as_array=as_array)
    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.corpus_tokens_flattened(corp, **kwargs)
        elif select == 'empty' and len(corp) > 0:
            with pytest.raises(ValueError, match=r'but only non-empty documents should be retrieved$'):
                c.corpus_tokens_flattened(corp, **kwargs)
        else:
            res = c.corpus_tokens_flattened(corp, **kwargs)
            vocab = c.vocabulary(corp, tokens_as_hashes=tokens_as_hashes, sort=False)

            if sentences:
                assert isinstance(res, list)
                assert len(res) >= 1   # always at least contains an empty sentence `[[]]`

                n_tok = 0
                for sent in res:
                    if len(corp) > 0 and select not in ('empty', []):
                        assert len(sent) > 0
                        _check_tokens(sent, vocab)
                    else:
                        assert len(sent) == 0
                    n_tok += len(sent)
            else:
                _check_tokens(res, vocab)
                n_tok = len(res)

            doc_len = c.doc_lengths(corp)
            if isinstance(select, str):
                select = [select]

            if select is not None:
                doc_len = [n for lbl, n in doc_len.items() if lbl in select]
            else:
                doc_len = doc_len.values()

            assert n_tok == sum(doc_len)


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]))
def test_corpus_num_tokens(corpora_en_serial_and_parallel_module, select):
    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.corpus_num_tokens(corp, select=select)
        else:
            res = c.corpus_num_tokens(corp, select=select)
            assert res == sum(c.doc_lengths(corp, select=select).values())
            if len(corp) == 0 or select in ('empty', []):
                assert res == 0


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]))
def test_corpus_num_chars(corpora_en_serial_and_parallel_module, select):
    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.corpus_num_chars(corp, select=select)
        else:
            res = c.corpus_num_chars(corp, select=select)
            if len(corp) == 0 or select in ('empty', []):
                assert res == 0
            else:
                assert res > 0


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]))
def test_corpus_unique_chars(corpora_en_serial_and_parallel_module, select):
    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.corpus_unique_chars(corp, select=select)
        else:
            res = c.corpus_unique_chars(corp, select=select)
            if len(corp) == 0 or select in ('empty', []):
                assert res == set()
            else:
                assert isinstance(res, set)
                assert all([isinstance(c, str) and len(c) == 1 for c in res])

                if select == 'small2':
                    assert res == {'.', 'T', 'a', 'c', 'd', 'e', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 's', 't', 'u', 'x'}



@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       threshold=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
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
            with pytest.raises(ValueError, match='`glue` cannot be None if `as_table` is True'):
                c.corpus_collocations(corp, **args)
        elif args['select'] == 'nonexistent' or (args['select'] not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.corpus_collocations(corp, **args)
        elif args['select'] == 'empty' and len(corp) > 0:
            with pytest.raises(ValueError, match=r'but only non-empty documents should be retrieved$'):
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


@settings(deadline=None)
@given(max_documents=st.one_of(st.none(), st.integers()),
       max_tokens_string_length=st.one_of(st.none(), st.integers()))
def test_corpus_summary(corpora_en_serial_and_parallel_module, max_documents, max_tokens_string_length):
    for corp in corpora_en_serial_and_parallel_module:
        res = c.corpus_summary(corp, max_documents=max_documents, max_tokens_string_length=max_tokens_string_length)
        assert isinstance(res, str)
        assert str(len(corp)) in res
        assert LANGUAGE_LABELS[corp.language].capitalize() in res
        assert str(c.corpus_num_tokens(corp)) in res
        assert str(c.vocabulary_size(corp)) in res

        lines = res.split('\n')
        if max_documents is None:
            n_docs_printed = corp.print_summary_default_max_documents
        elif max_documents >= 0:
            n_docs_printed = max_documents
        else:
            n_docs_printed = len(corp)
        assert len(lines) == 2 + min(len(corp), n_docs_printed + bool(len(corp) > n_docs_printed))

        if corp.ngrams > 1:
            assert f'{corp.ngrams}-grams' in lines[-1]


def test_print_summary(capsys, corpora_en_serial_and_parallel_module):
    for corp in corpora_en_serial_and_parallel_module:
        c.print_summary(corp)
        assert capsys.readouterr().out == c.corpus_summary(corp) + '\n'


@settings(deadline=None)
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       as_table=st.booleans(),
       dtype=st.sampled_from([None, 'uint16', 'float64']),
       return_doc_labels=st.booleans(),
       return_vocab=st.booleans())
def test_dtm(corpora_en_serial_and_parallel_module, select, as_table, dtype, return_doc_labels, return_vocab):
    kwargs = dict(select=select, as_table=as_table, dtype=dtype,
                  return_doc_labels=return_doc_labels, return_vocab=return_vocab)

    for corp in corpora_en_serial_and_parallel_module:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.dtm(corp, **kwargs)
        else:
            res = c.dtm(corp, **kwargs)

            expected_vocab = c.vocabulary(corp, select=select, sort=True)
            if select is None:
                expected_labels = c.doc_labels(corp, sort=True)
            elif isinstance(select, str):
                expected_labels = [select]
            else:
                expected_labels = sorted(select)

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
            assert dtm.shape[0] == len(expected_labels)
            assert dtm.shape[1] == len(expected_vocab)

            if as_table:
                assert isinstance(dtm, pd.DataFrame)
                assert dtm.index.tolist() == expected_labels
                assert dtm.columns.tolist() == expected_vocab

                if len(corp) > 0 and select is None:
                    assert np.sum(dtm.iloc[expected_labels.index('empty'), :]) == 0
                    assert np.sum(dtm.iloc[:, expected_vocab.index('the')]) > 1
                    assert dtm.iloc[expected_labels.index('small1'), expected_vocab.index('the')] == 1
            else:
                assert isinstance(dtm, csr_matrix)
                assert dtm.dtype == np.dtype(dtype or 'int32')

                if len(corp) > 0 and select is None:
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

            corp_tokens = c.doc_tokens(corp)

            for lbl, ng in res.items():
                dtok = corp_tokens[lbl]
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
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       search_term_exists=st.booleans(),
       context_size=st.one_of(st.integers(-1, 3), st.tuples(st.integers(-1, 2), st.integers(-1, 2))),
       by_attr=st.sampled_from([None, 'nonexistent', 'pos', 'lemma']),
       inverse=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos', 'lemma', ['pos', 'lemma']])),
       as_tables=st.booleans(),
       only_non_empty=st.booleans(),
       glue=st.one_of(st.none(), st.text(string.printable, max_size=1)),
       highlight_keyword=st.sampled_from([None, '*', '^']))
def test_kwic_hypothesis(corpora_en_serial_and_parallel_module, **args):
    search_term_exists = args.pop('search_term_exists')
    matchattr = args['by_attr'] or 'token'

    for corp in corpora_en_serial_and_parallel_module:
        csize = args['context_size']
        if (isinstance(csize, int) and csize <= 0) or \
                (isinstance(csize, tuple) and (any(x < 0 for x in csize) or all(x == 0 for x in csize))):
            with pytest.raises(ValueError):
                c.kwic(corp, 'doesntmatter', **args)
        elif args['glue'] is not None and args['with_attr']:
            with pytest.raises(ValueError):
                c.kwic(corp, 'doesntmatter', **args)
        elif args['select'] == 'nonexistent' or (args['select'] not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.kwic(corp, 'doesntmatter', **args)
        elif args['by_attr'] == 'nonexistent' and len(corp) > 0 and args['select'] != []:
            with pytest.raises(KeyError):
                c.kwic(corp, 'doesntmatter', **args)
        else:
            if matchattr == 'token':
                vocab = list(c.vocabulary(corp, select=args['select']))
            else:
                if matchattr != 'nonexistent':
                    tok_attrs = [attrs[matchattr]
                                 for attrs in c.doc_tokens(corp,
                                                           select={args['select']} if isinstance(args['select'], str)
                                                                   else args['select'],
                                                           with_attr=matchattr).values()]
                    vocab = list(set(flatten_list(tok_attrs)))
                else:
                    vocab = []

            if search_term_exists and len(vocab) > 0:
                s = random.choice(vocab)
            else:
                s = 'thisdoesnotexist'

            res = c.kwic(corp, s, **args)
            assert isinstance(res, dict)

            if args['only_non_empty']:
                assert all([len(dkwic) > 0 for dkwic in res.values()])
            else:
                if args['select'] is None:
                    assert set(res.keys()) == set(corp.keys())
                else:
                    assert set(res.keys()) == ({args['select']} if isinstance(args['select'], str)
                                               else set(args['select'])) - {'nonexistent'}

            res_windows = {}
            if args['as_tables']:
                for lbl, dkwic in res.items():
                    assert isinstance(dkwic, pd.DataFrame)

                    if len(dkwic) > 0:
                        if args['glue'] is None:
                            expected_cols = ['doc', 'context', 'position', matchattr]
                            if args['with_attr'] is True:
                                expected_cols.extend([a for a in corp.token_attrs if a != args['by_attr']])
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
                                expected_keys.update(corp.token_attrs)
                            elif isinstance(args['with_attr'], list):
                                expected_keys.update(args['with_attr'])
                            elif isinstance(args['with_attr'], str):
                                expected_keys.add(args['with_attr'])
                            assert set(ctx.keys()) == expected_keys
                    res_windows = {lbl: [ctx[matchattr] for ctx in dkwic] for lbl, dkwic in res.items()}
                else:
                    res_windows = res

            if s in vocab:
                for lbl, win in res_windows.items():
                    for w in win:
                        if not args['inverse']:
                            if args['highlight_keyword'] is not None:
                                assert (args['highlight_keyword'] + s + args['highlight_keyword']) in w
                            else:
                                assert s in w

                            if args['glue'] is not None:
                                # `w` is string and should contain the "glue" string at least once
                                # or less if the document is empty
                                assert isinstance(w, str)
                                assert w.count(args['glue']) >= min(1, len(corp[lbl]))
                            else:
                                # `w` is a list of tokens around the search term
                                assert isinstance(w, list)
                                # the length `w` is "context size left + context size right + 1" (b/c of search term)
                                if isinstance(csize, int):  # symmetric context size
                                    assert 1 <= len(w) <= csize * 2 + 1
                                else:                       # possibly asymm. context size
                                    assert 1 <= len(w) <= sum(csize) + 1
            else:
                if args['only_non_empty']:
                    if args['inverse']:
                        if args['select'] is None:
                            n_docs = len(corp) - 1   # -1 because of empty doc.
                        elif isinstance(args['select'], str):
                            n_docs = 0 if args['select'] == 'empty' else 1
                        else:
                            n_docs = len(set(args['select']) - {'empty'})
                        assert len(res_windows) == max(n_docs, 0)
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
@given(select=st.sampled_from([None, 'empty', 'small2', 'nonexistent', ['small1', 'small2'], []]),
       search_term_exists=st.booleans(),
       context_size=st.one_of(st.integers(-1, 3), st.tuples(st.integers(-1, 2), st.integers(-1, 2))),
       by_attr=st.sampled_from([None, 'nonexistent', 'pos', 'lemma']),
       inverse=st.booleans(),
       with_attr=st.one_of(st.booleans(), st.sampled_from(['pos', 'lemma', ['pos', 'lemma']])),
       glue=st.one_of(st.none(), st.text(string.printable, max_size=1)),
       highlight_keyword=st.sampled_from([None, '*', '^']))
def test_kwic_table_hypothesis(corpora_en_serial_and_parallel_module, **args):
    search_term_exists = args.pop('search_term_exists')
    matchattr = args['by_attr'] or 'token'

    for corp in corpora_en_serial_and_parallel_module:
        csize = args['context_size']
        if (isinstance(csize, int) and csize <= 0) or \
                (isinstance(csize, tuple) and (any(x < 0 for x in csize) or all(x == 0 for x in csize))):
            with pytest.raises(ValueError):
                c.kwic_table(corp, 'doesntmatter', **args)
        elif args['glue'] is not None and args['with_attr']:
            with pytest.raises(ValueError):
                c.kwic_table(corp, 'doesntmatter', **args)
        elif args['select'] == 'nonexistent' or (args['select'] not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.kwic_table(corp, 'doesntmatter', **args)
        elif args['by_attr'] == 'nonexistent' and len(corp) > 0 and args['select'] != []:
            with pytest.raises(KeyError):
                c.kwic_table(corp, 'doesntmatter', **args)
        else:
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

            res = c.kwic_table(corp, s, **args)
            assert isinstance(res, pd.DataFrame)
            if args['glue'] is None:
                expected_cols = ['doc', 'context', 'position', matchattr]
                if args['with_attr'] is True:
                    expected_cols.extend([a for a in corp.token_attrs if a != args['by_attr']])
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
                        # disabled since this is not always the case: the keyword is in a very small document or at the
                        # start or end of a sentence, there may not be the "glue" string in the context
                        # if len(corp[lbl]) > 1:
                        #     assert all([args['glue'] in x for x in dkwic[matchattr]])

                        if not args['inverse']:
                            assert all([s in x for x in dkwic[matchattr]])
                            if args['highlight_keyword'] and args['highlight_keyword'] != args['glue']:
                                assert all([x.count(args['highlight_keyword']) == 2 for x in dkwic[matchattr]])


def test_save_load_corpus(corpora_en_serial_and_parallel_module):
    for corp in corpora_en_serial_and_parallel_module:
        with tempfile.TemporaryFile(suffix='.pickle') as ftemp:
            c.save_corpus_to_picklefile(corp, ftemp)
            ftemp.seek(0)
            unpickled_corp = c.load_corpus_from_picklefile(ftemp)

            _check_copies(corp, unpickled_corp, same_nlp_instance=False)


@settings(deadline=None)
@given(with_attr=st.one_of(st.booleans(), st.sampled_from([
           ['whitespace', 'token', 'pos', 'lemma'],
           ['whitespace', 'token', 'pos'],
           ['whitespace', 'pos'],
           ['token', 'pos'],
           [],
       ])),
       with_orig_corpus_opt=st.booleans(),
       sentences=st.booleans(),
       pass_doc_attr_names=st.booleans(),
       pass_token_attr_names=st.booleans())
def test_load_corpus_from_tokens_hypothesis(corpora_en_serial_and_parallel_module, with_attr, with_orig_corpus_opt,
                                            sentences, pass_doc_attr_names, pass_token_attr_names):
    for corp in corpora_en_serial_and_parallel_module:
        emptycorp = len(corp) == 0

        if sentences:
            sent_start_per_doc = {lbl: d['sent_start'] for lbl, d in corp.items()}
        else:
            sent_start_per_doc = None

        if len(corp) > 0:
            doc_attrs = {'empty': 'yes', 'small1': 'yes', 'small2': 'yes'}
        else:
            doc_attrs = {}
        c.set_document_attr(corp, 'docattr_test', doc_attrs, default='no')
        c.set_token_attr(corp, 'tokenattr_test', {'the': True}, default=False)
        tokens = c.doc_tokens(corp, sentences=sentences, with_attr=with_attr)

        kwargs = {'sentences': sentences}
        if with_orig_corpus_opt:
            kwargs['spacy_instance'] = corp.nlp
            kwargs['max_workers'] = corp.max_workers
        else:
            kwargs['language'] = corp.language

        if pass_doc_attr_names:
            kwargs['doc_attr'] = {'docattr_test': 'no'}
        if pass_token_attr_names:
            kwargs['token_attr'] = {'tokenattr_test': False}

        if not emptycorp and (with_attr is False or with_attr == []):
            with pytest.raises(ValueError, match=r'`tokens_w_attr` must be given as dict with token attributes'):
                c.load_corpus_from_tokens(tokens, **kwargs)
        elif not emptycorp and (with_attr is True or 'whitespace' not in with_attr):
            with pytest.raises(ValueError, match=r'^at least the following base token attributes must be given: '):
                c.load_corpus_from_tokens(tokens, **kwargs)
        else:
            corp2 = c.load_corpus_from_tokens(tokens, **kwargs)
            assert len(corp) == len(corp2)
            assert corp2.language == 'en'

            # check if tokens are the same
            assert c.doc_tokens(corp) == c.doc_tokens(corp2)

            # check sentences
            if sentences:
                assert {lbl: d['sent_start'] for lbl, d in corp2.items()} == sent_start_per_doc
                assert c.doc_tokens(corp, sentences=True) == c.doc_tokens(corp2, sentences=True)
            else:
                assert all([not d.has_sents for d in corp2.values()])

            # check if token dataframes are the same
            corp_table = c.tokens_table(corp, sentences=sentences, with_attr=with_attr)
            corp2_table = c.tokens_table(corp2, sentences=sentences, with_attr=with_attr)
            if len(corp) == 0 and (not pass_token_attr_names or not pass_doc_attr_names):
                # in this case the columns of the tables are different because for corp2 the custom  attributes could
                # not be set
                assert len(corp_table) == len(corp2_table)
            else:
                assert _dataframes_equal(corp_table, corp2_table, require_same_index=False)

            if with_orig_corpus_opt:
                assert corp.nlp is corp2.nlp
                assert corp.max_workers == corp2.max_workers
            else:
                assert corp.nlp is not corp2.nlp


@pytest.mark.parametrize('with_orig_corpus_opt, sentences, with_attr', [
    (False, False, True),
    (False, True, True),
    (True, False, True),
    (True, True, True),
    (False, False, False),
    (False, False, ['whitespace', 'pos', 'lemma']),
    (False, True, ['whitespace', 'pos', 'lemma']),
    (True, False, ['whitespace', 'pos', 'lemma']),
    (True, True, ['whitespace', 'pos', 'lemma']),
])
def test_load_corpus_from_tokens_table(corpora_en_serial_and_parallel, with_orig_corpus_opt, sentences, with_attr):
    for corp in corpora_en_serial_and_parallel:
        if len(corp) > 0:
            doc_attrs = {'empty': 'yes', 'small1': 'yes', 'small2': 'yes'}
        else:
            doc_attrs = {}
        c.set_document_attr(corp, 'docattr_test', doc_attrs, default='no')
        c.set_token_attr(corp, 'tokenattr_test', {'the': True}, default=False)

        tokenstab = c.tokens_table(corp, sentences=sentences, with_attr=with_attr)

        kwargs = {}
        if with_orig_corpus_opt:
            kwargs['spacy_instance'] = corp.nlp
            kwargs['max_workers'] = corp.max_workers
        else:
            kwargs['language'] = corp.language

        if isinstance(with_attr, bool) or 'whitespace' not in with_attr:
            with pytest.raises(ValueError, match=r'^`tokens` dataframe must at least contain the following columns: '):
                c.load_corpus_from_tokens_table(tokenstab, **kwargs)
        else:
            corp2 = c.load_corpus_from_tokens_table(tokenstab, **kwargs)
            if len(corp) > 0:
                assert len(corp) - 1 == len(corp2)   # empty doc. not in result
            assert corp2.language == 'en'

            # check if tokens are the same
            assert c.doc_tokens(corp, sentences=sentences, only_non_empty=True) == \
                   c.doc_tokens(corp2, sentences=sentences)
            # check if token dataframes are the same
            assert _dataframes_equal(c.tokens_table(corp, sentences=sentences, with_attr=with_attr), tokenstab)

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
    (5, [os.path.join(DATADIR_GUTENB, 'kafka_verwandlung.txt'),
         os.path.join(DATADIR_WERTHER, 'goethe_werther1.txt'),
         os.path.join(DATADIR_WERTHER, 'goethe_werther2.txt')],
     '{basename}', True),
])
def test_corpus_add_files_and_from_files(corpora_en_serial_and_parallel, testtype, files, doc_label_fmt, inplace):
    # make it a bit quicker by reading only 100 chars
    common_kwargs = dict(doc_label_fmt=doc_label_fmt, read_size=100, force_unix_linebreaks=False)

    if testtype == 5:
        common_kwargs['sample'] = 2

    ### test Corpus.from_files ###
    kwargs = dict(language='de', max_workers=1, **common_kwargs)               # Corpus constructor args
    if testtype == 4:
        with pytest.raises(ValueError, match='^duplicate document label'):
            c.Corpus.from_files(files, **kwargs)
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
        elif testtype == 5:
            assert len(doc_lbls) == 2
        else:
            raise ValueError(f'unknown testtype {testtype}')

    ### test corpus_add_files ###
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}
    kwargs = dict(inplace=inplace, **common_kwargs)
    for corp in corpora_en_serial_and_parallel:
        n_docs_before = len(corp)

        if testtype == 4:
            with pytest.raises(ValueError, match=r'duplicate document label'):
                c.corpus_add_files(corp, files, **kwargs)
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
            elif testtype == 5:
                assert len(res) == n_docs_before + 2
            else:
                raise ValueError(f'unknown testtype {testtype}')


@pytest.mark.parametrize('testtype, folder, valid_extensions, inplace', [
    (1, 'nonexistent', ('txt',), True),
    (2, DATADIR_GUTENB, ('txt',), True),
    (2, DATADIR_GUTENB, ('txt',), False),
    (2, DATADIR_GUTENB, ('txt', 'foo',), True),
    (3, DATADIR_GUTENB, ('foo',), True),
    (4, DATADIR_WERTHER, ('txt',), True),
    (5, DATADIR_GUTENB, ('txt',), True),
])
def test_corpus_add_folder_and_from_folder(corpora_en_serial_and_parallel, testtype, folder, valid_extensions, inplace):
    if testtype == 1:
        expected_doclbls = None
    elif testtype in {2, 3, 5}:
        expected_doclbls = {'kafka_verwandlung', 'werther-goethe_werther1', 'werther-goethe_werther2'}
    elif testtype == 4:
        expected_doclbls = {'goethe_werther1', 'goethe_werther2'}
    else:
        raise ValueError(f'unknown testtype {testtype}')

    # make it a bit quicker by reading only 100 chars
    common_kwargs = dict(valid_extensions=valid_extensions, read_size=100, force_unix_linebreaks=False)

    if testtype == 5:
        common_kwargs['sample'] = 2

    ### test Corpus.from_folder ###
    kwargs = dict(language='de', max_workers=1, **common_kwargs)               # Corpus constructor args

    if testtype == 1:
        with pytest.raises(IOError, match=r'^path does not exist'):
            c.Corpus.from_folder(folder, **kwargs)
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
        else:  # testtype == 5
            assert len(doclbls) == 2
            assert all(lbl in expected_doclbls for lbl in doclbls)

    ### test corpus_add_folder ###
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}
    kwargs = dict(inplace=inplace, **common_kwargs)
    for corp in corpora_en_serial_and_parallel:
        n_docs_before = len(corp)

        if testtype == 1:
            with pytest.raises(IOError, match=r'^path does not exist'):
                c.corpus_add_folder(corp, folder, **kwargs)
        else:
            res = c.corpus_add_folder(corp, folder, **kwargs)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            doclbls = set(c.doc_labels(res))

            if testtype == 2:
                if n_docs_before == 0:
                    assert expected_doclbls == doclbls
                else:
                    assert expected_doclbls < doclbls
            elif testtype == 3:
                assert expected_doclbls & doclbls == set()
            elif testtype == 4:
                if n_docs_before == 0:
                    assert expected_doclbls == doclbls
                else:
                    assert expected_doclbls < doclbls
            else:   # testtype == 5
                assert len(doclbls) == n_docs_before + 2


@pytest.mark.parametrize('testtype, files, id_column, text_column, prepend_columns, inplace', [
    (1, 'invalid_ext.foo', 0, 1, None, True),
    (2, os.path.join(DATADIR, '100NewsArticles.csv'), 'article_id', 'text', None, True),
    (2, os.path.join(DATADIR, '100NewsArticles.csv'), 'article_id', 'text', None, False),
    (2, os.path.join(DATADIR, '100NewsArticles.xlsx'), 'article_id', 'text', None, True),
    (3, os.path.join(DATADIR, '100NewsArticles.xlsx'), 'article_id', 'text', ['title', 'subtitle'], True),
    (4, [os.path.join(DATADIR, '100NewsArticles.csv'), os.path.join(DATADIR, '3ExampleDocs.xlsx')],
     'article_id', 'text', None, True),
    (5, [os.path.join(DATADIR, '100NewsArticles.csv'), os.path.join(DATADIR, '3ExampleDocs.xlsx')],
     'article_id', 'text', None, True),
])
def test_corpus_add_tabular_and_from_tabular(corpora_en_serial_and_parallel, testtype, files, id_column, text_column,
                                             prepend_columns, inplace):
    if testtype == 1:
        expected_doclbls = None
    elif testtype in {2, 3}:
        expected_doclbls = set(f'100NewsArticles-{i}' for i in range(1, 101))
    elif testtype in {4, 5}:
        expected_doclbls = set(f'100NewsArticles-{i}' for i in range(1, 101))
        expected_doclbls.update(f'3ExampleDocs-example{i}' for i in range(1, 4))
    else:
        raise ValueError(f'unknown testtype {testtype}')

    common_kwargs = dict(files=files, id_column=id_column, text_column=text_column, prepend_columns=prepend_columns)

    if testtype == 5:
        common_kwargs['sample'] = 2

    ### test Corpus.from_tabular ###
    kwargs = dict(language='de', max_workers=1, **common_kwargs)               # Corpus constructor args

    if testtype == 1:
        with pytest.raises(ValueError, match='only file extensions ".csv", ".xls" and ".xlsx" are supported'):
            c.Corpus.from_tabular(**kwargs)
    else:
        corp = c.Corpus.from_tabular(**kwargs)
        assert isinstance(corp, c.Corpus)
        assert corp.language == 'de'
        assert corp.max_workers == 1

        if testtype in {2, 3}:
            assert len(corp) == 100
        elif testtype == 4:
            assert len(corp) == 103
        else:  # testtype == 5
            assert len(corp) == 2

    ### test corpus_add_tabular ###
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}
    kwargs = dict(inplace=inplace, **common_kwargs)
    for corp in corpora_en_serial_and_parallel:
        n_docs_before = len(corp)

        if testtype == 1:
            with pytest.raises(ValueError, match='only file extensions ".csv", ".xls" and ".xlsx" are supported'):
                c.corpus_add_tabular(corp, **kwargs)
        else:
            res = c.corpus_add_tabular(corp, **kwargs)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            doclbls = set(c.doc_labels(res))
            doctxts = c.doc_texts(res)

            if testtype in {2, 3}:
                assert len(res) == n_docs_before + 100
            elif testtype == 4:
                assert len(res) == n_docs_before + 103
            else:  # testtype == 5
                assert len(doclbls) == n_docs_before + 2

            if testtype != 5:
                if n_docs_before == 0:
                    assert expected_doclbls == doclbls
                else:
                    assert expected_doclbls <= doclbls

                if testtype in {2, 4}:
                    assert doctxts['100NewsArticles-23'].startswith('The limited scope of')
                    if testtype == 4:
                        assert doctxts['3ExampleDocs-example2'] == 'Second example document.'
                else:   # testtype == 3
                    assert doctxts['100NewsArticles-23'].startswith(
                        'A vote for DeVos is a vote for resegregation\n\n'
                        'Felicia Wong is President and CEO of the Roosevelt '
                        'Institute, an economic and social policy think tank '
                        'working to re-imagine the rules so they work for all '
                        'Americans, and co-author of the forthcoming book '
                        '"Rewrite the Racial Rules: Building an Inclusive '
                        'American Economy." Randi Weingarten, President of the '
                        'American Federation of Teachers, is on Roosevelt\'s '
                        'board. The views expressed in this commentary are her '
                        'own.\n\nThe limited scope of'
                    )


@pytest.mark.parametrize('inplace, sample', [
    (True, None),
    (False, None),
    (True, 2),
])
def test_corpus_add_zip_and_from_zip(corpora_en_serial_and_parallel, inplace, sample):
    add_tabular_opts = dict(id_column='article_id', text_column='text')

    ### test Corpus.from_zip ###
    corp = c.Corpus.from_zip(os.path.join(DATADIR, 'zipdata.zip'), language='de', max_workers=1, sample=sample,
                             add_tabular_opts=add_tabular_opts)
    assert isinstance(corp, c.Corpus)
    assert corp.language == 'de'
    assert corp.max_workers == 1
    expected_n_docs = 101 if sample is None else sample
    assert len(corp) == expected_n_docs

    ### test corpus_add_zip ###
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}
    for corp in corpora_en_serial_and_parallel:
        n_docs_before = len(corp)

        res = c.corpus_add_zip(corp, os.path.join(DATADIR, 'zipdata.zip'), sample=sample,
                               add_tabular_opts=add_tabular_opts, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del corp

        doclbls = c.doc_labels(res)

        assert len(res) == n_docs_before + expected_n_docs

        if sample is None:
            assert sum(dl.startswith('100NewsArticles-') for dl in doclbls) == 100
            assert sum(dl == 'german-goethe_werther1' for dl in doclbls) == 1


@pytest.mark.parametrize('max_workers, sample', [
    (1, 10),
    (2, 10),
    (1, 2),
])
def test_corpus_from_builtin_corpus(max_workers, sample):
    builtin_corp = c.builtin_corpora_info()
    assert sorted(builtin_corp) == sorted(c.Corpus._BUILTIN_CORPORA_LOAD_KWARGS.keys())

    kwargs = {'max_workers': max_workers} if max_workers > 1 else {}
    kwargs['sample'] = sample

    for corpname in builtin_corp + ['nonexistent']:
        if corpname == 'nonexistent':
            with pytest.raises(ValueError, match=r'^built-in corpus does not exist: '):
                c.Corpus.from_builtin_corpus(corpname, **kwargs)
        else:
            lang = corpname[:2]

            if lang not in installed_lang:
                with pytest.raises(RuntimeError):
                    c.Corpus.from_builtin_corpus(corpname, **kwargs)
            else:
                corp = c.Corpus.from_builtin_corpus(corpname, **kwargs)
                assert isinstance(corp, c.Corpus)
                assert len(corp) > 0
                if sample is not None:
                    assert len(corp) == sample
                assert corp.language == lang
                assert corp.max_workers == max_workers


@pytest.mark.parametrize('attrname, data, default, inplace', [
    ['is_small', {'empty': True, 'small1': True, 'small2': True}, False, True],
    ['is_small', {'empty': True, 'small1': True, 'small2': True}, False, False],
    ['is_small', {}, False, True],
    ['is_small', {}, False, False],
    ['is_empty', {'empty': 'yes'}, 'no', True],
    ['is_empty', {'empty': 'yes'}, 'no', False],
])
def test_set_remove_document_attr(corpora_en_serial_and_parallel, attrname, data, default, inplace):
    dont_check_attrs = {'doc_attrs', 'doc_attrs_defaults'}

    for corp in corpora_en_serial_and_parallel:
        res = c.set_document_attr(corp, attrname=attrname, data=data, default=default, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del corp

        assert attrname in res.doc_attrs
        assert res.doc_attrs_defaults[attrname] == default

        tok = c.doc_tokens(res, with_attr=attrname)

        set_docs = set(data.keys())
        if attrname == 'is_small':
            expected_val = True
        else:
            expected_val = 'yes'

        for lbl, d in res.items():
            attrval = d.doc_attrs[attrname]
            tok_attrval = tok[lbl][attrname]
            if lbl in set_docs:
                assert attrval == expected_val
                assert tok_attrval == expected_val
            else:
                assert attrval == default
                assert tok_attrval == default

        res2 = c.remove_document_attr(res, attrname, inplace=inplace)
        res2 = _check_corpus_inplace_modif(res, res2, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del res

        assert attrname not in res2.doc_attrs
        assert attrname not in res2.doc_attrs_defaults.keys()

        if len(res2) > 0:
            with pytest.raises(KeyError):   # this attribute doesn't exist anymore
                c.doc_tokens(res2, with_attr=attrname)


@pytest.mark.parametrize('attrname, data, default, per_token_occurrence, inplace', [
    ['the_or_a', {'the': True, 'a': True}, False, True, True],
    ['the_or_a', {'the': True, 'a': True}, False, True, False],
    ['the_or_a', {}, False, True, True],
    ['the_or_a', {}, False, True, False],
    ['foobar_fail', {'small1': 'failure'}, '-', False, False],
    ['foobar', {'small1': ['foo'], 'small2': ['foo', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar']}, '-', False, False],
    ['foobar', {'small1': ['foo'], 'small2': ['foo', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar']}, '-', False, True],
])
def test_set_remove_token_attr(corpora_en_serial_and_parallel, attrname, data, default, per_token_occurrence,
                               inplace):
    dont_check_attrs = {'token_attrs', 'custom_token_attrs_defaults'}
    args = dict(attrname=attrname, data=data, default=default,
                per_token_occurrence=per_token_occurrence, inplace=inplace)

    for corp in corpora_en_serial_and_parallel:
        if attrname == 'foobar_fail' and len(corp) > 0:
            with pytest.raises(ValueError, match=r'^token attributes for document "small1" are neither tuple'):
                c.set_token_attr(corp, **args)
        else:
            res = c.set_token_attr(corp, **args)
            res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
            del corp

            assert attrname in res.token_attrs
            assert res.custom_token_attrs_defaults[attrname] == default

            tok = c.doc_tokens(res, with_attr=attrname)

            for lbl, d in res.items():
                assert attrname in d.token_attrs
                assert isinstance(d.custom_token_attrs[attrname], np.ndarray)
                assert attrname in tok[lbl]
                assert d.custom_token_attrs[attrname].tolist() == d[attrname] == tok[lbl][attrname]
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

            for d in res2.values():
                assert attrname not in d.token_attrs
                assert attrname not in d.custom_token_attrs

            if len(res2) > 0:
                with pytest.raises(KeyError):   # this attribute doesn't exist anymore
                    c.doc_tokens(res2, with_attr=attrname)


@pytest.mark.parametrize('testcase, inplace', [
    (1, False),
    (1, True),
    (2, False),
])
def test_corpus_retokenize(corpora_en_serial_and_parallel, testcase, inplace):
    for corp in corpora_en_serial_and_parallel:
        if testcase == 2:
            selected_docs = ['unicode1', 'unicode2'] if len(corp) > 0 else None
            c.remove_punctuation(corp, select=selected_docs)
            c.to_lowercase(corp, select=selected_docs)

        orig_vocab = c.vocabulary(corp, sort=False)
        orig_texts = c.doc_texts(corp, collapse=None)

        res = c.corpus_retokenize(corp, collapse=None, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)
        del corp

        assert c.vocabulary(res, sort=False) == orig_vocab - {''}
        assert c.doc_texts(res, collapse=None) == orig_texts


@pytest.mark.parametrize('testcase, func, select, inplace', [
    ('identity', lambda x: x, None, True),
    ('identity', lambda x: x, None, False),
    ('identity', lambda x: x, 'nonexistent', False),
    ('upper', lambda x: x.upper(), None, True),
    ('upper', lambda x: x.upper(), None, False),
    ('upper', lambda x: x.upper(), {'small1', 'small2'}, True),
    ('lower', lambda x: x.lower(), None, True),
    ('lower', lambda x: x.lower(), None, False),
    ('lower', lambda x: x.lower(), 'empty', False),
])
def test_transform_tokens_upper_lower(corpora_en_serial_and_parallel, testcase, func, select, inplace):
    for corp in corpora_en_serial_and_parallel:
        if select == 'nonexistent' or (select not in (None, []) and len(corp) == 0):
            with pytest.raises(KeyError):
                c.transform_tokens(corp, func, select=select, inplace=inplace)
        else:
            select_set = {select} if isinstance(select, str) else select
            orig_tokens = c.doc_tokens(corp, select=select_set)
            if select_set is not None:
                orig_tokens_unmod_set = c.doc_tokens(corp, select=set(corp.keys()) - set(select_set))
            else:
                orig_tokens_unmod_set = None

            if testcase == 'upper':
                trans_tokens = c.doc_tokens(c.to_uppercase(corp, inplace=False), select=select_set)
                expected = {lbl: [t.upper() for t in tok] for lbl, tok in orig_tokens.items()}
            elif testcase == 'lower':
                trans_tokens = c.doc_tokens(c.to_lowercase(corp, inplace=False), select=select_set)
                expected = {lbl: [t.lower() for t in tok] for lbl, tok in orig_tokens.items()}
            else:
                trans_tokens = None
                expected = None

            res = c.transform_tokens(corp, func, select=select, inplace=inplace)
            res = _check_corpus_inplace_modif(corp, res, inplace=inplace)
            del corp

            if testcase == 'identity':
                assert c.doc_tokens(res, select=select_set) == orig_tokens
            else:
                assert c.doc_tokens(res, select=select_set) == trans_tokens == expected

                if select_set is not None:
                    assert c.doc_tokens(res, select=set(res.keys()) - set(select_set)) == orig_tokens_unmod_set


@pytest.mark.parametrize('testcase, chars, inplace', [
    ('nochars', [], True),
    ('nochars', [], False),
    ('fewchars', ['.', ','], True),
    ('fewchars', ['.', ','], False),
    ('punct', list(string.punctuation) + [' ', '\r', '\n', '\t'], True),
])
def test_remove_chars_or_punctuation(corpora_en_serial_and_parallel, testcase, chars, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call

    for corp in corpora_en_serial_and_parallel:
        orig_vocab = c.vocabulary(corp)

        if testcase == 'punct':
            no_punct_vocab = c.vocabulary(c.remove_punctuation(corp, inplace=False))
        else:
            no_punct_vocab = None

        res = c.remove_chars(corp, chars=chars, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)
        del corp

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

    for corp in corpora_en_serial_and_parallel:
        orig_vocab = c.vocabulary(corp)
        orig_tok = c.doc_tokens(corp)

        res = c.normalize_unicode(corp, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)
        del corp

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

    for corp in corpora_en_serial_and_parallel:
        orig_vocab = c.vocabulary(corp)

        if method == 'icu' and len(corp) > 0 and not find_spec('icu'):
            with pytest.raises(RuntimeError, match=r'^package PyICU'):
                c.simplify_unicode(corp, method=method, inplace=inplace)
        else:
            res = c.simplify_unicode(corp, method=method, inplace=inplace)
            res = _check_corpus_inplace_modif(corp, res, inplace=inplace)
            del corp

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
def test_numbers_to_magnitudes(corpora_en_serial_and_parallel, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call

    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0
        orig_vocab = c.vocabulary(corp)
        res = c.numbers_to_magnitudes(corp, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)
        del corp

        new_vocab = c.vocabulary(res)
        assert len(new_vocab) <= len(orig_vocab)

        if not emptycorp:
            assert '180,000' in orig_vocab
            assert '100000' in new_vocab


@pytest.mark.parametrize('inplace', [True, False])
def test_lemmatize(corpora_en_serial_and_parallel, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call

    for corp in corpora_en_serial_and_parallel:
        orig_lemmata = {lbl: tok['lemma'] for lbl, tok in c.doc_tokens(corp, with_attr='lemma').items()}
        res = c.lemmatize(corp, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)
        del corp

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
    args = dict(patterns=patterns, glue=glue, match_type=match_type, return_joint_tokens=return_joint_tokens,
                inplace=inplace)

    for corp in corpora_en_serial_and_parallel:
        c.set_token_attr(corp, 'foo', data={'the': True}, default=False)

        if not isinstance(patterns, (list, tuple)) or len(patterns) < 2:
            with pytest.raises(ValueError, match=r'`patterns` must be a list or tuple containing at least two '
                                                 r'elements'):
                c.join_collocations_by_patterns(corp, **args)
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

            res = _check_corpus_inplace_modif(corp, res, inplace=inplace)
            del corp

            for a in res.custom_token_attrs_defaults.keys():
                for d in res.values():
                    assert len(d[a]) == len(d)

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
       test_w_tokenattr=st.booleans(),
       return_joint_tokens=st.booleans())
def test_join_collocations_by_statistic_hypothesis(corpora_en_serial_and_parallel_module, threshold, glue, min_count,
                                                   embed_tokens_min_docfreq, pass_embed_tokens, test_w_tokenattr,
                                                   return_joint_tokens):
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

        if test_w_tokenattr:
            corp = c.set_token_attr(corp, 'foo', data={'the': True}, default=False, inplace=False)

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

        assert all([glue in t for t in colloc])

        vocab = c.vocabulary(res, sort=False)
        assert len(set(colloc)) <= len(vocab)

        if test_w_tokenattr:
            for a in res.custom_token_attrs_defaults.keys():
                for d in res.values():
                    assert len(d[a]) == len(d)

        # if return_joint_tokens:    # TODO: sometimes this breaks, dunno why
        #     assert joint_tokens == set(colloc)


@pytest.mark.parametrize('inverse, inplace', [
    (False, True),
    (True, False),
    (False, False),
    (True,  True),
])
def test_filter_tokens_by_mask(corpora_en_serial_and_parallel, inverse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call

    mask1 = {'small2': [True, False, False, True, True, True, False]}
    mask2 = {'small2': [False, True]}

    for corp in corpora_en_serial_and_parallel:
        if len(corp) == 0:
            with pytest.raises(ValueError, match=r'does not exist in Corpus object `docs`'):
                c.filter_tokens_by_mask(corp, mask=mask1, inverse=inverse, inplace=inplace)

            with pytest.raises(ValueError, match=r'does not exist in Corpus object `docs`'):
                c.remove_tokens_by_mask(corp, mask=mask1, inplace=False)
        else:
            res = c.filter_tokens_by_mask(corp, mask=mask1, inverse=inverse, inplace=inplace)
            res = _check_corpus_inplace_modif(corp, res, inplace=inplace)

            tok = c.doc_tokens(res, select='small2')
            if inverse:
                assert tok == ['is', 'a', '.']
            else:
                assert tok == ['This', 'small', 'example', 'document']

            if inverse and not inplace:
                res_inv = c.remove_tokens_by_mask(corp, mask=mask1, inplace=False)
                assert c.doc_tokens(res_inv, select='small2') == tok

            with pytest.raises(ValueError, match=r'^length of provided mask for document '):
                c.filter_tokens_by_mask(res, mask=mask2, inverse=inverse, inplace=inplace)

            with pytest.raises(ValueError, match=r'^length of provided mask for document '):
                c.remove_tokens_by_mask(res, mask=mask2, inplace=inplace)


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

    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0

        if testtype == 7:
            c.set_token_attr(corp, 'is_the', {'the': True}, default=False)

        res = c.filter_tokens(corp, search_tokens, by_attr=by_attr, match_type=match_type, ignore_case=ignore_case,
                              glob_method=glob_method, inverse=inverse, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)

        vocab = c.vocabulary(res, sort=False)

        if inverse:
            res_inv = c.remove_tokens(corp, search_tokens, by_attr=by_attr, match_type=match_type,
                                      ignore_case=ignore_case, glob_method=glob_method, inplace=False)
            vocab_inv = c.vocabulary(res_inv, sort=False)
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
                assert all([t == ' ' for tok in tokens_ws.values() for t in tok['whitespace']])
            elif testtype == 5:
                assert all([t.startswith('Dis') for t in vocab])
            elif testtype == 6:
                assert all(['y' in t for t in vocab])
            elif testtype == 7:
                assert vocab == {'the'}
            else:
                raise ValueError(f'unknown testtype {testtype}')


def test_filter_tokens_custom_attr_bug(corpora_en_serial_and_parallel):
    # check that when setting a custom token attribute, this attribute's data is also filtered when using a filtering
    # function
    for corp in corpora_en_serial_and_parallel:
        doctoks = c.doc_tokens(corp)
        attrdata = {lbl: random.randint(0, 1) for lbl, tok in doctoks.items()}
        c.set_token_attr(corp, 'testattr', data=attrdata, per_token_occurrence=True)
        c.filter_tokens(corp, 1, by_attr='testattr')

        for lbl, doc in corp.items():
            assert all(v == 1 for v in doc['testattr'])
            assert len(doc) == len(doc['testattr'])


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
    for corp in corpora_en_serial_and_parallel:
        emptycorp = len(corp) == 0
        res = c.filter_for_pos(corp, search_pos=search_pos, simplify_pos=simplify_pos, inverse=inverse, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)

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
    (1, 'common', 0.5, 1, False, False, True),
    (1, 'common', 0.5, 1, False, False, False),
    (1, '>=', 0.5, 1, False, False, True),
    (1, '<', 0.5, 1, False, True, True),
    (1, '<', math.log(0.5), 2, False, True, True),
    (2, 'uncommon', 3, 0, False, False, True),
    (2, 'uncommon', 3, 0, True, False, True),
    (3, 'common', 0.7, 1, False, True, True),
    (4, 'uncommon', 0.3, 1, False, True, True),
])
def test_filter_tokens_by_doc_frequency(corpora_en_serial_and_parallel, testtype, which, df_threshold, proportions,
                                        return_filtered_tokens, inverse, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call

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
            vocab_remove = c.vocabulary(res_remove, sort=False)
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

        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)
        vocab = c.vocabulary(res, sort=False)

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
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}

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
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}

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
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}

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

    for corp in corpora_en_serial_and_parallel:
        vocab_before = c.vocabulary(corp, sort=False)
        res = c.filter_clean_tokens(corp,
                                    remove_punct=remove_punct,
                                    remove_stopwords=remove_stopwords,
                                    remove_empty=remove_empty,
                                    remove_shorter_than=remove_shorter_than,
                                    remove_longer_than=remove_longer_than,
                                    remove_numbers=remove_numbers,
                                    inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)

        vocab = c.vocabulary(res, sort=False)
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

    for corp in corpora_en_serial_and_parallel:
        doctok_before = c.doc_tokens(corp)

        if by_attr == 'is_the':
            c.set_token_attr(corp, by_attr, {'the': True}, default=False)

        res = c.filter_tokens_with_kwic(corp, search_tokens=search_tokens, context_size=context_size, by_attr=by_attr,
                                        match_type=match_type, ignore_case=ignore_case, glob_method=glob_method,
                                        inverse=inverse, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, inplace=inplace)

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


@pytest.mark.parametrize('n, join_str, inplace', [
    (2, ' ', True),
    (2, ' ', False),
    (2, '_', True),
    (3, '//', True),
    (1, ' ', True),
])
def test_corpus_ngramify(corpora_en_serial_and_parallel, n, join_str, inplace):
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

        res = c.corpus_ngramify(corp, n=n, join_str=join_str, inplace=inplace)
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


@pytest.mark.parametrize('n, inplace', [
    (0, True),
    (1, True),
    (2, True),
    (2, False),
    (9, True),
    (100, True),
])
def test_corpus_sample(corpora_en_serial_and_parallel, n, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}

    for corp in corpora_en_serial_and_parallel:
        if len(corp) == 0:
            with pytest.raises(ValueError, match=r'cannot sample from empty corpus'):
                c.corpus_sample(corp, n, inplace=inplace)
        else:
            if n < 1 or n > len(corp):
                with pytest.raises(ValueError, match=r'`n` must be between 1 and '):
                    c.corpus_sample(corp, n, inplace=inplace)
            else:
                res = c.corpus_sample(corp, n, inplace=inplace)
                res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
                del corp

                assert len(res) == n


@pytest.mark.parametrize('inplace', [True, False])
def test_corpus_split_by_paragraph(corpora_en_serial_and_parallel, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}

    for corp in corpora_en_serial_and_parallel:
        n_docs_before = len(corp)
        res = c.corpus_split_by_paragraph(corp, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del corp

        if n_docs_before > 0:
            assert n_docs_before < len(res)
            texts = c.doc_texts(res)
            for lbl in {'empty', 'small1', 'small2', 'unicode1', 'unicode2'}:
                assert lbl in texts.keys()
                assert texts[lbl] == textdata_en[lbl]

            assert texts['NewsArticles-1-1'] == 'Disney Parks Just Got More Expensive As Ticket Prices Rise Again\n\n'
            assert texts['NewsArticles-1-2'] == 'A single day in a Disney park can cost as much as $124.\n\n'
        else:
            assert n_docs_before == len(res)


@pytest.mark.parametrize('join, glue, match_type, doc_opts, inplace', [
    ({}, '\n\n', 'exact', None, False),
    ({}, '\n\n', 'exact', None, True),
    ({'foo': 'nonexistent'}, '\n\n', 'exact', None, True),
    ({'joint-unicode': 'unicode*'}, '\n\n', 'glob', None, False),
    ({'joint-unicode': 'unicode*'}, '\n\n', 'glob', None, True),
    ({'joint-unicode': 'unicode*', 'joint-small': 'small*'}, '\n\n', 'glob', None, True),
    ({'joint-unicode': 'unicode*', 'joint-small': 'small*'}, '', 'glob', None, True),
    ({'joint-unicode': 'unicode*', 'joint-small': 'small*'}, '', 'glob', {'doc_attrs': {'is_joint': True}}, True),
    ({'new-empty': '^empty'}, '\n\n', 'regex', None, True),
])
def test_corpus_join_documents(corpora_en_serial_and_parallel, join, glue, match_type, doc_opts, inplace):
    # using corpora_en_serial_and_parallel fixture here which is re-instantiated on each test function call
    dont_check_attrs = {'doc_labels', 'n_docs', 'workers_docs'}

    for corp in corpora_en_serial_and_parallel:
        n_docs_before = len(corp)
        texts_before = c.doc_texts(corp)
        res = c.corpus_join_documents(corp, join=join, glue=glue, match_type=match_type, doc_opts=doc_opts, inplace=inplace)
        res = _check_corpus_inplace_modif(corp, res, dont_check_attrs=dont_check_attrs, inplace=inplace)
        del corp

        if n_docs_before > 0:
            texts = c.doc_texts(res)

            if not join or 'foo' in join:
                assert texts == texts_before
            else:
                if 'joint-unicode' in set(join.keys()):
                    assert texts['joint-unicode'] == texts_before['unicode1'] + glue + texts_before['unicode2']
                    assert 'unicode1' not in res
                    assert 'unicode2' not in res

                    if doc_opts:
                        assert res['joint-unicode'].doc_attrs['is_joint']

                if 'joint-small' in set(join.keys()):
                    assert texts['joint-small'] == texts_before['small1'] + glue + texts_before['small2']
                    assert 'small1' not in res
                    assert 'small2' not in res

                    if doc_opts:
                        assert res['joint-small'].doc_attrs['is_joint']

                if 'new-empty' in set(join.keys()):
                    assert texts['new-empty'] == ''
                    assert 'empty' not in res
        else:
            assert n_docs_before == len(res)


#%% other functions

@pytest.mark.parametrize('with_paths', [False, True])
def test_builtin_corpora_info(with_paths):
    corpinfo = c.builtin_corpora_info(with_paths=with_paths)
    if with_paths:
        assert isinstance(corpinfo, dict)
        corpnames = list(corpinfo.keys())
        for name, path in corpinfo.items():
            namecomponents = name.split('-')
            assert path.endswith(os.path.join('data', namecomponents[0], f'{"-".join(namecomponents[1:])}.zip'))
    else:
        assert isinstance(corpinfo, list)
        corpnames = corpinfo

        for name in corpnames:
            lang = name[:2]

            if lang not in installed_lang:
                with pytest.raises(RuntimeError):
                    c.Corpus.from_builtin_corpus(name, load_features=[], sample=5)
            else:
                corp = c.Corpus.from_builtin_corpus(name, load_features=[], sample=5)
                assert isinstance(corp, c.Corpus)
                assert corp.language == lang

    assert set(corpnames) == set(c.Corpus._BUILTIN_CORPORA_LOAD_KWARGS.keys())


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
            assert len(corp) == 0
        else:
            assert len(corp) == 6

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

        dtm_final = c.dtm(corp)
        assert np.all(dtm_final.todense() == c.dtm(corp).todense())
        assert dtm_final.shape == (len(corp), c.vocabulary_size(corp))


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


def _check_corpus_docs(corp: c.Corpus, has_sents: bool):
    for lbl, d in corp.items():
        assert isinstance(d, c.Document)
        assert d.label == lbl
        assert d.has_sents == has_sents
        assert d.bimaps is corp.bimaps
        assert isinstance(d.tokenmat, np.ndarray)
        assert d.tokenmat.ndim == 2
        assert np.issubdtype(d.tokenmat.dtype, 'uint64')
        assert len(d) >= 0
        assert len(d) == len(d.tokenmat)
        assert isinstance(d.tokenmat_attrs, list)
        assert len(d.tokenmat_attrs) == d.tokenmat.shape[1]
        if d.has_sents:
            assert 'sent_start' in d.tokenmat_attrs
        assert set(d.tokenmat_attrs) <= set(d.token_attrs)
        tok = d['token']
        assert isinstance(tok, list)
        assert len(tok) == len(d)


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
        check_attrs = {'uses_unigrams', 'token_attrs', 'custom_token_attrs_defaults', 'doc_attrs',
                       'doc_attrs_defaults', 'ngrams', 'ngrams_join_str', 'language', 'language_model',
                       'doc_labels', 'n_docs', 'workers_docs', 'max_workers', 'raw_preproc'}

    if dont_check_attrs is not None:
        check_attrs.difference_update(dont_check_attrs)

    for attr in check_attrs:
        assert attr in attrs_a
        assert attr in attrs_b
        val_a = getattr(corp_a, attr)
        val_b = getattr(corp_b, attr)

        if attr in {'token_attrs', 'doc_attrs', 'doc_labels'}:  # for these attribs, we can't guarantee the same order
            val_a = set(val_a)
            val_b = set(val_b)

        assert val_a == val_b

    if same_nlp_instance:
        assert corp_a.nlp is corp_b.nlp
    else:
        assert corp_a.nlp is not corp_b.nlp
        assert corp_a.nlp.meta == corp_b.nlp.meta


def _dataframes_equal(df1, df2, require_same_index=True):
    if require_same_index:
        comp_res = df1 == df2
    else:
        comp_res = df1.reset_index(drop=True) == df2.reset_index(drop=True)
    return df1.shape == df2.shape and comp_res.all(axis=1).sum() == len(df1)
