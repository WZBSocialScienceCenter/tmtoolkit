"""
Preprocessing: TMPreproc tests.
"""

import functools
import tempfile
from random import sample
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import nltk
import pytest
from scipy import sparse

from tmtoolkit._pd_dt_compat import USE_DT, FRAME_TYPE, pd_dt_frame, pd_dt_colnames, pd_dt_frame_to_list
from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess._common import simplified_pos
from tmtoolkit.bow.bow_stats import tfidf


#%% data preparation / helper functions

MAX_DOC_LEN = 5000
N_DOCS_EN = 7
N_DOCS_DE = 3   # given from corpus size

# create a sample of English corpus documents
all_docs_en = {f_id: nltk.corpus.gutenberg.raw(f_id) for f_id in nltk.corpus.gutenberg.fileids()}
smaller_docs_en = [(dl, txt[:min(nchar, MAX_DOC_LEN)])
                   for dl, txt, nchar in map(lambda x: (x[0], x[1], len(x[1])), all_docs_en.items())]

# make sure we always have moby dick because we use it in filter_* tests
corpus_en = Corpus(dict(sample([(dl, txt) for dl, txt in smaller_docs_en if dl != 'melville-moby_dick.txt'],
                               N_DOCS_EN-2)))
corpus_en.docs['empty_doc'] = ''  # additionally test empty document
corpus_en.docs['melville-moby_dick.txt'] = dict(smaller_docs_en)['melville-moby_dick.txt']

# get all documents from german corpus
corpus_de = Corpus.from_folder('examples/data/gutenberg', read_size=MAX_DOC_LEN)


def _dataframes_equal(df1, df2):
    # so far, datatable doesn't seem to support dataframe comparisons
    if USE_DT:
        if isinstance(df1, FRAME_TYPE):
            df1 = df1.to_pandas()
        if isinstance(df2, FRAME_TYPE):
            df2 = df2.to_pandas()
    return df1.shape == df2.shape and (df1 == df2).all(axis=1).sum() == len(df1)


def _check_save_load_state(preproc, repeat=1, recreate_from_state=False):
    # attributes to check
    simple_state_attrs = ('n_docs', 'n_tokens', 'doc_lengths', 'vocabulary_counts',
                          'language', 'stopwords', 'punctuation', 'special_chars',
                          'n_workers', 'n_max_workers', 'pos_tagset',
                          'pos_tagged', 'ngrams_generated', 'ngrams_as_tokens',
                          'doc_labels', 'vocabulary')

    # save the state for later comparisons
    pre_state = {attr: deepcopy(getattr(preproc, attr)) for attr in simple_state_attrs}
    pre_state['tokens'] = preproc.tokens
    pre_state['tokens_datatable'] = preproc.tokens_datatable
    pre_state['dtm'] = preproc.dtm

    if preproc.pos_tagged:
        pre_state['tokens_with_pos_tags'] = preproc.tokens_with_pos_tags
    if preproc.ngrams_generated:
        pre_state['ngrams'] = preproc.ngrams

    # save and then load the same state
    for _ in range(repeat):
        if recreate_from_state:
            with tempfile.TemporaryFile(suffix='.pickle') as ftemp:
                preproc.save_state(ftemp)
                ftemp.seek(0)
                preproc = TMPreproc.from_state(ftemp)
        else:
            with tempfile.TemporaryFile(suffix='.pickle') as ftemp:
                preproc.save_state(ftemp)
                ftemp.seek(0)
                preproc.load_state(ftemp)

    # check if simple attributes are the same
    attrs_preproc = dir(preproc)
    for attr in simple_state_attrs:
        assert attr in attrs_preproc
        assert pre_state[attr] == getattr(preproc, attr)

    # check if tokens are the same
    assert set(pre_state['tokens'].keys()) == set(preproc.tokens.keys())
    assert all(pre_state['tokens'][k] == preproc.tokens[k]
               for k in preproc.tokens.keys())

    # check if token dataframes are the same
    assert _dataframes_equal(pre_state['tokens_datatable'], preproc.tokens_datatable)

    # for DTMs, check the shape only
    assert pre_state['dtm'].shape == preproc.dtm.shape

    # optionally check POS tagged data frames
    if preproc.pos_tagged:
        assert set(pre_state['tokens_with_pos_tags'].keys()) == set(preproc.tokens_with_pos_tags.keys())
        assert all(_dataframes_equal(pre_state['tokens_with_pos_tags'][k], preproc.tokens_with_pos_tags[k])
                   for k in preproc.tokens_with_pos_tags.keys())

    # optionally check ngrams
    if preproc.ngrams_generated:
        assert set(pre_state['ngrams'].keys()) == set(preproc.ngrams.keys())
        assert all(pre_state['ngrams'][k] == preproc.ngrams[k]
                   for k in preproc.ngrams.keys())


def _check_TMPreproc_copies(preproc_a, preproc_b, shutdown_b_workers=True):
    attrs_a = dir(preproc_a)
    attrs_b = dir(preproc_b)

    # check if simple attributes are the same
    simple_state_attrs = ('n_docs', 'n_tokens', 'doc_lengths', 'vocabulary_counts',
                          'language', 'stopwords', 'punctuation', 'special_chars',
                          'n_workers', 'n_max_workers', 'pos_tagset',
                          'pos_tagged', 'ngrams_generated', 'ngrams_as_tokens',
                          'doc_labels', 'vocabulary')

    for attr in simple_state_attrs:
        assert attr in attrs_a
        assert attr in attrs_b
        assert getattr(preproc_a, attr) == getattr(preproc_b, attr)

    # check if tokens are the same
    assert set(preproc_a.tokens.keys()) == set(preproc_b.tokens.keys())
    assert all(preproc_a.tokens[k] == preproc_b.tokens[k]
               for k in preproc_a.tokens.keys())

    # check if token dataframes are the same
    assert _dataframes_equal(preproc_a.tokens_datatable, preproc_b.tokens_datatable)

    # for DTMs, check the shape only
    assert preproc_a.dtm.shape == preproc_b.dtm.shape

    # optionally check POS tagged data frames
    if preproc_a.pos_tagged:
        assert set(preproc_a.tokens_with_pos_tags.keys()) == set(preproc_b.tokens_with_pos_tags.keys())
        assert all(_dataframes_equal(preproc_a.tokens_with_pos_tags[k], preproc_b.tokens_with_pos_tags[k])
                   for k in preproc_a.tokens_with_pos_tags.keys())

    # optionally check ngrams
    if preproc_a.ngrams_generated:
        assert set(preproc_a.ngrams.keys()) == set(preproc_b.ngrams.keys())
        assert all(preproc_a.ngrams[k] == preproc_b.ngrams[k]
                   for k in preproc_a.ngrams.keys())

    if shutdown_b_workers:
        preproc_b.shutdown_workers()


def preproc_test(lang='en', make_checks=True, repeat_save_load=1, recreate_from_state=False):
    """TMPreproc decorator"""

    def decorator_wrapper(fn):
        if lang == 'en':
            @functools.wraps(fn)
            def wrapper(tmpreproc_en, *args, **kwargs):
                fn(tmpreproc_en, *args, **kwargs)

                if make_checks:
                    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
                    _check_save_load_state(tmpreproc_en, repeat=repeat_save_load,
                                           recreate_from_state=recreate_from_state)

                tmpreproc_en.shutdown_workers()
        elif lang == 'de':
            @functools.wraps(fn)
            def wrapper(tmpreproc_de, *args, **kwargs):
                fn(tmpreproc_de, *args, **kwargs)

                if make_checks:
                    _check_TMPreproc_copies(tmpreproc_de, tmpreproc_de.copy())
                    _check_save_load_state(tmpreproc_de, repeat=repeat_save_load,
                                           recreate_from_state=recreate_from_state)
                tmpreproc_de.shutdown_workers()
        else:
            raise ValueError('language not supported:', lang)

        return wrapper

    return decorator_wrapper


@pytest.fixture
def tmpreproc_en():
    return TMPreproc(corpus_en.docs, language='english')


@pytest.fixture
def tmpreproc_de():
    return TMPreproc(corpus_de.docs, language='german')


def test_fixtures_n_docs_and_doc_labels(tmpreproc_en, tmpreproc_de):
    assert tmpreproc_en.n_docs == N_DOCS_EN
    assert tmpreproc_de.n_docs == N_DOCS_DE

    assert list(sorted(tmpreproc_en.doc_labels)) == list(sorted(corpus_en.docs.keys()))
    assert list(sorted(tmpreproc_de.doc_labels)) == list(sorted(corpus_de.docs.keys()))

    tmpreproc_en.shutdown_workers()
    tmpreproc_de.shutdown_workers()


#%% Tests with empty corpus

def test_tmpreproc_empty_corpus():
    preproc = TMPreproc({})

    assert preproc.n_docs == 0
    assert preproc.doc_labels == []

    preproc.stem().tokens_to_lowercase().clean_tokens().filter_documents('Moby')

    assert preproc.n_docs == 0
    assert preproc.doc_labels == []

    _check_TMPreproc_copies(preproc, preproc.copy())
    _check_TMPreproc_copies(preproc, copy(preproc))
    _check_TMPreproc_copies(preproc, deepcopy(preproc))

    preproc.shutdown_workers()


#%% Tests with English corpus


@preproc_test(make_checks=False)
def test_tmpreproc_en_init(tmpreproc_en):
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    with pytest.raises(ValueError):    # because not POS tagged
        assert tmpreproc_en.tokens_with_pos_tags

    tmpreproc_en.ngrams == {}
    assert tmpreproc_en.ngrams_generated is False


@preproc_test()
def test_tmpreproc_en_add_stopwords(tmpreproc_en):
    sw = set(tmpreproc_en.stopwords)
    sw_ = set(tmpreproc_en.add_stopwords(['foobar']).stopwords)
    assert sw_ == sw | {'foobar'}


@preproc_test()
def test_tmpreproc_en_add_punctuation(tmpreproc_en):
    pct = set(tmpreproc_en.punctuation)
    pct_ = set(tmpreproc_en.add_punctuation(['X']).punctuation)
    assert pct_ == pct | {'X'}


@preproc_test()
def test_tmpreproc_en_add_special_chars(tmpreproc_en):
    sc = set(tmpreproc_en.special_chars)
    sc_ = set(tmpreproc_en.add_special_chars(['X']).special_chars)
    assert sc_ == sc | {'X'}


@preproc_test()
def test_tmpreproc_en_add_metadata_per_token_and_remove_metadata(tmpreproc_en):
    meta = {
        'Moby': 'important',
        'Ahab': 'important',
        'is': 'unimportant',
    }

    # first meta data
    tmpreproc_en.add_metadata_per_token('importance', meta, default='')

    assert 'importance' in tmpreproc_en.get_available_metadata_keys()

    for dl, df in tmpreproc_en.tokens_with_metadata.items():
        assert 'meta_importance' in pd_dt_colnames(df)

        if USE_DT:
            df = df.to_pandas()

        for k, v in meta.items():
            if sum(df.token == k) > 0:
                assert (df.loc[df.token == k, 'meta_importance'] == v).all()

        if sum(~df.token.isin(set(meta.keys())) > 0):
            assert (df.loc[~df.token.isin(set(meta.keys())), 'meta_importance'] == '').all()

    # second meta data with same values
    tmpreproc_en.add_metadata_per_token('foobar', meta, default='defaultfoo')

    assert 'importance' in tmpreproc_en.get_available_metadata_keys()
    assert 'foobar' in tmpreproc_en.get_available_metadata_keys()

    for dl, df in tmpreproc_en.tokens_with_metadata.items():
        assert 'meta_importance' in pd_dt_colnames(df)
        assert 'meta_foobar' in pd_dt_colnames(df)

        if USE_DT:
            df = df.to_pandas()

        for k, v in meta.items():
            if sum(df.token == k) > 0:
                assert (df.loc[df.token == k, 'meta_importance'] == v).all()
                assert (df.loc[df.token == k, 'meta_foobar'] == v).all()

        if sum(~df.token.isin(set(meta.keys())) > 0):
            assert (df.loc[~df.token.isin(set(meta.keys())), 'meta_importance'] == '').all()
            assert (df.loc[~df.token.isin(set(meta.keys())), 'meta_foobar'] == 'defaultfoo').all()

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.remove_metadata('foobar')

    assert 'foobar' not in tmpreproc_en.get_available_metadata_keys()

    for dl, df in tmpreproc_en.tokens_with_metadata.items():
        assert 'meta_importance' in pd_dt_colnames(df)
        assert 'meta_foobar' not in pd_dt_colnames(df)

    tmpreproc_en.remove_metadata('importance')

    assert 'importance' not in tmpreproc_en.get_available_metadata_keys()

    for dl, df in tmpreproc_en.tokens_with_metadata.items():
        assert 'meta_importance' not in pd_dt_colnames(df)
        assert 'meta_foobar' not in pd_dt_colnames(df)


@preproc_test(make_checks=False)
def test_tmpreproc_en_add_metadata_per_doc_and_remove_metadata(tmpreproc_en):
    doc_lengths = tmpreproc_en.doc_lengths
    first_doc = next(iter(doc_lengths))
    docs = [first_doc, 'melville-moby_dick.txt']

    meta = {}
    for dl in docs:
        if dl not in meta:
            meta[dl] = np.random.randint(0, 10, size=doc_lengths[dl])

    tmpreproc_en.add_metadata_per_doc('random_foo', meta, default=-1)

    assert 'random_foo' in tmpreproc_en.get_available_metadata_keys()

    for dl, df in tmpreproc_en.tokens_with_metadata.items():
        assert 'meta_random_foo' in pd_dt_colnames(df)

        if USE_DT:
            df = df.to_pandas()

        if dl in meta:
            assert np.array_equal(df.meta_random_foo.values, meta[dl])
        else:
            assert (df.meta_random_foo == -1).all()

    # setting meta data to document that does not exist should fail:
    meta = {'non-existent': []}
    with pytest.raises(ValueError):
        tmpreproc_en.add_metadata_per_doc('will_fail', meta)

    # setting meta data with length that does not match number of tokens should fail:
    meta = {'melville-moby_dick.txt': [1, 2, 3]}
    with pytest.raises(ValueError):
        tmpreproc_en.add_metadata_per_doc('will_fail', meta)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    # remove meta data again
    tmpreproc_en.remove_metadata('random_foo')

    assert 'random_foo' not in tmpreproc_en.get_available_metadata_keys()

    for dl, df in tmpreproc_en.tokens_with_metadata.items():
        assert 'meta_random_foo' not in pd_dt_colnames(df)

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    # does not exist anymore:
    with pytest.raises(ValueError):
        tmpreproc_en.remove_metadata('random_foo')


@preproc_test(repeat_save_load=5)
def test_tmpreproc_en_save_load_state_several_times(tmpreproc_en):
    assert tmpreproc_en.language == 'english'


@preproc_test(recreate_from_state=True)
def test_tmpreproc_en_save_load_state_recreate_from_state(tmpreproc_en):
    assert tmpreproc_en.language == 'english'


@preproc_test()
def test_tmpreproc_en_create_from_tokens(tmpreproc_en):
    preproc2 = TMPreproc.from_tokens(tmpreproc_en.tokens)

    assert set(tmpreproc_en.tokens.keys()) == set(preproc2.tokens.keys())
    assert all(preproc2.tokens[k] == tmpreproc_en.tokens[k]
               for k in tmpreproc_en.tokens.keys())


@preproc_test(make_checks=False)
def test_tmpreproc_en_load_tokens(tmpreproc_en):
    # two modification: remove word "Moby" and remove a document
    tokens = {}
    removed_doc = None
    for i, (dl, dtok) in enumerate(tmpreproc_en.tokens.items()):
        if i > 0:
            tokens[dl] = [t for t in dtok if t != 'Moby']
        else:
            removed_doc = dl

    assert removed_doc is not None

    assert removed_doc in tmpreproc_en.doc_labels
    assert 'Moby' in tmpreproc_en.vocabulary

    tmpreproc_en.load_tokens(tokens)

    assert removed_doc not in tmpreproc_en.doc_labels
    assert 'Moby' not in tmpreproc_en.vocabulary


@preproc_test(make_checks=False)
def test_tmpreproc_en_load_tokens_with_metadata(tmpreproc_en):
    meta = {
        'Moby': 'important',
        'Ahab': 'important',
        'is': 'unimportant',
    }

    # add meta data
    tmpreproc_en.add_metadata_per_token('importance', meta, default='')

    # two modifications: remove word marked as unimportant and remove a document
    tokens = {}
    removed_doc = None
    n_unimp = 0
    for i, (dl, doc) in enumerate(tmpreproc_en.tokens_with_metadata.items()):
        assert pd_dt_colnames(doc) == ['token', 'meta_importance']

        if USE_DT:
            doc = doc.to_pandas()

        if i > 0:
            n_unimp += sum(doc.meta_importance != 'unimportant')
            tokens[dl] = pd_dt_frame(doc.loc[doc.meta_importance != 'unimportant', :],
                                     colnames=['token', 'meta_importance'])
        else:
            removed_doc = dl

    assert n_unimp > 0
    assert removed_doc is not None
    assert removed_doc in tmpreproc_en.doc_labels

    tmpreproc_en.load_tokens(tokens)

    assert removed_doc not in tmpreproc_en.doc_labels

    for i, (dl, doc) in enumerate(tmpreproc_en.tokens_with_metadata.items()):
        assert pd_dt_colnames(doc) == ['token', 'meta_importance']

        if USE_DT:
            doc = doc.to_pandas()

        assert sum(doc.meta_importance == 'unimportant') == 0


@preproc_test(make_checks=False)
def test_tmpreproc_en_load_tokens_datatable(tmpreproc_en):
    if not USE_DT:
        pytest.skip('datatable not installed')

    tokensdf = tmpreproc_en.tokens_datatable

    if USE_DT:
        from datatable import f
        tokensdf = tokensdf[f.token != 'Moby', :]
    else:
        tokensdf = tokensdf.loc[tokensdf.token != 'Moby', :]

    assert 'Moby' in tmpreproc_en.vocabulary

    tmpreproc_en.load_tokens_datatable(tokensdf)

    assert 'Moby' not in tmpreproc_en.vocabulary


@preproc_test(make_checks=False)
def test_tmpreproc_en_create_from_tokens_datatable(tmpreproc_en):
    if not USE_DT:
        pytest.skip('datatable not installed')

    preproc2 = TMPreproc.from_tokens_datatable(tmpreproc_en.tokens_datatable)

    assert _dataframes_equal(preproc2.tokens_datatable, tmpreproc_en.tokens_datatable)


@preproc_test()
def test_tmpreproc_en_tokens_property(tmpreproc_en):
    tokens_all = tmpreproc_en.tokens
    assert set(tokens_all.keys()) == set(corpus_en.docs.keys())

    for dtok in tokens_all.values():
        assert isinstance(dtok, list)
        if len(dtok) > 0:
            # make sure that not all tokens only consist of a single character:
            assert np.sum(np.char.str_len(dtok) > 1) > 1


@preproc_test()
def test_tmpreproc_en_get_tokens_and_tokens_with_metadata_property(tmpreproc_en):
    tokens_from_prop = tmpreproc_en.tokens
    tokens_w_meta = tmpreproc_en.tokens_with_metadata
    assert set(tokens_w_meta.keys()) == set(corpus_en.docs.keys())

    tokens_w_meta_from_fn = tmpreproc_en.get_tokens(with_metadata=True, as_datatables=True)

    for dl, df in tokens_w_meta.items():
        assert _dataframes_equal(df, tokens_w_meta_from_fn[dl])

        assert pd_dt_colnames(df) == ['token']
        if USE_DT:
            assert df[:, 'token'].to_list()[0] == list(tokens_from_prop[dl])
        else:
            assert df.loc[:, 'token'].to_list() == list(tokens_from_prop[dl])


@preproc_test()
def test_tmpreproc_en_get_tokens_non_empty(tmpreproc_en):
    tokens = tmpreproc_en.get_tokens(non_empty=True)
    assert set(tokens.keys()) == set(corpus_en.docs.keys()) - {'empty_doc'}


@preproc_test(make_checks=False)
def test_tmpreproc_en_doc_lengths(tmpreproc_en):
    doc_lengths = tmpreproc_en.doc_lengths
    assert set(doc_lengths.keys()) == set(corpus_en.docs.keys())

    for dl, dtok in tmpreproc_en.tokens.items():
        assert doc_lengths[dl] == len(dtok)


@preproc_test()
def test_tmpreproc_en_tokens_datatable(tmpreproc_en):
    doc_lengths = tmpreproc_en.doc_lengths

    df = tmpreproc_en.tokens_datatable
    assert isinstance(df, FRAME_TYPE)
    assert pd_dt_colnames(df) == ['doc', 'position', 'token']
    assert df.shape[0] == sum(doc_lengths.values())

    df_as_list = pd_dt_frame_to_list(df)

    ind0 = df_as_list[0]
    labels, lens = zip(*sorted(doc_lengths.items(), key=lambda x: x[0]))
    assert np.array_equal(ind0, np.repeat(labels, lens))

    ind1 = df_as_list[1]
    expected_indices = []
    for n in lens:
        expected_indices.append(np.arange(n))

    assert np.array_equal(ind1, np.concatenate(expected_indices))


@preproc_test()
def test_tmpreproc_en_tokens_dataframe(tmpreproc_en):
    doc_lengths = tmpreproc_en.doc_lengths

    df = tmpreproc_en.tokens_dataframe
    assert isinstance(df, pd.DataFrame)
    assert list(df.index.names) == ['doc', 'position']
    assert list(df.columns) == ['token']
    assert df.shape[0] == sum(doc_lengths.values())


@preproc_test()
def test_tmpreproc_en_vocabulary(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    vocab = tmpreproc_en.vocabulary

    assert isinstance(vocab, list)
    assert len(vocab) <= sum(tmpreproc_en.doc_lengths.values())

    # all tokens exist in vocab
    for dtok in tokens.values():
        assert all(t in vocab for t in dtok)

    # each word in vocab is used at least once
    for w in vocab:
        assert any(w in set(dtok) for dtok in tokens.values())


@preproc_test()
def test_tmpreproc_en_ngrams(tmpreproc_en):
    bigrams = tmpreproc_en.generate_ngrams(2).ngrams

    assert tmpreproc_en.ngrams_generated is True
    assert set(bigrams.keys()) == set(corpus_en.docs.keys())

    for dl, dtok in bigrams.items():
        assert isinstance(dtok, list)

        if dl == 'empty_doc':
            assert len(dtok) == 0
        else:
            assert len(dtok) > 0
            assert all([n == 2 for n in map(len, dtok)])

    # normal tokens are still unigrams
    for dtok in tmpreproc_en.tokens.values():
        assert isinstance(dtok, list)
        assert all([isinstance(t, str) for t in dtok])

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.join_ngrams()
    assert tmpreproc_en.ngrams_as_tokens is True
    assert tmpreproc_en.ngrams_generated is False   # is reset!
    assert tmpreproc_en.ngrams == {}

    # now tokens are bigrams
    for dtok in tmpreproc_en.tokens.values():
        assert isinstance(dtok, list)
        assert all([isinstance(t, str) for t in dtok])

        if len(dtok) > 0:
            for t in dtok:
                split_t = t.split(' ')
                assert len(split_t) == 2

    # fail when ngrams are used as tokens
    with pytest.raises(ValueError):
        tmpreproc_en.stem()
    with pytest.raises(ValueError):
        tmpreproc_en.lemmatize()
    with pytest.raises(ValueError):
        tmpreproc_en.expand_compound_tokens()
    with pytest.raises(ValueError):
        tmpreproc_en.pos_tag()


@preproc_test()
def test_tmpreproc_en_transform_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    # runs on workers because can be pickled:
    tokens_upper = tmpreproc_en.transform_tokens(str.upper).tokens

    for dl, dtok in tokens.items():
        dtok_ = tokens_upper[dl]
        assert len(dtok) == len(dtok_)
        assert all(t.upper() == t_ for t, t_ in zip(dtok, dtok_))


@preproc_test()
def test_tmpreproc_en_transform_tokens_lambda(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    # runs on main thread because cannot be pickled:
    tokens_upper = tmpreproc_en.transform_tokens(lambda x: x.upper()).tokens

    for dl, dtok in tokens.items():
        dtok_ = tokens_upper[dl]
        assert len(dtok) == len(dtok_)
        assert all([t.upper() == t_ for t, t_ in zip(dtok, dtok_)])


@preproc_test()
def test_tmpreproc_en_tokens_to_lowercase(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tokens_lower = tmpreproc_en.tokens_to_lowercase().tokens

    assert set(tokens.keys()) == set(tokens_lower.keys())

    for dl, dtok in tokens.items():
        dtok_ = tokens_lower[dl]
        assert len(dtok) == len(dtok_)
        assert all(t.lower() == t_ for t, t_ in zip(dtok, dtok_))


@preproc_test()
def test_tmpreproc_en_stem(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    stems = tmpreproc_en.stem().tokens

    assert set(tokens.keys()) == set(stems.keys())

    for dl, dtok in tokens.items():
        dtok_ = stems[dl]
        assert len(dtok) == len(dtok_)


@preproc_test()
def test_tmpreproc_en_pos_tag(tmpreproc_en):
    tmpreproc_en.pos_tag()
    tokens = tmpreproc_en.tokens
    tokens_with_pos_tags = tmpreproc_en.tokens_with_pos_tags

    assert set(tokens.keys()) == set(tokens_with_pos_tags.keys())

    for dl, dtok in tokens.items():
        tok_pos_df = tokens_with_pos_tags[dl]
        assert len(dtok) == tok_pos_df.shape[0]
        assert list(pd_dt_colnames(tok_pos_df)) == ['token', 'meta_pos']

        if USE_DT:
            tok_pos_df = tok_pos_df.to_pandas()

        assert np.array_equal(dtok, tok_pos_df.token)
        if dl != 'empty_doc':
            assert all(tok_pos_df.meta_pos.str.len() > 0)


@preproc_test()
def test_tmpreproc_en_lemmatize_fail_no_pos_tags(tmpreproc_en):
    with pytest.raises(ValueError):
        tmpreproc_en.lemmatize()


@preproc_test()
def test_tmpreproc_en_lemmatize(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    vocab = tmpreproc_en.vocabulary
    lemmata = tmpreproc_en.pos_tag().lemmatize().tokens

    assert set(tokens.keys()) == set(lemmata.keys())

    for dl, dtok in tokens.items():
        dtok_ = lemmata[dl]
        assert len(dtok) == len(dtok_)

    assert len(tmpreproc_en.vocabulary) < len(vocab)


@preproc_test()
def test_tmpreproc_en_expand_compound_tokens(tmpreproc_en):
    tmpreproc_en.clean_tokens()
    tokens = tmpreproc_en.tokens
    tokens_exp = tmpreproc_en.expand_compound_tokens().tokens

    assert set(tokens.keys()) == set(tokens_exp.keys())

    for dl, dtok in tokens.items():
        dtok_ = tokens_exp[dl]
        assert len(dtok) <= len(dtok_)


@preproc_test()
def test_tmpreproc_en_expand_compound_tokens_same(tmpreproc_en):
    tmpreproc_en.remove_special_chars_in_tokens().clean_tokens()
    tokens = tmpreproc_en.tokens
    tokens_exp = tmpreproc_en.expand_compound_tokens().tokens

    assert set(tokens.keys()) == set(tokens_exp.keys())

    for dl, dtok in tokens.items():
        dtok_ = tokens_exp[dl]
        assert all(t == t_ for t, t_ in zip(dtok, dtok_))


@preproc_test()
def test_tmpreproc_en_remove_special_chars_in_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tokens_ = tmpreproc_en.remove_special_chars_in_tokens().tokens

    assert set(tokens.keys()) == set(tokens_.keys())

    for dl, dtok in tokens.items():
        dtok_ = tokens_[dl]
        assert len(dtok) == len(dtok_)
        if len(dtok) > 0:
            assert np.all(np.char.str_len(dtok) >= np.char.str_len(dtok_))


@preproc_test()
def test_tmpreproc_en_clean_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    vocab = tmpreproc_en.vocabulary

    tmpreproc_en.clean_tokens()

    tokens_cleaned = tmpreproc_en.tokens
    vocab_cleaned = tmpreproc_en.vocabulary

    assert set(tokens.keys()) == set(tokens_cleaned.keys())

    assert len(vocab) > len(vocab_cleaned)

    for dl, dtok in tokens.items():
        dtok_ = tokens_cleaned[dl]
        assert len(dtok) >= len(dtok_)


@preproc_test()
def test_tmpreproc_en_clean_tokens_shorter(tmpreproc_en):
    min_len = 5
    tokens = tmpreproc_en.tokens
    cleaned = tmpreproc_en.clean_tokens(remove_punct=False,
                                        remove_stopwords=False,
                                        remove_empty=False,
                                        remove_shorter_than=min_len).tokens

    assert set(tokens.keys()) == set(cleaned.keys())

    for dl, dtok in tokens.items():
        dtok_ = cleaned[dl]
        assert len(dtok) >= len(dtok_)
        assert all(len(t) >= min_len for t in dtok_)


@preproc_test()
def test_tmpreproc_en_clean_tokens_longer(tmpreproc_en):
    max_len = 7
    tokens = tmpreproc_en.tokens
    cleaned = tmpreproc_en.clean_tokens(remove_punct=False,
                                        remove_stopwords=False,
                                        remove_empty=False,
                                        remove_longer_than=max_len).tokens

    assert set(tokens.keys()) == set(cleaned.keys())

    for dl, dtok in tokens.items():
        dtok_ = cleaned[dl]
        assert len(dtok) >= len(dtok_)
        assert all(len(t) <= max_len for t in dtok_)


@preproc_test()
def test_tmpreproc_en_clean_tokens_remove_numbers(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    cleaned = tmpreproc_en.clean_tokens(remove_numbers=True).tokens

    assert set(tokens.keys()) == set(cleaned.keys())

    for dl, dtok in tokens.items():
        dtok_ = cleaned[dl]
        assert len(dtok) >= len(dtok_)
        assert all(not t.isnumeric() for t in dtok_)


@preproc_test()
def test_tmpreproc_en_filter_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens('Moby')
    filtered = tmpreproc_en.tokens

    assert tmpreproc_en.vocabulary == ['Moby']
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dtok in tokens.items():
        dtok_ = filtered[dl]
        assert len(dtok_) <= len(dtok)

        if len(dtok_) > 0:
            assert set(dtok_) == {'Moby'}


@preproc_test()
def test_tmpreproc_en_filter_tokens_with_kwic(tmpreproc_en):
    # better tests are done with func API
    tokens = tmpreproc_en.tokens
    orig_vocab_size = tmpreproc_en.vocabulary_size
    tmpreproc_en.filter_tokens_with_kwic('Moby')
    filtered = tmpreproc_en.tokens

    assert tmpreproc_en.vocabulary_size < orig_vocab_size
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dtok in tokens.items():
        dtok_ = filtered[dl]
        assert len(dtok_) <= len(dtok)

        if len(dtok_) > 0:
            assert 'Moby' in dtok_


@preproc_test()
def test_tmpreproc_en_filter_tokens_by_meta(tmpreproc_en):
    tmpreproc_en.add_metadata_per_token('is_moby', {'Moby': True}, default=False)
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens(True, by_meta='is_moby')
    filtered = tmpreproc_en.tokens

    assert tmpreproc_en.vocabulary == ['Moby']
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dtok in tokens.items():
        dtok_ = filtered[dl]
        assert len(dtok_) <= len(dtok)

        if len(dtok_) > 0:
            assert set(dtok_) == {'Moby'}


@preproc_test()
def test_tmpreproc_en_filter_tokens_by_mask(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    orig_vocabulary = tmpreproc_en.vocabulary

    mask = {dl: [bool(i % 2) for i in range(len(dtok))] for dl, dtok in tokens.items()}
    tmpreproc_en.filter_tokens_by_mask(mask)
    filtered = tmpreproc_en.tokens

    assert len(orig_vocabulary) > len(tmpreproc_en.vocabulary)
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dtok in filtered.items():
        dmsk = mask[dl]
        assert sum(dmsk) == len(dtok)


@preproc_test()
def test_tmpreproc_en_filter_tokens_multiple(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens(['the', 'Moby'])
    filtered = tmpreproc_en.tokens

    assert set(tmpreproc_en.vocabulary) == {'the', 'Moby'}
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dtok in tokens.items():
        dtok_ = filtered[dl]
        assert len(dtok_) <= len(dtok)

        if len(dtok_) > 0:
            assert set(dtok_) <= {'the', 'Moby'}


@preproc_test()
def test_tmpreproc_en_filter_tokens_multiple_inverse(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    orig_vocab = set(tmpreproc_en.vocabulary)
    tmpreproc_en.filter_tokens(['the', 'Moby'], inverse=True)
    filtered = tmpreproc_en.tokens

    assert set(tmpreproc_en.vocabulary) == orig_vocab - {'the', 'Moby'}
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dtok in tokens.items():
        dtok_ = filtered[dl]
        assert len(dtok_) <= len(dtok)

        if len(dtok_) > 0:
            assert 'the' not in set(dtok_) and 'Moby' not in set(dtok_)


@preproc_test()
def test_tmpreproc_en_filter_tokens_inverse(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens('Moby', inverse=True)
    filtered = tmpreproc_en.tokens

    assert 'Moby' not in tmpreproc_en.vocabulary
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dtok in tokens.items():
        dtok_ = filtered[dl]
        assert len(dtok_) <= len(dtok)
        assert 'Moby' not in dtok_


@preproc_test()
def test_tmpreproc_en_filter_tokens_inverse_glob(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens('Mob*', inverse=True, match_type='glob')
    filtered = tmpreproc_en.tokens

    for w in tmpreproc_en.vocabulary:
        assert not w.startswith('Mob')

    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dtok in tokens.items():
        dtok_ = filtered[dl]
        assert len(dtok_) <= len(dtok)

        for t_ in dtok_:
            assert not t_.startswith('Mob')


@preproc_test()
def test_tmpreproc_en_filter_tokens_by_pattern(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens('^Mob.*', match_type='regex')
    filtered = tmpreproc_en.tokens

    for w in tmpreproc_en.vocabulary:
        assert w.startswith('Mob')

    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dtok in tokens.items():
        dtok_ = filtered[dl]
        assert len(dtok_) <= len(dtok)

        for t_ in dtok_:
            assert t_.startswith('Mob')


@preproc_test()
def test_tmpreproc_en_filter_documents(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_documents('Moby')
    filtered = tmpreproc_en.tokens

    assert set(filtered.keys()) == {'melville-moby_dick.txt'}
    assert set(filtered.keys()) == set(tmpreproc_en.doc_labels)
    assert 'Moby' in tmpreproc_en.vocabulary
    assert filtered['melville-moby_dick.txt'] == tokens['melville-moby_dick.txt']


@preproc_test()
def test_tmpreproc_en_filter_documents_by_meta(tmpreproc_en):
    tmpreproc_en.add_metadata_per_token('is_moby', {'Moby': True}, default=False)
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_documents(True, by_meta='is_moby')
    filtered = tmpreproc_en.tokens

    assert set(filtered.keys()) == {'melville-moby_dick.txt'}
    assert set(filtered.keys()) == set(tmpreproc_en.doc_labels)
    assert 'Moby' in tmpreproc_en.vocabulary
    assert filtered['melville-moby_dick.txt'] == tokens['melville-moby_dick.txt']


@preproc_test()
def test_tmpreproc_en_filter_documents_by_pattern(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_documents('^Mob.*', match_type='regex')
    filtered = tmpreproc_en.tokens

    assert 'melville-moby_dick.txt' in set(filtered.keys())
    assert set(filtered.keys()) == set(tmpreproc_en.doc_labels)
    assert 'Moby' in tmpreproc_en.vocabulary
    assert filtered['melville-moby_dick.txt'] == tokens['melville-moby_dick.txt']


@pytest.mark.parametrize(
    'testcase, name_pattern, match_type, ignore_case, inverse, use_drop',
    [
        (1, 'melville-moby_dick.txt', 'exact', False, False, False),
        (2, 'Melville-moby_dick.txt', 'exact', True, False, False),
        (3, 'melville-moby_dick.txt', 'exact', False, True, False),
        (4, '*.txt', 'glob', False, False, False),
        (5, '*', 'glob', False, False, False),
        (6, '*', 'glob', False, True, False),
        (7, r'\.txt$', 'regex', False, False, False),
        (8, 'empty_doc', 'exact', False, True, False),
        (9, 'empty_doc', 'exact', False, False, True),
        (10, {'empty_doc', 'melville-moby_dick.txt'}, 'exact', False, True, False),
        (11, {'empty_doc', 'melville-moby_dick.txt'}, 'exact', False, False, True),
    ]
)
@preproc_test()
def test_tmpreproc_en_filter_documents_by_name(tmpreproc_en, testcase, name_pattern, match_type, ignore_case, inverse,
                                               use_drop):
    orig_docs = set(tmpreproc_en.doc_labels)

    if use_drop:
        tmpreproc_en.remove_documents_by_name(name_patterns=name_pattern, match_type=match_type, ignore_case=ignore_case)
    else:
        tmpreproc_en.filter_documents_by_name(name_patterns=name_pattern, match_type=match_type,
                                              ignore_case=ignore_case, inverse=inverse)

    new_docs = set(tmpreproc_en.doc_labels)

    assert new_docs <= orig_docs   # subset test

    if testcase in (1, 2):
        assert new_docs == {'melville-moby_dick.txt'}
    elif testcase == 3:
        assert new_docs == orig_docs - {'melville-moby_dick.txt'}
    elif testcase in (4, 7, 8, 9):
        assert new_docs == orig_docs - {'empty_doc'}
    elif testcase == 5:
        assert new_docs == orig_docs
    elif testcase == 6:
        assert new_docs == set()
    elif testcase in (10, 11):
        assert new_docs == orig_docs - {'melville-moby_dick.txt', 'empty_doc'}
    else:
        raise ValueError('unknown `testcase`')


@preproc_test()
def test_tmpreproc_en_filter_for_pos(tmpreproc_en):
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos('N').tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert tok_pos_.shape[0] <= tok_pos.shape[0]
        if USE_DT:
            meta_pos_ = np.array(tok_pos_.to_dict()['meta_pos'], dtype=np.str)
        else:
            meta_pos_ = np.array(tok_pos_['meta_pos'].tolist(), dtype=np.str)
        assert np.all(np.char.startswith(meta_pos_, 'N'))


@preproc_test()
def test_tmpreproc_en_filter_for_pos_none(tmpreproc_en):
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(None).tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert tok_pos_.shape[0] <= tok_pos.shape[0]
        if USE_DT:
            meta_pos_ = np.array(tok_pos_.to_dict()['meta_pos'], dtype=np.str)
        else:
            meta_pos_ = np.array(tok_pos_['meta_pos'].tolist(), dtype=np.str)
        simpl_postags = [simplified_pos(pos) for pos in meta_pos_]
        assert all(pos is None for pos in simpl_postags)


@preproc_test()
def test_tmpreproc_en_filter_for_multiple_pos1(tmpreproc_en):
    req_tags = ['N', 'V']
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(req_tags).tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert tok_pos_.shape[0] <= tok_pos.shape[0]
        if USE_DT:
            meta_pos_ = np.array(tok_pos_.to_dict()['meta_pos'], dtype=np.str)
        else:
            meta_pos_ = np.array(tok_pos_['meta_pos'].tolist(), dtype=np.str)
        simpl_postags = [simplified_pos(pos) for pos in meta_pos_]
        assert all(pos in req_tags for pos in simpl_postags)


@preproc_test()
def test_tmpreproc_en_filter_for_multiple_pos2(tmpreproc_en):
    req_tags = {'N', 'V', None}
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(req_tags).tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert tok_pos_.shape[0] <= tok_pos.shape[0]
        if USE_DT:
            meta_pos_ = np.array(tok_pos_.to_dict()['meta_pos'], dtype=np.str)
        else:
            meta_pos_ = np.array(tok_pos_['meta_pos'].tolist(), dtype=np.str)
        simpl_postags = [simplified_pos(pos) for pos in meta_pos_]
        assert all(pos in req_tags for pos in simpl_postags)


@preproc_test()
def test_tmpreproc_en_filter_for_pos_and_2nd_pass(tmpreproc_en):
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(['N', 'V']).filter_for_pos('V').tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert tok_pos_.shape[0] <= tok_pos.shape[0]
        if USE_DT:
            meta_pos_ = np.array(tok_pos_.to_dict()['meta_pos'], dtype=np.str)
        else:
            meta_pos_ = np.array(tok_pos_['meta_pos'].tolist(), dtype=np.str)
        assert np.all(np.char.startswith(meta_pos_, 'V'))


@pytest.mark.parametrize(
    'context_size, highlight_keyword, search_token',
    [(2, None, 'the'),    # default
     (1, None, 'the'),
     ((0, 2), None, 'the'),
     ((2, 0), None, 'the'),
     (2, '*', 'Moby')]
)
@preproc_test(make_checks=False)
def test_tmpreproc_en_get_kwic(tmpreproc_en, context_size, highlight_keyword, search_token):
    doc_labels = tmpreproc_en.doc_labels
    vocab = tmpreproc_en.vocabulary
    dtm = tmpreproc_en.dtm.tocsr()
    max_win_size = sum(context_size) + 1 if type(context_size) == tuple else 2*context_size + 1

    res = tmpreproc_en.get_kwic('foobarthisworddoesnotexist')
    assert type(res) == dict
    assert set(res.keys()) == set(doc_labels)
    assert all([n == 0 for n in map(len, res.values())])

    assert tmpreproc_en.get_kwic('foobarthisworddoesnotexist', non_empty=True) == dict()

    if highlight_keyword is not None:
        highlighted_token = highlight_keyword + search_token + highlight_keyword
    else:
        highlighted_token = search_token

    res = tmpreproc_en.get_kwic(search_token, context_size=context_size, highlight_keyword=highlight_keyword)
    ind_search_tok = vocab.index(search_token)

    for dl, windows in res.items():
        n_search_tok = dtm.getrow(doc_labels.index(dl)).getcol(ind_search_tok).A[0][0]

        assert dl in doc_labels
        assert len(windows) == n_search_tok
        assert all([0 < len(win) <= max_win_size for win in windows])    # default context size is (2, 2) -> max. window size is 5
        assert all([highlighted_token in set(win) for win in windows])


@pytest.mark.parametrize(
    'highlight_keyword, search_token',
    [(None, 'the'),    # default
     ('*', 'Moby')]
)
@preproc_test(make_checks=False)
def test_tmpreproc_en_get_kwic_glued(tmpreproc_en, highlight_keyword, search_token):
    doc_labels = tmpreproc_en.doc_labels
    vocab = tmpreproc_en.vocabulary
    dtm = tmpreproc_en.dtm.tocsr()

    if highlight_keyword is not None:
        highlighted_token = highlight_keyword + search_token + highlight_keyword
    else:
        highlighted_token = search_token

    res = tmpreproc_en.get_kwic(search_token, glue=' ', highlight_keyword=highlight_keyword)
    ind_search_tok = vocab.index(search_token)

    for dl, windows in res.items():
        n_search_tok = dtm.getrow(doc_labels.index(dl)).getcol(ind_search_tok).A[0][0]

        assert dl in doc_labels
        assert len(windows) == n_search_tok
        assert all([type(excerpt) == str and ' ' in excerpt and highlighted_token in excerpt for excerpt in windows])


@pytest.mark.parametrize(
    'search_token, with_metadata, as_datatable',
    [('the', True, False),    # default
     ('Moby', False, True),
     ('the', True, True)]
)
@preproc_test(make_checks=False)
def test_tmpreproc_en_get_kwic_metadata_datatable(tmpreproc_en, search_token, with_metadata, as_datatable):
    tmpreproc_en.pos_tag()

    doc_labels = tmpreproc_en.doc_labels
    vocab = tmpreproc_en.vocabulary
    dtm = tmpreproc_en.dtm.tocsr()

    res = tmpreproc_en.get_kwic(search_token, with_metadata=with_metadata, as_datatable=as_datatable)
    ind_search_tok = vocab.index(search_token)

    if as_datatable:
        assert isinstance(res, FRAME_TYPE)
        meta_cols = ['meta_pos'] if with_metadata else []

        assert pd_dt_colnames(res) == ['doc', 'context', 'position', 'token'] + meta_cols

        if USE_DT:
            res = res.to_pandas()   # so far, datatable doesn't seem to provide iteration through groups

        for (dl, context), windf in res.groupby(['doc', 'context']):
            assert dl in doc_labels
            dtm_row = dtm.getrow(doc_labels.index(dl))
            n_search_tok = dtm_row.getcol(ind_search_tok).A[0][0]
            assert 0 <= context < n_search_tok
            assert windf.reset_index().position.min() >= 0
            assert windf.reset_index().position.min() < np.sum(dtm_row.A)
            assert 0 < len(windf) <= 5     # default context size is (2, 2) -> max. window size is 5
    else:
        assert type(res) == dict
        for dl, windows in res.items():
            n_search_tok = dtm.getrow(doc_labels.index(dl)).getcol(ind_search_tok).A[0][0]

            assert dl in doc_labels
            assert len(windows) == n_search_tok

            for win in windows:
                assert type(win) == dict
                assert set(win.keys()) == {'token', 'meta_pos'}
                winsize = len(win['token'])
                assert len(win['meta_pos']) == winsize


@pytest.mark.parametrize(
    'search_token',
    ['the', 'Moby', 'the']
)
@preproc_test(make_checks=False)
def test_tmpreproc_en_get_kwic_table(tmpreproc_en, search_token):
    tmpreproc_en.pos_tag()

    doc_labels = tmpreproc_en.doc_labels
    vocab = tmpreproc_en.vocabulary
    dtm = tmpreproc_en.dtm.tocsr()

    res = tmpreproc_en.get_kwic_table(search_token)
    ind_search_tok = vocab.index(search_token)

    assert isinstance(res, FRAME_TYPE)
    assert pd_dt_colnames(res)[:2] == ['doc', 'context']

    if USE_DT:
        res = res.to_pandas().copy()  # so far, datatable doesn't seem to provide iteration through groups
    else:
        res = res.copy()

    for (dl, context), windf in res.groupby(['doc', 'context']):
        assert dl in doc_labels
        dtm_row = dtm.getrow(doc_labels.index(dl))
        n_search_tok = dtm_row.getcol(ind_search_tok).A[0][0]
        assert 0 <= context < n_search_tok
        assert len(windf) == 1
        kwic_str = windf.kwic.iloc[0]
        assert isinstance(kwic_str, str)
        assert len(kwic_str) > 0
        assert kwic_str.count('*') == 2


@pytest.mark.parametrize(
    'patterns, glue, match_type',
    [
        ('fail', '_', 'exact'),
        (['fail'], '_', 'exact'),
        (['with', 'all'], '_', 'exact'),
        (['with', 'all', 'the'], '_', 'exact'),
        (['Moby', '*'], '//', 'glob'),
    ]
)
@preproc_test(make_checks=False)
def test_tmpreproc_en_glue_tokens(tmpreproc_en, patterns, glue, match_type):
    if not isinstance(patterns, list) or len(patterns) < 2:
        with pytest.raises(ValueError):
            tmpreproc_en.glue_tokens(patterns, glue=glue, match_type=match_type)
    else:
        glued = tmpreproc_en.glue_tokens(patterns, glue=glue, match_type=match_type)
        assert isinstance(glued, set)
        assert len(glued) > 0

        for t in glued:
            assert t.count(glue) == len(patterns) - 1

        if match_type == 'exact':
            assert glued == {glue.join(patterns)}

        for g in glued:
            assert any(g in dtok for dtok in tmpreproc_en.tokens.values())


@pytest.mark.parametrize(
    'patterns, glue, match_type',
    [
        (['with', 'all'], '_', 'exact'),
        (['with', 'all', 'the'], '_', 'exact'),
        (['Moby', '*'], '//', 'glob'),
    ]
)
@preproc_test(make_checks=False)
def test_tmpreproc_en_pos_tagged_glue_tokens(tmpreproc_en, patterns, glue, match_type):
    tmpreproc_en.pos_tag()

    glued = tmpreproc_en.glue_tokens(patterns, glue=glue, match_type=match_type)
    assert isinstance(glued, set)
    assert len(glued) > 0

    for df in tmpreproc_en.tokens_with_metadata.values():
        if USE_DT:
            df = df.to_pandas()

        assert df.loc[df.token.isin(glued), 'meta_pos'].isnull().all()
        assert df.loc[~df.token.isin(glued), 'meta_pos'].notnull().all()


@preproc_test()
def test_tmpreproc_en_get_dtm(tmpreproc_en):
    dtm = tmpreproc_en.get_dtm()
    dtm_prop = tmpreproc_en.dtm

    assert dtm.ndim == 2
    assert len(tmpreproc_en.doc_labels) == dtm.shape[0]
    assert len(tmpreproc_en.vocabulary) == dtm.shape[1]
    assert not (dtm != dtm_prop).toarray().any()


@preproc_test(make_checks=False)
def test_tmpreproc_en_get_dtm_calc_tfidf(tmpreproc_en):
    tmpreproc_en.remove_documents_by_name('empty_doc')
    dtm = tmpreproc_en.dtm

    tfidf_mat = tfidf(dtm)
    assert tfidf_mat.ndim == 2
    assert tfidf_mat.shape == dtm.shape
    assert np.issubdtype(tfidf_mat.dtype, np.float)
    assert isinstance(tfidf_mat, sparse.spmatrix)
    assert np.all(tfidf_mat.A >= -1e-10)


@preproc_test(make_checks=False)
def test_tmpreproc_en_n_tokens(tmpreproc_en):
    assert tmpreproc_en.n_tokens == sum(map(len, tmpreproc_en.tokens.values()))


@preproc_test(make_checks=False)
def test_tmpreproc_en_vocabulary_counts(tmpreproc_en):
    counts = tmpreproc_en.vocabulary_counts
    assert isinstance(counts, dict)
    assert len(counts) > 0
    assert set(counts.keys()) == set(tmpreproc_en.vocabulary)
    assert 'Moby' in counts.keys()
    assert all(0 < n <= tmpreproc_en.n_tokens for n in counts.values())
    assert tmpreproc_en.n_tokens == sum(counts.values())


@preproc_test(make_checks=False)
def test_tmpreproc_en_vocabulary_doc_frequency(tmpreproc_en):
    vocab = tmpreproc_en.vocabulary
    n_docs = tmpreproc_en.n_docs

    doc_freqs = tmpreproc_en.vocabulary_rel_doc_frequency
    abs_doc_freqs = tmpreproc_en.vocabulary_abs_doc_frequency
    assert len(doc_freqs) == len(abs_doc_freqs) == len(vocab)

    for t, f in doc_freqs.items():
        assert 0 < f <= 1
        n = abs_doc_freqs[t]
        assert 0 < n <= n_docs
        assert abs(f - n/n_docs) < 1e-6
        assert t in vocab


@preproc_test()
def test_tmpreproc_en_remove_common_or_uncommon_tokens(tmpreproc_en):
    tmpreproc_en.tokens_to_lowercase()
    vocab_orig = tmpreproc_en.vocabulary

    tmpreproc_en.remove_uncommon_tokens(0.0)
    assert len(tmpreproc_en.vocabulary) == len(vocab_orig)

    doc_freqs = tmpreproc_en.vocabulary_rel_doc_frequency

    max_df = max(doc_freqs.values()) - 0.1
    tmpreproc_en.remove_common_tokens(max_df)
    assert len(tmpreproc_en.vocabulary) < len(vocab_orig)

    doc_freqs = tmpreproc_en.vocabulary_rel_doc_frequency
    assert all(f < max_df for f in doc_freqs.values())

    min_df = min(doc_freqs.values()) + 0.1
    tmpreproc_en.remove_uncommon_tokens(min_df)
    assert len(tmpreproc_en.vocabulary) < len(vocab_orig)

    doc_freqs = tmpreproc_en.vocabulary_rel_doc_frequency
    assert all(f > min_df for f in doc_freqs.values())


@preproc_test(make_checks=False)
def test_tmpreproc_en_remove_common_or_uncommon_tokens_absolute(tmpreproc_en):
    tmpreproc_en.tokens_to_lowercase()
    vocab_orig = tmpreproc_en.vocabulary

    tmpreproc_en.remove_common_tokens(6, absolute=True)
    assert len(tmpreproc_en.vocabulary) < len(vocab_orig)

    doc_freqs = tmpreproc_en.vocabulary_abs_doc_frequency
    assert all(n < 6 for n in doc_freqs.values())

    tmpreproc_en.remove_uncommon_tokens(1, absolute=True)
    assert len(tmpreproc_en.vocabulary) < len(vocab_orig)

    doc_freqs = tmpreproc_en.vocabulary_abs_doc_frequency
    assert all(n > 1 for n in doc_freqs.values())

    tmpreproc_en.remove_common_tokens(1, absolute=True)
    assert len(tmpreproc_en.vocabulary) == 0
    assert all(len(t) == 0 for t in tmpreproc_en.tokens.values())


@preproc_test()
def test_tmpreproc_en_apply_custom_filter(tmpreproc_en):
    tmpreproc_en.tokens_to_lowercase()
    vocab_orig = tmpreproc_en.vocabulary
    docs_orig = tmpreproc_en.doc_labels

    vocab_max_strlen = np.char.str_len(vocab_orig).max()

    def strip_words_with_max_len(tokens):
        new_tokens = {}
        for dl, dtok in tokens.items():
            dtok = np.array(dtok, dtype=str)
            dt_len = np.char.str_len(dtok)
            new_tokens[dl] = dtok[dt_len < vocab_max_strlen].tolist()

        return new_tokens

    # apply for first time
    tmpreproc_en.apply_custom_filter(strip_words_with_max_len)

    new_vocab = tmpreproc_en.vocabulary

    assert new_vocab != vocab_orig
    assert max(map(len, new_vocab)) < vocab_max_strlen

    assert tmpreproc_en.doc_labels == docs_orig

    # applying the second time shouldn't change anything:
    tmpreproc_en.apply_custom_filter(strip_words_with_max_len)
    assert new_vocab == tmpreproc_en.vocabulary


@preproc_test()
def test_tmpreproc_en_pipeline(tmpreproc_en):
    orig_docs = tmpreproc_en.doc_labels
    orig_vocab = tmpreproc_en.vocabulary
    orig_tokensdf = tmpreproc_en.tokens_datatable
    assert {'doc', 'position', 'token'} == set(pd_dt_colnames(orig_tokensdf))

    # part 1
    tmpreproc_en.pos_tag().lemmatize().tokens_to_lowercase().clean_tokens()

    assert orig_docs == tmpreproc_en.doc_labels
    assert set(tmpreproc_en.tokens.keys()) == set(orig_docs)
    new_vocab = tmpreproc_en.vocabulary
    assert len(orig_vocab) > len(new_vocab)

    tokensdf_part1 = tmpreproc_en.tokens_datatable
    assert {'doc', 'position', 'token', 'meta_pos'} == set(pd_dt_colnames(tokensdf_part1))
    assert tokensdf_part1.shape[0] < orig_tokensdf.shape[0]   # because of "clean_tokens"

    dtm = tmpreproc_en.dtm
    assert dtm.ndim == 2
    assert dtm.shape[0] == tmpreproc_en.n_docs == len(tmpreproc_en.doc_labels)
    assert dtm.shape[1] == len(new_vocab)

    # part 2
    tmpreproc_en.filter_for_pos('N')

    assert len(new_vocab) > len(tmpreproc_en.vocabulary)

    tokensdf_part2 = tmpreproc_en.tokens_datatable
    assert {'doc', 'position', 'token', 'meta_pos'} == set(pd_dt_colnames(tokensdf_part2))
    assert tokensdf_part2.shape[0] < tokensdf_part1.shape[0]   # because of "filter_for_pos"

    dtm = tmpreproc_en.dtm
    assert dtm.ndim == 2
    assert dtm.shape[0] == tmpreproc_en.n_docs == len(tmpreproc_en.doc_labels)
    assert dtm.shape[1] == len(tmpreproc_en.vocabulary)

    new_vocab2 = tmpreproc_en.vocabulary

    # part 3
    tmpreproc_en.filter_documents('moby')  # lower case already

    assert len(new_vocab2) > len(tmpreproc_en.vocabulary)

    tokensdf_part3 = tmpreproc_en.tokens_datatable
    assert {'doc', 'position', 'token', 'meta_pos'} == set(pd_dt_colnames(tokensdf_part3))
    assert tokensdf_part3.shape[0] < tokensdf_part2.shape[0]   # because of "filter_documents"

    dtm = tmpreproc_en.dtm
    assert dtm.ndim == 2
    assert dtm.shape[0] == tmpreproc_en.n_docs == len(tmpreproc_en.doc_labels) == 1
    assert dtm.shape[1] == len(tmpreproc_en.vocabulary)


#%% Tests with German corpus (only methods dependent on language are tested)


@preproc_test(lang='de')
def test_tmpreproc_de_init(tmpreproc_de):
    assert tmpreproc_de.doc_labels == corpus_de.doc_labels
    assert len(tmpreproc_de.doc_labels) == N_DOCS_DE
    assert tmpreproc_de.language == 'german'


@preproc_test(lang='de')
def test_tmpreproc_de_tokenize(tmpreproc_de):
    tokens = tmpreproc_de.tokens
    assert set(tokens.keys()) == set(tmpreproc_de.doc_labels)

    for dtok in tokens.values():
        assert isinstance(dtok, list)
        assert len(dtok) > 0

        # make sure that not all tokens only consist of a single character:
        assert np.sum(np.char.str_len(dtok) > 1) > 1


@preproc_test(lang='de')
def test_tmpreproc_de_stem(tmpreproc_de):
    tokens = tmpreproc_de.tokens
    stems = tmpreproc_de.stem().tokens

    assert set(tokens.keys()) == set(stems.keys())

    for dl, dtok in tokens.items():
        dtok_ = stems[dl]
        assert len(dtok) == len(dtok_)


@preproc_test(lang='de')
def test_tmpreproc_de_pos_tag(tmpreproc_de):
    tmpreproc_de.pos_tag()
    tokens = tmpreproc_de.tokens
    tokens_with_pos_tags = tmpreproc_de.tokens_with_pos_tags

    assert set(tokens.keys()) == set(tokens_with_pos_tags.keys())

    for dl, dtok in tokens.items():
        tok_pos_df = tokens_with_pos_tags[dl]
        assert len(dtok) == tok_pos_df.shape[0]
        assert list(pd_dt_colnames(tok_pos_df)) == ['token', 'meta_pos']

        if USE_DT:
            tok_pos_df = tok_pos_df.to_pandas()

        assert np.array_equal(dtok, tok_pos_df.token)
        assert all(tok_pos_df.meta_pos.str.len() > 0)


@preproc_test(lang='de')
def test_tmpreproc_de_lemmatize_fail_no_pos_tags(tmpreproc_de):
    with pytest.raises(ValueError):
        tmpreproc_de.lemmatize()


@preproc_test(lang='de')
def test_tmpreproc_de_lemmatize(tmpreproc_de):
    tokens = tmpreproc_de.tokens
    vocab = tmpreproc_de.vocabulary
    lemmata = tmpreproc_de.pos_tag().lemmatize().tokens

    assert set(tokens.keys()) == set(lemmata.keys())

    for dl, dtok in tokens.items():
        dtok_ = lemmata[dl]
        assert len(dtok) == len(dtok_)

    assert len(tmpreproc_de.vocabulary) < len(vocab)
