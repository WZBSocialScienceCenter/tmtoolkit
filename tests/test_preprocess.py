from random import sample
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import nltk
import pytest

from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.corpus import Corpus
from tmtoolkit.utils import simplified_pos

TMPREPROC_TEMP_STATE_FILE = '/tmp/tmpreproc_tests_state.pickle'


#
# TMPreproc method tests
#

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
    pre_state['tokens_dataframe'] = preproc.tokens_dataframe
    pre_state['dtm'] = preproc.dtm

    if preproc.pos_tagged:
        pre_state['tokens_with_pos_tags'] = preproc.tokens_with_pos_tags
    if preproc.ngrams_generated:
        pre_state['ngrams'] = preproc.ngrams

    # save and then load the same state
    for _ in range(repeat):
        if recreate_from_state:
            preproc.save_state(TMPREPROC_TEMP_STATE_FILE)
            preproc = TMPreproc.from_state(TMPREPROC_TEMP_STATE_FILE)
        else:
            preproc.save_state(TMPREPROC_TEMP_STATE_FILE).load_state(TMPREPROC_TEMP_STATE_FILE)

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
    assert _dataframes_equal(pre_state['tokens_dataframe'], preproc.tokens_dataframe)

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
    assert _dataframes_equal(preproc_a.tokens_dataframe, preproc_b.tokens_dataframe)

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


#
# Tests with empty corpus
#

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

#
# Tests with English corpus
#


def test_tmpreproc_en_init(tmpreproc_en):
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    with pytest.raises(ValueError):    # because not POS tagged
        assert tmpreproc_en.tokens_with_pos_tags

    tmpreproc_en.ngrams == {}
    assert tmpreproc_en.ngrams_generated is False

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_add_stopwords(tmpreproc_en):
    sw = set(tmpreproc_en.stopwords)
    sw_ = set(tmpreproc_en.add_stopwords(['foobar']).stopwords)
    assert sw_ == sw | {'foobar'}

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_add_punctuation(tmpreproc_en):
    pct = set(tmpreproc_en.punctuation)
    pct_ = set(tmpreproc_en.add_punctuation(['X']).punctuation)
    assert pct_ == pct | {'X'}

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_add_special_chars(tmpreproc_en):
    sc = set(tmpreproc_en.special_chars)
    sc_ = set(tmpreproc_en.add_special_chars(['X']).special_chars)
    assert sc_ == sc | {'X'}

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


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
        assert 'meta_importance' in df.columns

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
        assert 'meta_importance' in df.columns
        assert 'meta_foobar' in df.columns

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
        assert 'meta_importance' in df.columns
        assert 'meta_foobar' not in df.columns

    tmpreproc_en.remove_metadata('importance')

    assert 'importance' not in tmpreproc_en.get_available_metadata_keys()

    for dl, df in tmpreproc_en.tokens_with_metadata.items():
        assert 'meta_importance' not in df.columns
        assert 'meta_foobar' not in df.columns

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


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
        assert 'meta_random_foo' in df.columns
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
        assert 'meta_random_foo' not in df.columns

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    # does not exist anymore:
    with pytest.raises(ValueError):
        tmpreproc_en.remove_metadata('random_foo')

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_save_load_state_several_times(tmpreproc_en):
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en, 5)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_save_load_state_recreate_from_state(tmpreproc_en):
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en, recreate_from_state=True)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_create_from_tokens(tmpreproc_en):
    preproc2 = TMPreproc.from_tokens(tmpreproc_en.tokens)

    assert set(tmpreproc_en.tokens.keys()) == set(preproc2.tokens.keys())
    assert all(preproc2.tokens[k] == tmpreproc_en.tokens[k]
               for k in tmpreproc_en.tokens.keys())

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_TMPreproc_copies(preproc2, preproc2.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_load_tokens(tmpreproc_en):
    # two modification: remove word "Moby" and remove a document
    tokens = {}
    removed_doc = None
    for i, (dl, dt) in enumerate(tmpreproc_en.tokens.items()):
        if i > 0:
            tokens[dl] = [t for t in dt if t != 'Moby']
        else:
            removed_doc = dl

    assert removed_doc is not None

    assert removed_doc in tmpreproc_en.doc_labels
    assert 'Moby' in tmpreproc_en.vocabulary

    tmpreproc_en.load_tokens(tokens)

    assert removed_doc not in tmpreproc_en.doc_labels
    assert 'Moby' not in tmpreproc_en.vocabulary

    tmpreproc_en.shutdown_workers()


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
        assert set(doc.columns) == {'token', 'meta_importance'}

        if i > 0:
            n_unimp += sum(doc.meta_importance != 'unimportant')
            tokens[dl] = doc.loc[doc.meta_importance != 'unimportant', :].copy()
        else:
            removed_doc = dl

    assert n_unimp > 0
    assert removed_doc is not None
    assert removed_doc in tmpreproc_en.doc_labels

    tmpreproc_en.load_tokens(tokens)

    assert removed_doc not in tmpreproc_en.doc_labels

    for i, (dl, doc) in enumerate(tmpreproc_en.tokens_with_metadata.items()):
        assert set(doc.columns) == {'token', 'meta_importance'}
        assert sum(doc.meta_importance == 'unimportant') == 0

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_load_tokens_dataframe(tmpreproc_en):
    tokensdf = tmpreproc_en.tokens_dataframe
    tokensdf = tokensdf.loc[tokensdf.token != 'Moby', :]

    assert 'Moby' in tmpreproc_en.vocabulary

    tmpreproc_en.load_tokens_dataframe(tokensdf)

    assert 'Moby' not in tmpreproc_en.vocabulary

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_create_from_tokens_dataframe(tmpreproc_en):
    preproc2 = TMPreproc.from_tokens_dataframe(tmpreproc_en.tokens_dataframe)

    assert _dataframes_equal(preproc2.tokens_dataframe, tmpreproc_en.tokens_dataframe)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_tokens_property(tmpreproc_en):
    tokens_all = tmpreproc_en.tokens
    assert set(tokens_all.keys()) == set(corpus_en.docs.keys())

    for dt in tokens_all.values():
        assert isinstance(dt, list)
        if len(dt) > 0:
            # make sure that not all tokens only consist of a single character:
            assert np.sum(np.char.str_len(dt) > 1) > 1

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_get_tokens_and_tokens_with_metadata_property(tmpreproc_en):
    tokens_from_prop = tmpreproc_en.tokens
    tokens_w_meta = tmpreproc_en.tokens_with_metadata
    assert set(tokens_w_meta.keys()) == set(corpus_en.docs.keys())

    tokens_w_meta_from_fn = tmpreproc_en.get_tokens(with_metadata=True, as_data_frames=True)

    for dl, df in tokens_w_meta.items():
        assert _dataframes_equal(df, tokens_w_meta_from_fn[dl])

        assert list(df.columns) == ['token']
        assert list(df.token) == list(tokens_from_prop[dl])

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_get_tokens_non_empty(tmpreproc_en):
    tokens = tmpreproc_en.get_tokens(non_empty=True)
    assert set(tokens.keys()) == set(corpus_en.docs.keys()) - {'empty_doc'}

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_doc_lengths(tmpreproc_en):
    doc_lengths = tmpreproc_en.doc_lengths
    assert set(doc_lengths.keys()) == set(corpus_en.docs.keys())

    for dl, dt in tmpreproc_en.tokens.items():
        assert doc_lengths[dl] == len(dt)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_tokens_dataframe(tmpreproc_en):
    doc_lengths = tmpreproc_en.doc_lengths

    df = tmpreproc_en.tokens_dataframe
    assert list(df.columns) == ['token']
    assert len(df.token) == sum(doc_lengths.values())

    ind0 = df.index.get_level_values(0)
    labels, lens = zip(*sorted(doc_lengths.items(), key=lambda x: x[0]))
    assert np.array_equal(ind0.to_numpy(), np.repeat(labels, lens))

    ind1 = df.index.get_level_values(1)
    expected_indices = []
    for n in lens:
        expected_indices.append(np.arange(n))

    assert np.array_equal(ind1.to_numpy(), np.concatenate(expected_indices))

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_vocabulary(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    vocab = tmpreproc_en.vocabulary

    assert isinstance(vocab, list)
    assert len(vocab) <= sum(tmpreproc_en.doc_lengths.values())

    # all tokens exist in vocab
    for dt in tokens.values():
        assert all(t in vocab for t in dt)

    # each word in vocab is used at least once
    for w in vocab:
        assert any(w in np.unique(dt) for dt in tokens.values())

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_ngrams(tmpreproc_en):
    bigrams = tmpreproc_en.generate_ngrams(2).ngrams

    assert tmpreproc_en.ngrams_generated is True
    assert set(bigrams.keys()) == set(corpus_en.docs.keys())

    for dl, dt in bigrams.items():
        assert isinstance(dt, list)

        if dl == 'empty_doc':
            assert len(dt) == 0
        else:
            assert len(dt) > 0
            assert all([n == 2 for n in map(len, dt)])

    # normal tokens are still unigrams
    for dt in tmpreproc_en.tokens.values():
        assert isinstance(dt, list)
        assert all([isinstance(t, str) for t in dt])

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.use_joined_ngrams_as_tokens()
    assert tmpreproc_en.ngrams_as_tokens is True
    assert tmpreproc_en.ngrams_generated is False   # is reset!
    assert tmpreproc_en.ngrams == {}

    # now tokens are bigrams
    for dt in tmpreproc_en.tokens.values():
        assert isinstance(dt, list)
        assert all([isinstance(t, str) for t in dt])

        if len(dt) > 0:
            for t in dt:
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

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_transform_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    # runs on workers because can be pickled:
    tokens_upper = tmpreproc_en.transform_tokens(str.upper).tokens

    for dl, dt in tokens.items():
        dt_ = tokens_upper[dl]
        assert len(dt) == len(dt_)
        assert all(t.upper() == t_ for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_transform_tokens_lambda(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    # runs on main thread because cannot be pickled:
    tokens_upper = tmpreproc_en.transform_tokens(lambda x: x.upper()).tokens

    for dl, dt in tokens.items():
        dt_ = tokens_upper[dl]
        assert len(dt) == len(dt_)
        assert all([t.upper() == t_ for t, t_ in zip(dt, dt_)])

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_tokens_to_lowercase(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tokens_lower = tmpreproc_en.tokens_to_lowercase().tokens

    assert set(tokens.keys()) == set(tokens_lower.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_lower[dl]
        assert len(dt) == len(dt_)
        assert all(t.lower() == t_ for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_stem(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    stems = tmpreproc_en.stem().tokens

    assert set(tokens.keys()) == set(stems.keys())

    for dl, dt in tokens.items():
        dt_ = stems[dl]
        assert len(dt) == len(dt_)

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_pos_tag(tmpreproc_en):
    tmpreproc_en.pos_tag()
    tokens = tmpreproc_en.tokens
    tokens_with_pos_tags = tmpreproc_en.tokens_with_pos_tags

    assert set(tokens.keys()) == set(tokens_with_pos_tags.keys())

    for dl, dt in tokens.items():
        tok_pos_df = tokens_with_pos_tags[dl]
        assert len(dt) == len(tok_pos_df)
        assert list(tok_pos_df.columns) == ['token', 'meta_pos']
        assert np.array_equal(dt, tok_pos_df.token)
        assert all(tok_pos_df.meta_pos.str.len() > 0)

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_lemmatize_fail_no_pos_tags(tmpreproc_en):
    with pytest.raises(ValueError):
        tmpreproc_en.lemmatize()


def test_tmpreproc_en_lemmatize(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    vocab = tmpreproc_en.vocabulary
    lemmata = tmpreproc_en.pos_tag().lemmatize().tokens

    assert set(tokens.keys()) == set(lemmata.keys())

    for dl, dt in tokens.items():
        dt_ = lemmata[dl]
        assert len(dt) == len(dt_)

    assert len(tmpreproc_en.vocabulary) < len(vocab)

    _check_save_load_state(tmpreproc_en)
    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_expand_compound_tokens(tmpreproc_en):
    tmpreproc_en.clean_tokens()
    tokens = tmpreproc_en.tokens
    tokens_exp = tmpreproc_en.expand_compound_tokens().tokens

    assert set(tokens.keys()) == set(tokens_exp.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_exp[dl]
        assert len(dt) <= len(dt_)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_expand_compound_tokens_same(tmpreproc_en):
    tmpreproc_en.remove_special_chars_in_tokens().clean_tokens()
    tokens = tmpreproc_en.tokens
    tokens_exp = tmpreproc_en.expand_compound_tokens().tokens

    assert set(tokens.keys()) == set(tokens_exp.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_exp[dl]
        assert all(t == t_ for t, t_ in zip(dt, dt_))

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_remove_special_chars_in_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tokens_ = tmpreproc_en.remove_special_chars_in_tokens().tokens

    assert set(tokens.keys()) == set(tokens_.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_[dl]
        assert len(dt) == len(dt_)
        if len(dt) > 0:
            assert np.all(np.char.str_len(dt) >= np.char.str_len(dt_))

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_clean_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    vocab = tmpreproc_en.vocabulary

    tmpreproc_en.clean_tokens()

    tokens_cleaned = tmpreproc_en.tokens
    vocab_cleaned = tmpreproc_en.vocabulary

    assert set(tokens.keys()) == set(tokens_cleaned.keys())

    assert len(vocab) > len(vocab_cleaned)

    for dl, dt in tokens.items():
        dt_ = tokens_cleaned[dl]
        assert len(dt) >= len(dt_)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_clean_tokens_shorter(tmpreproc_en):
    min_len = 5
    tokens = tmpreproc_en.tokens
    cleaned = tmpreproc_en.clean_tokens(remove_punct=False,
                                        remove_stopwords=False,
                                        remove_empty=False,
                                        remove_shorter_than=min_len).tokens

    assert set(tokens.keys()) == set(cleaned.keys())

    for dl, dt in tokens.items():
        dt_ = cleaned[dl]
        assert len(dt) >= len(dt_)
        assert all(len(t) >= min_len for t in dt_)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_clean_tokens_longer(tmpreproc_en):
    max_len = 7
    tokens = tmpreproc_en.tokens
    cleaned = tmpreproc_en.clean_tokens(remove_punct=False,
                                        remove_stopwords=False,
                                        remove_empty=False,
                                        remove_longer_than=max_len).tokens

    assert set(tokens.keys()) == set(cleaned.keys())

    for dl, dt in tokens.items():
        dt_ = cleaned[dl]
        assert len(dt) >= len(dt_)
        assert all(len(t) <= max_len for t in dt_)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_clean_tokens_remove_numbers(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    cleaned = tmpreproc_en.clean_tokens(remove_numbers=True).tokens

    assert set(tokens.keys()) == set(cleaned.keys())

    for dl, dt in tokens.items():
        dt_ = cleaned[dl]
        assert len(dt) >= len(dt_)
        assert all(not t.isnumeric() for t in dt_)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens('Moby')
    filtered = tmpreproc_en.tokens

    assert tmpreproc_en.vocabulary == ['Moby']
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dt in tokens.items():
        dt_ = filtered[dl]
        assert len(dt_) <= len(dt)

        if len(dt_) > 0:
            assert set(dt_) == {'Moby'}

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_tokens_inverse(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens('Moby', inverse=True)
    filtered = tmpreproc_en.tokens

    assert 'Moby' not in tmpreproc_en.vocabulary
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dt in tokens.items():
        dt_ = filtered[dl]
        assert len(dt_) <= len(dt)
        assert 'Moby' not in dt_

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_tokens_inverse_glob(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens('Mob*', inverse=True, match_type='glob')
    filtered = tmpreproc_en.tokens

    for w in tmpreproc_en.vocabulary:
        assert not w.startswith('Mob')

    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dt in tokens.items():
        dt_ = filtered[dl]
        assert len(dt_) <= len(dt)

        for t_ in dt_:
            assert not t_.startswith('Mob')

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_tokens_by_pattern(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens_by_pattern('^Mob.*')
    filtered = tmpreproc_en.tokens

    for w in tmpreproc_en.vocabulary:
        assert w.startswith('Mob')

    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dt in tokens.items():
        dt_ = filtered[dl]
        assert len(dt_) <= len(dt)

        for t_ in dt_:
            assert t_.startswith('Mob')

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_documents(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_documents('Moby')
    filtered = tmpreproc_en.tokens

    assert set(filtered.keys()) == {'melville-moby_dick.txt'}
    assert set(filtered.keys()) == set(tmpreproc_en.doc_labels)
    assert 'Moby' in tmpreproc_en.vocabulary
    assert filtered['melville-moby_dick.txt'] == tokens['melville-moby_dick.txt']

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_documents_by_pattern(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_documents_by_pattern('^Mob.*')
    filtered = tmpreproc_en.tokens

    assert 'melville-moby_dick.txt' in set(filtered.keys())
    assert set(filtered.keys()) == set(tmpreproc_en.doc_labels)
    assert 'Moby' in tmpreproc_en.vocabulary
    assert filtered['melville-moby_dick.txt'] == tokens['melville-moby_dick.txt']

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_for_pos(tmpreproc_en):
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos('N').tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        assert tok_pos_.meta_pos.str.startswith('N').all()

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_for_pos_none(tmpreproc_en):
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(None).tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        simpl_postags = [simplified_pos(pos) for pos in tok_pos_.meta_pos]
        assert all(pos is None for pos in simpl_postags)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_for_multiple_pos1(tmpreproc_en):
    req_tags = ['N', 'V']
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(req_tags).tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        simpl_postags = [simplified_pos(pos) for pos in tok_pos_.meta_pos]
        assert all(pos in req_tags for pos in simpl_postags)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_for_multiple_pos2(tmpreproc_en):
    req_tags = {'N', 'V', None}
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(req_tags).tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        simpl_postags = [simplified_pos(pos) for pos in tok_pos_.meta_pos]
        assert all(pos in req_tags for pos in simpl_postags)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_filter_for_pos_and_2nd_pass(tmpreproc_en):
    all_tok = tmpreproc_en.pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(['N', 'V']).filter_for_pos('V').tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        assert tok_pos_.meta_pos.str.startswith('V').all()

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


@pytest.mark.parametrize(
    'context_size, highlight_keyword, search_token',
    [(2, None, 'the'),    # default
     (1, None, 'the'),
     ((0, 2), None, 'the'),
     ((2, 0), None, 'the'),
     (2, '*', 'Moby')]
)
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
    'search_token, with_metadata, as_data_frame',
    [('the', True, False),    # default
     ('Moby', False, True),
     ('the', True, True)]
)
def test_tmpreproc_en_get_kwic_metadata_dataframe(tmpreproc_en, search_token, with_metadata, as_data_frame):
    tmpreproc_en.pos_tag()

    doc_labels = tmpreproc_en.doc_labels
    vocab = tmpreproc_en.vocabulary
    dtm = tmpreproc_en.dtm.tocsr()

    res = tmpreproc_en.get_kwic(search_token, with_metadata=with_metadata, as_data_frame=as_data_frame)
    ind_search_tok = vocab.index(search_token)

    if as_data_frame:
        assert isinstance(res, pd.DataFrame)
        assert list(res.index.names) == ['doc', 'context', 'position']

        meta_cols = ['meta_pos'] if with_metadata else []
        assert list(res.columns) == ['token'] + meta_cols
        for (dl, context), windf in res.groupby(level=[0, 1]):
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
def test_tmpreproc_en_get_kwic_table(tmpreproc_en, search_token):
    tmpreproc_en.pos_tag()

    doc_labels = tmpreproc_en.doc_labels
    vocab = tmpreproc_en.vocabulary
    dtm = tmpreproc_en.dtm.tocsr()

    res = tmpreproc_en.get_kwic_table(search_token)
    ind_search_tok = vocab.index(search_token)

    assert isinstance(res, pd.DataFrame)
    assert list(res.index.names) == ['doc', 'context']

    for (dl, context), windf in res.groupby(level=[0, 1]):
        assert dl in doc_labels
        dtm_row = dtm.getrow(doc_labels.index(dl))
        n_search_tok = dtm_row.getcol(ind_search_tok).A[0][0]
        assert 0 <= context < n_search_tok
        assert len(windf) == 1
        kwic_str = windf.kwic[0]
        assert isinstance(kwic_str, str)
        assert len(kwic_str) > 0
        assert kwic_str.count('*') == 2


def test_tmpreproc_en_get_dtm(tmpreproc_en):
    dtm = tmpreproc_en.get_dtm()
    dtm_prop = tmpreproc_en.dtm

    assert dtm.ndim == 2
    assert len(tmpreproc_en.doc_labels) == dtm.shape[0]
    assert len(tmpreproc_en.vocabulary) == dtm.shape[1]
    assert not (dtm != dtm_prop).toarray().any()

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_n_tokens(tmpreproc_en):
    assert tmpreproc_en.n_tokens == sum(map(len, tmpreproc_en.tokens.values()))


def test_tmpreproc_en_vocabulary_counts(tmpreproc_en):
    counts = tmpreproc_en.vocabulary_counts
    assert isinstance(counts, dict)
    assert len(counts) > 0
    assert set(counts.keys()) == set(tmpreproc_en.vocabulary)
    assert 'Moby' in counts.keys()
    assert all(0 < n <= tmpreproc_en.n_tokens for n in counts.values())
    assert tmpreproc_en.n_tokens == sum(counts.values())

    tmpreproc_en.shutdown_workers()


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

    tmpreproc_en.shutdown_workers()


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

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


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

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_apply_custom_filter(tmpreproc_en):
    tmpreproc_en.tokens_to_lowercase()
    vocab_orig = tmpreproc_en.vocabulary
    docs_orig = tmpreproc_en.doc_labels

    vocab_max_strlen = np.char.str_len(vocab_orig).max()

    def strip_words_with_max_len(tokens):
        new_tokens = {}
        for dl, dt in tokens.items():
            dt = np.array(dt, dtype=str)
            dt_len = np.char.str_len(dt)
            new_tokens[dl] = dt[dt_len < vocab_max_strlen].tolist()

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

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


def test_tmpreproc_en_pipeline(tmpreproc_en):
    orig_docs = tmpreproc_en.doc_labels
    orig_vocab = tmpreproc_en.vocabulary
    orig_tokensdf = tmpreproc_en.tokens_dataframe
    assert {'token'} == set(orig_tokensdf.columns)

    # part 1
    tmpreproc_en.pos_tag().lemmatize().tokens_to_lowercase().clean_tokens()

    assert orig_docs == tmpreproc_en.doc_labels
    assert set(tmpreproc_en.tokens.keys()) == set(orig_docs)
    new_vocab = tmpreproc_en.vocabulary
    assert len(orig_vocab) > len(new_vocab)

    tokensdf_part1 = tmpreproc_en.tokens_dataframe
    assert {'token', 'meta_pos'} == set(tokensdf_part1.columns)
    assert len(tokensdf_part1) < len(orig_tokensdf)   # because of "clean_tokens"

    dtm = tmpreproc_en.dtm
    assert dtm.ndim == 2
    assert dtm.shape[0] == tmpreproc_en.n_docs == len(tmpreproc_en.doc_labels)
    assert dtm.shape[1] == len(new_vocab)

    # part 2
    tmpreproc_en.filter_for_pos('N')

    assert len(new_vocab) > len(tmpreproc_en.vocabulary)

    tokensdf_part2 = tmpreproc_en.tokens_dataframe
    assert {'token', 'meta_pos'} == set(tokensdf_part2.columns)
    assert len(tokensdf_part2) < len(tokensdf_part1)   # because of "filter_for_pos"

    dtm = tmpreproc_en.dtm
    assert dtm.ndim == 2
    assert dtm.shape[0] == tmpreproc_en.n_docs == len(tmpreproc_en.doc_labels)
    assert dtm.shape[1] == len(tmpreproc_en.vocabulary)

    new_vocab2 = tmpreproc_en.vocabulary

    # part 3
    tmpreproc_en.filter_documents('moby')  # lower case already

    assert len(new_vocab2) > len(tmpreproc_en.vocabulary)

    tokensdf_part3 = tmpreproc_en.tokens_dataframe
    assert {'token', 'meta_pos'} == set(tokensdf_part3.columns)
    assert len(tokensdf_part3) < len(tokensdf_part2)   # because of "filter_documents"

    dtm = tmpreproc_en.dtm
    assert dtm.ndim == 2
    assert dtm.shape[0] == tmpreproc_en.n_docs == len(tmpreproc_en.doc_labels) == 1
    assert dtm.shape[1] == len(tmpreproc_en.vocabulary)

    _check_TMPreproc_copies(tmpreproc_en, tmpreproc_en.copy())
    _check_save_load_state(tmpreproc_en)

    tmpreproc_en.shutdown_workers()


#
# Tests with German corpus
# (only methods dependent on language are tested)
#


def test_tmpreproc_de_init(tmpreproc_de):
    assert tmpreproc_de.doc_labels == corpus_de.doc_labels
    assert len(tmpreproc_de.doc_labels) == N_DOCS_DE
    assert tmpreproc_de.language == 'german'

    _check_TMPreproc_copies(tmpreproc_de, tmpreproc_de.copy())
    _check_save_load_state(tmpreproc_de)

    tmpreproc_de.shutdown_workers()


def test_tmpreproc_de_tokenize(tmpreproc_de):
    tokens = tmpreproc_de.tokens
    assert set(tokens.keys()) == set(tmpreproc_de.doc_labels)

    for dt in tokens.values():
        assert isinstance(dt, list)
        assert len(dt) > 0

        # make sure that not all tokens only consist of a single character:
        assert np.sum(np.char.str_len(dt) > 1) > 1

    _check_TMPreproc_copies(tmpreproc_de, tmpreproc_de.copy())
    _check_save_load_state(tmpreproc_de)

    tmpreproc_de.shutdown_workers()


def test_tmpreproc_de_stem(tmpreproc_de):
    tokens = tmpreproc_de.tokens
    stems = tmpreproc_de.stem().tokens

    assert set(tokens.keys()) == set(stems.keys())

    for dl, dt in tokens.items():
        dt_ = stems[dl]
        assert len(dt) == len(dt_)

    _check_TMPreproc_copies(tmpreproc_de, tmpreproc_de.copy())
    _check_save_load_state(tmpreproc_de)

    tmpreproc_de.shutdown_workers()


def test_tmpreproc_de_pos_tag(tmpreproc_de):
    tmpreproc_de.pos_tag()
    tokens = tmpreproc_de.tokens
    tokens_with_pos_tags = tmpreproc_de.tokens_with_pos_tags

    assert set(tokens.keys()) == set(tokens_with_pos_tags.keys())

    for dl, dt in tokens.items():
        tok_pos_df = tokens_with_pos_tags[dl]
        assert len(dt) == len(tok_pos_df)
        assert list(tok_pos_df.columns) == ['token', 'meta_pos']
        assert np.array_equal(dt, tok_pos_df.token)
        assert all(tok_pos_df.meta_pos.str.len() > 0)

    _check_TMPreproc_copies(tmpreproc_de, tmpreproc_de.copy())
    _check_save_load_state(tmpreproc_de)

    tmpreproc_de.shutdown_workers()


def test_tmpreproc_de_lemmatize_fail_no_pos_tags(tmpreproc_de):
    with pytest.raises(ValueError):
        tmpreproc_de.lemmatize()

    _check_TMPreproc_copies(tmpreproc_de, tmpreproc_de.copy())
    _check_save_load_state(tmpreproc_de)

    tmpreproc_de.shutdown_workers()


def test_tmpreproc_de_lemmatize(tmpreproc_de):
    tokens = tmpreproc_de.tokens
    vocab = tmpreproc_de.vocabulary
    lemmata = tmpreproc_de.pos_tag().lemmatize().tokens

    assert set(tokens.keys()) == set(lemmata.keys())

    for dl, dt in tokens.items():
        dt_ = lemmata[dl]
        assert len(dt) == len(dt_)

    assert len(tmpreproc_de.vocabulary) < len(vocab)

    _check_TMPreproc_copies(tmpreproc_de, tmpreproc_de.copy())
    _check_save_load_state(tmpreproc_de)

    tmpreproc_de.shutdown_workers()
