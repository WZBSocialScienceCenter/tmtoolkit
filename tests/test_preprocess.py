from random import sample
from copy import deepcopy
import string

import numpy as np
import nltk
import pytest
import hypothesis.strategies as st
from hypothesis import given

from tmtoolkit.preprocess import TMPreproc, str_multisplit, expand_compound_token, remove_chars_in_tokens,\
    create_ngrams, ids2tokens
from tmtoolkit.corpus import Corpus
from tmtoolkit.utils import simplified_pos, tokens2ids

TMPREPROC_TEMP_STATE_FILE = '/tmp/tmpreproc_tests_state.pickle'


def test_str_multisplit():
    punct = list(string.punctuation)

    assert str_multisplit('US-Student', punct) == ['US', 'Student']
    assert str_multisplit('-main_file.exe,', punct) == ['', 'main', 'file', 'exe', '']


@given(s=st.text(), split_chars=st.lists(st.characters()))
def test_str_multisplit_hypothesis(s, split_chars):
    res = str_multisplit(s, split_chars)

    assert type(res) is list

    if len(s) == 0:
        assert res == ['']

    if len(split_chars) == 0:
        assert res == [s]

    for p in res:
        assert all(c not in p for c in split_chars)

    n_asserted_parts = 0
    for c in set(split_chars):
        n_asserted_parts += s.count(c)
    assert len(res) == n_asserted_parts + 1


def test_expand_compound_token():
    assert expand_compound_token('US-Student') == ['US', 'Student']
    assert expand_compound_token('US-Student-X') == ['US', 'StudentX']
    assert expand_compound_token('Student-X') == ['StudentX']
    assert expand_compound_token('Do-Not-Disturb') == ['Do', 'Not', 'Disturb']
    assert expand_compound_token('E-Mobility-Strategy') == ['EMobility', 'Strategy']

    assert expand_compound_token('US-Student', split_on_len=None, split_on_casechange=True) == ['USStudent']
    assert expand_compound_token('Do-Not-Disturb', split_on_len=None, split_on_casechange=True) == ['Do', 'Not', 'Disturb']
    assert expand_compound_token('E-Mobility-Strategy', split_on_len=None, split_on_casechange=True) == ['EMobility', 'Strategy']

    assert expand_compound_token('US-Student', split_on_len=2, split_on_casechange=True) == ['US', 'Student']
    assert expand_compound_token('Do-Not-Disturb', split_on_len=2, split_on_casechange=True) == ['Do', 'Not', 'Disturb']
    assert expand_compound_token('E-Mobility-Strategy', split_on_len=2, split_on_casechange=True) == ['EMobility', 'Strategy']

    assert expand_compound_token('E-Mobility-Strategy', split_on_len=1) == ['E', 'Mobility', 'Strategy']


@given(s=st.text(), split_chars=st.lists(st.characters()), split_on_len=st.integers(0), split_on_casechange=st.booleans())
def test_expand_compound_token_hypothesis(s, split_chars, split_on_len, split_on_casechange):
    if not split_on_len and not split_on_casechange:
        with pytest.raises(ValueError):
            expand_compound_token(s, split_chars, split_on_len=split_on_len, split_on_casechange=split_on_casechange)
    else:
        res = expand_compound_token(s, split_chars, split_on_len=split_on_len, split_on_casechange=split_on_casechange)

        assert type(res) is list

        if len(s) == 0:
            assert res == []

        assert all(p for p in res)

        # if res and split_chars and any(c in s for c in split_chars):
        #     if split_on_len:
        #         assert min(map(len, res)) >= split_on_len

        for p in res:
            assert all(c not in p for c in split_chars)


@given(tokens=st.lists(st.text()), special_chars=st.lists(st.characters()))
def test_remove_chars_in_tokens(tokens, special_chars):
    if len(special_chars) == 0:
        with pytest.raises(ValueError):
            remove_chars_in_tokens(tokens, special_chars)
    else:
        tokens_ = remove_chars_in_tokens(tokens, special_chars)
        assert len(tokens_) == len(tokens)

        for t_, t in zip(tokens_, tokens):
            assert len(t_) <= len(t)
            assert all(c not in t_ for c in special_chars)


@given(tokens=st.lists(st.text()), n=st.integers(0, 4))
def test_create_ngrams(tokens, n):
    n_tok = len(tokens)

    if n < 2:
        with pytest.raises(ValueError):
            create_ngrams(tokens, n)
    else:
        ngrams = create_ngrams(tokens, n, join=False)

        if n_tok < n:
            assert len(ngrams) == 1
            assert ngrams == [tokens]
        else:
            assert len(ngrams) == n_tok - n + 1
            assert all(len(g) == n for g in ngrams)

            tokens_ = list(ngrams[0])
            if len(ngrams) > 1:
                tokens_ += [g[-1] for g in ngrams[1:]]
            assert tokens_ == tokens

        ngrams_joined = create_ngrams(tokens, n, join=True, join_str='')
        assert len(ngrams_joined) == len(ngrams)

        for g_joined, g_tuple in zip(ngrams_joined, ngrams):
            assert g_joined == ''.join(g_tuple)


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


def _dataframes_equal(df1, df2):
    return df1.shape == df2.shape and (df1 == df2).all(axis=1).sum() == len(df1)


def _check_save_load_state(preproc, repeat=1, recreate_from_state=False):
    # copy simple attribute states
    simple_state_attrs = ('language', 'stopwords', 'punctuation', 'special_chars', 'n_workers',
                          'pos_tagged', 'ngrams_generated', 'ngrams_as_tokens')
    pre_state = {attr: deepcopy(getattr(preproc, attr)) for attr in simple_state_attrs}

    pre_state['tokens'] = preproc.tokens
    pre_state['vocabulary'] = preproc.vocabulary
    pre_state['doc_labels'] = preproc.doc_labels

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

    # check if states are the same now
    for attr in simple_state_attrs:
        assert pre_state[attr] == getattr(preproc, attr)

    assert set(pre_state['doc_labels']) == set(preproc.doc_labels)

    assert set(pre_state['tokens'].keys()) == set(preproc.tokens.keys())
    assert all(np.array_equal(pre_state['tokens'][k], preproc.tokens[k])
               for k in preproc.tokens.keys())

    assert np.array_equal(pre_state['vocabulary'], preproc.vocabulary)

    if preproc.pos_tagged:
        assert set(pre_state['tokens_with_pos_tags'].keys()) == set(preproc.tokens_with_pos_tags.keys())
        assert all(_dataframes_equal(pre_state['tokens_with_pos_tags'][k], preproc.tokens_with_pos_tags[k])
                   for k in preproc.tokens_with_pos_tags.keys())

    if preproc.ngrams_generated:
        assert set(pre_state['ngrams'].keys()) == set(preproc.ngrams.keys())
        assert all(np.array_equal(pre_state['ngrams'][k], preproc.ngrams[k])
                   for k in preproc.ngrams.keys())


#
# Tests with English corpus
#


def test_tmpreproc_en_init(tmpreproc_en):
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en)

    with pytest.raises(ValueError):    # because not POS tagged
        assert tmpreproc_en.tokens_with_pos_tags

    with pytest.raises(ValueError):    # because no ngrams generated
        assert tmpreproc_en.ngrams


def test_tmpreproc_en_save_load_state_several_times(tmpreproc_en):
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en, 5)


def test_tmpreproc_en_save_load_state_recreate_from_state(tmpreproc_en):
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en, recreate_from_state=True)


def test_tmpreproc_en_tokens_property(tmpreproc_en):
    tokens_all = tmpreproc_en.tokens
    assert set(tokens_all.keys()) == set(corpus_en.docs.keys())

    for dt in tokens_all.values():
        assert isinstance(dt, np.ndarray)
        assert dt.ndim == 1
        assert dt.dtype.char == 'U'
        if len(dt) > 0:
            # make sure that not all tokens only consist of a single character:
            assert np.sum(np.char.str_len(dt) > 1) > 1

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_get_tokens_and_tokens_with_metadata_property(tmpreproc_en):
    tokens_from_prop = tmpreproc_en.tokens
    tokens_w_meta = tmpreproc_en.tokens_with_metadata
    assert set(tokens_w_meta.keys()) == set(corpus_en.docs.keys())

    tokens_w_meta_from_fn = tmpreproc_en.get_tokens(with_metadata=True)

    for dl, df in tokens_w_meta.items():
        assert _dataframes_equal(df, tokens_w_meta_from_fn[dl])

        assert list(df.columns) == ['token']
        assert list(df.token) == list(tokens_from_prop[dl])

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_get_tokens_non_empty(tmpreproc_en):
    tokens = tmpreproc_en.get_tokens(non_empty=True)
    assert set(tokens.keys()) == set(corpus_en.docs.keys()) - {'empty_doc'}

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_doc_lengths(tmpreproc_en):
    doc_lengths = tmpreproc_en.doc_lengths
    assert set(doc_lengths.keys()) == set(corpus_en.docs.keys())

    for dl, dt in tmpreproc_en.tokens.items():
        assert doc_lengths[dl] == len(dt)


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


def test_tmpreproc_en_vocabulary(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    vocab = tmpreproc_en.vocabulary

    assert isinstance(vocab, np.ndarray)
    assert vocab.dtype.char == 'U'
    assert len(vocab) <= sum(tmpreproc_en.doc_lengths.values())

    # all tokens exist in vocab
    for dt in tokens.values():
        assert all(t in vocab for t in dt)

    # each word in vocab is used at least once
    for w in vocab:
        assert any(w in np.unique(dt) for dt in tokens.values())

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_ngrams(tmpreproc_en):
    bigrams = tmpreproc_en.generate_ngrams(2).ngrams

    assert tmpreproc_en.ngrams_generated is True
    assert set(bigrams.keys()) == set(corpus_en.docs.keys())

    for dl, dt in bigrams.items():
        assert isinstance(dt, np.ndarray)
        assert dt.dtype.char == 'U'

        if dl == 'empty_doc':
            assert dt.ndim == 1
            assert dt.shape[0] == 0
        else:
            assert dt.ndim == 2
            assert dt.shape[1] == 2


    # normal tokens are still unigrams
    for dt in tmpreproc_en.tokens.values():
        assert isinstance(dt, np.ndarray)
        assert dt.ndim == 1
        assert dt.dtype.char == 'U'

    tmpreproc_en.use_joined_ngrams_as_tokens()
    assert tmpreproc_en.ngrams_as_tokens is True

    # now tokens are bigrams
    for dt in tmpreproc_en.tokens.values():
        assert isinstance(dt, np.ndarray)
        assert dt.ndim == 1
        assert dt.dtype.char == 'U'

        if len(dt) > 0:
            for t in dt:
                split_t = t.split(' ')
                assert len(split_t) == 2

    #_check_save_load_state(tmpreproc_en)

    # fail when ngrams are used as tokens
    with pytest.raises(ValueError):
        tmpreproc_en.stem()
    with pytest.raises(ValueError):
        tmpreproc_en.lemmatize()
    with pytest.raises(ValueError):
        tmpreproc_en.expand_compound_tokens()
    with pytest.raises(ValueError):
        tmpreproc_en.pos_tag()


def test_tmpreproc_en_transform_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    # runs on workers because can be pickled:
    tokens_upper = tmpreproc_en.transform_tokens(str.upper).tokens

    for dl, dt in tokens.items():
        dt_ = tokens_upper[dl]
        assert len(dt) == len(dt_)
        assert all(t.upper() == t_ for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_transform_tokens_lambda(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    # runs on main thread because cannot be pickled:
    tokens_upper = tmpreproc_en.transform_tokens(lambda x: x.upper()).tokens

    for dl, dt in tokens.items():
        dt_ = tokens_upper[dl]
        assert len(dt) == len(dt_)
        assert all(t.upper() == t_ for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_tokens_to_lowercase(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tokens_lower = tmpreproc_en.tokens_to_lowercase().tokens

    assert set(tokens.keys()) == set(tokens_lower.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_lower[dl]
        assert len(dt) == len(dt_)
        assert all(t.lower() == t_ for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_stem(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    stems = tmpreproc_en.stem().tokens

    assert set(tokens.keys()) == set(stems.keys())

    for dl, dt in tokens.items():
        dt_ = stems[dl]
        assert len(dt) == len(dt_)

    _check_save_load_state(tmpreproc_en)


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


def test_tmpreproc_en_expand_compound_tokens(tmpreproc_en):
    tmpreproc_en.clean_tokens()
    tokens = tmpreproc_en.tokens
    tokens_exp = tmpreproc_en.expand_compound_tokens().tokens

    assert set(tokens.keys()) == set(tokens_exp.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_exp[dl]
        assert len(dt) <= len(dt_)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_expand_compound_tokens_same(tmpreproc_en):
    tmpreproc_en.remove_special_chars_in_tokens().clean_tokens()
    tokens = tmpreproc_en.tokens
    tokens_exp = tmpreproc_en.expand_compound_tokens().tokens

    assert set(tokens.keys()) == set(tokens_exp.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_exp[dl]
        assert all(t == t_ for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_remove_special_chars_in_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tokens_ = tmpreproc_en.remove_special_chars_in_tokens().tokens

    assert set(tokens.keys()) == set(tokens_.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_[dl]
        assert len(dt) == len(dt_)
        assert np.all(np.char.str_len(dt) >= np.char.str_len(dt_))

    _check_save_load_state(tmpreproc_en)


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

    _check_save_load_state(tmpreproc_en)


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

    _check_save_load_state(tmpreproc_en)


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

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_clean_tokens_remove_numbers(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    cleaned = tmpreproc_en.clean_tokens(remove_numbers=True).tokens

    assert set(tokens.keys()) == set(cleaned.keys())

    for dl, dt in tokens.items():
        dt_ = cleaned[dl]
        assert len(dt) >= len(dt_)
        assert all(not t.isnumeric() for t in dt_)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_tokens('Moby')
    filtered = tmpreproc_en.tokens

    assert np.array_equal(tmpreproc_en.vocabulary, np.array(['Moby']))
    assert set(tokens.keys()) == set(filtered.keys())

    for dl, dt in tokens.items():
        dt_ = filtered[dl]
        assert len(dt_) <= len(dt)

        if len(dt_) > 0:
            assert np.unique(dt_) == np.array(['Moby'])

    _check_save_load_state(tmpreproc_en)


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

    _check_save_load_state(tmpreproc_en)


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

    _check_save_load_state(tmpreproc_en)


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

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_documents(tmpreproc_en):
    tokens = tmpreproc_en.tokens
    tmpreproc_en.filter_documents('Moby')
    filtered = tmpreproc_en.tokens

    # fails:
    assert set(filtered.keys()) == {'melville-moby_dick.txt'}
    assert np.array_equal(tmpreproc_en.vocabulary, np.array(['Moby']))

    assert np.array_equal(filtered['melville-moby_dick.txt'], tokens['melville-moby_dick.txt'])

    _check_save_load_state(tmpreproc_en)



def test_tmpreproc_en_filter_for_pos(tmpreproc_en):
    all_tok = tmpreproc_en.tokenize().pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos('N').tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        assert all(pos.startswith('N') for _, pos in tok_pos_)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_pos_none(tmpreproc_en):
    all_tok = tmpreproc_en.tokenize().pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(None).tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        simpl_postags = [simplified_pos(pos) for _, pos in tok_pos_]
        assert all(pos is None for pos in simpl_postags)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_multiple_pos1(tmpreproc_en):
    req_tags = ['N', 'V']
    all_tok = tmpreproc_en.tokenize().pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(req_tags).tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        simpl_postags = [simplified_pos(pos) for _, pos in tok_pos_]
        assert all(pos in req_tags for pos in simpl_postags)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_multiple_pos2(tmpreproc_en):
    req_tags = {'N', 'V', None}
    all_tok = tmpreproc_en.tokenize().pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos(req_tags).tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        simpl_postags = [simplified_pos(pos) for _, pos in tok_pos_]
        assert all(pos in req_tags for pos in simpl_postags)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_pos_and_reset(tmpreproc_en):
    all_tok = tmpreproc_en.tokenize().pos_tag().tokens_with_pos_tags
    reset_tok = tmpreproc_en.filter_for_pos('N').reset_filter().tokens_with_pos_tags

    assert set(all_tok.keys()) == set(reset_tok.keys())

    for dl, tok_pos in all_tok.items():
        assert tok_pos == reset_tok[dl]

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_pos_and_2nd_pass(tmpreproc_en):
    all_tok = tmpreproc_en.tokenize().pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos('N').reset_filter().filter_for_pos('V').tokens_with_pos_tags

    assert set(all_tok.keys()) == set(filtered_tok.keys())

    for dl, tok_pos in all_tok.items():
        tok_pos_ = filtered_tok[dl]

        assert len(tok_pos_) <= len(tok_pos)
        assert all(pos.startswith('V') for _, pos in tok_pos_)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_get_dtm(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    dtm_res = tmpreproc_en.get_dtm()

    assert type(dtm_res) is tuple
    assert len(dtm_res) == 3

    doc_labels, vocab, dtm = dtm_res

    assert set(doc_labels) == set(tokens.keys())
    assert len(vocab) > 0
    assert len(doc_labels) == dtm.shape[0]
    assert len(vocab) == dtm.shape[1]
    assert set(vocab) == tmpreproc_en.vocabulary

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_get_dtm_from_ngrams(tmpreproc_en):
    tmpreproc_en.tokenize()

    with pytest.raises(ValueError):  # no ngrams generated
        tmpreproc_en.get_dtm(from_ngrams=True)

    bigrams = tmpreproc_en.generate_ngrams(2).ngrams
    dtm_res = tmpreproc_en.get_dtm(from_ngrams=True)

    assert type(dtm_res) is tuple
    assert len(dtm_res) == 3

    doc_labels, vocab, dtm = dtm_res

    assert set(doc_labels) == set(bigrams.keys())
    assert len(vocab) > 0
    assert len(doc_labels) == dtm.shape[0]
    assert len(vocab) == dtm.shape[1]

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_vocabulary_doc_frequency(tmpreproc_en):
    tmpreproc_en.tokenize()
    tokens = tmpreproc_en.tokens
    vocab = tmpreproc_en.vocabulary

    doc_freqs = tmpreproc_en.vocabulary_rel_doc_frequency
    abs_doc_freqs = tmpreproc_en.vocabulary_abs_doc_frequency
    assert len(doc_freqs) == len(abs_doc_freqs) == len(vocab)

    for t, f in doc_freqs.items():
        assert 0 < f <= 1
        n = abs_doc_freqs[t]
        assert n == sum([t in dt for dt in tokens.values()])
        assert abs(f - n/N_DOCS_EN) < 1e-6
        assert t in vocab


def test_tmpreproc_en_remove_common_or_uncommon_tokens(tmpreproc_en):
    tmpreproc_en.tokenize().tokens_to_lowercase()
    vocab_orig = tmpreproc_en.vocabulary

    tmpreproc_en.remove_uncommon_tokens(0.0)
    assert len(tmpreproc_en.vocabulary) == len(vocab_orig)

    tmpreproc_en.remove_common_tokens(0.9)
    assert len(tmpreproc_en.vocabulary) <= len(vocab_orig)

    doc_freqs = tmpreproc_en.vocabulary_rel_doc_frequency
    assert all(f < 0.9 for f in doc_freqs.values())

    tmpreproc_en.remove_uncommon_tokens(0.1)
    assert len(tmpreproc_en.vocabulary) <= len(vocab_orig)

    doc_freqs = tmpreproc_en.vocabulary_rel_doc_frequency
    assert all(f > 0.1 for f in doc_freqs.values())

    tmpreproc_en.remove_common_tokens(0.0)
    assert len(tmpreproc_en.vocabulary) == 0
    assert all(len(t) == 0 for t in tmpreproc_en.tokens.values())


def test_tmpreproc_en_remove_common_or_uncommon_tokens_absolute(tmpreproc_en):
    tmpreproc_en.tokenize().tokens_to_lowercase()
    vocab_orig = tmpreproc_en.vocabulary

    tmpreproc_en.remove_common_tokens(6, absolute=True)
    assert len(tmpreproc_en.vocabulary) <= len(vocab_orig)

    doc_freqs = tmpreproc_en.vocabulary_abs_doc_frequency
    assert all(n < 6 for n in doc_freqs.values())

    tmpreproc_en.remove_uncommon_tokens(1, absolute=True)
    assert len(tmpreproc_en.vocabulary) <= len(vocab_orig)

    doc_freqs = tmpreproc_en.vocabulary_abs_doc_frequency
    assert all(n > 1 for n in doc_freqs.values())

    tmpreproc_en.remove_common_tokens(1, absolute=True)
    assert len(tmpreproc_en.vocabulary) == 0
    assert all(len(t) == 0 for t in tmpreproc_en.tokens.values())


def test_tmpreproc_en_apply_custom_filter(tmpreproc_en):
    tmpreproc_en.tokenize().tokens_to_lowercase()
    vocab_orig = tmpreproc_en.vocabulary
    docs_orig = set(tmpreproc_en.tokens.keys())

    vocab_max_strlen = max(map(len, vocab_orig))
    def strip_words_with_max_len(tokens):
        return {dl: [tup for tup in dt if len(tup[0]) < vocab_max_strlen]   # tup is tuple(token) but could be tuple(token, pos)
                for dl, dt in tokens.items()}
    tmpreproc_en.apply_custom_filter(strip_words_with_max_len)

    new_vocab = tmpreproc_en.vocabulary

    assert new_vocab != vocab_orig
    assert max(map(len, new_vocab)) < vocab_max_strlen

    assert set(tmpreproc_en.tokens.keys()) == docs_orig

    tmpreproc_en.apply_custom_filter(strip_words_with_max_len)
    assert tmpreproc_en.vocabulary == new_vocab   # applying twice shouldn't change anything

#
# Tests with German corpus
# (only methods dependent on language are tested)
#


def test_tmpreproc_de_init(tmpreproc_de):
    assert tmpreproc_de.docs == corpus_de.docs
    assert len(tmpreproc_de.docs) == N_DOCS_DE
    assert tmpreproc_de.language == 'german'

    _check_save_load_state(tmpreproc_de)


def test_tmpreproc_de_tokenize(tmpreproc_de):
    tokens = tmpreproc_de.tokenize().tokens
    assert set(tokens.keys()) == set(tmpreproc_de.docs.keys())

    for dt in tokens.values():
        assert type(dt) in (tuple, list)
        assert len(dt) > 0
        assert any(len(t) > 1 for t in dt)  # make sure that not all tokens only consist of a single character

    _check_save_load_state(tmpreproc_de)


def test_tmpreproc_de_stem(tmpreproc_de):
    tokens = tmpreproc_de.tokenize().tokens
    stems = tmpreproc_de.stem().tokens

    assert set(tokens.keys()) == set(stems.keys())

    for dl, dt in tokens.items():
        dt_ = stems[dl]
        assert len(dt) == len(dt_)

    _check_save_load_state(tmpreproc_de)


def test_tmpreproc_de_pos_tag(tmpreproc_de):
    tmpreproc_de.tokenize().pos_tag()
    tokens = tmpreproc_de.tokens
    tokens_with_pos_tags = tmpreproc_de.tokens_with_pos_tags

    assert set(tokens.keys()) == set(tokens_with_pos_tags.keys())

    for dl, dt in tokens.items():
        tok_pos = tokens_with_pos_tags[dl]
        assert len(dt) == len(tok_pos)
        for t, (t_, pos) in zip(dt, tok_pos):
            assert t == t_
            assert pos

    _check_save_load_state(tmpreproc_de)


def test_tmpreproc_de_lemmatize_fail_no_pos_tags(tmpreproc_de):
    with pytest.raises(ValueError):
        tmpreproc_de.tokenize().lemmatize()

    _check_save_load_state(tmpreproc_de)


def test_tmpreproc_de_lemmatize(tmpreproc_de):
    tokens = tmpreproc_de.tokenize().tokens
    lemmata = tmpreproc_de.pos_tag().lemmatize().tokens

    assert set(tokens.keys()) == set(lemmata.keys())

    for dl, dt in tokens.items():
        dt_ = lemmata[dl]
        assert len(dt) == len(dt_)

    _check_save_load_state(tmpreproc_de)


def test_utils_tokens2ids_lists():
    tok = [list('ABC'), list('ACAB'), list('DEA')]  # tokens2ids converts those to numpy arrays

    vocab, tokids = tokens2ids(tok)

    assert isinstance(vocab, np.ndarray)
    assert np.array_equal(vocab, np.array(list('ABCDE')))
    assert len(tokids) == 3
    assert isinstance(tokids[0], np.ndarray)
    assert np.array_equal(tokids[0], np.array([0, 1, 2]))
    assert np.array_equal(tokids[1], np.array([0, 2, 0, 1]))
    assert np.array_equal(tokids[2], np.array([3, 4, 0]))


def test_utils_tokens2ids_nparrays():
    tok = [list('ABC'), list('ACAB'), list('DEA')]  # tokens2ids converts those to numpy arrays
    tok = list(map(np.array, tok))

    vocab, tokids = tokens2ids(tok)

    assert isinstance(vocab, np.ndarray)
    assert np.array_equal(vocab, np.array(list('ABCDE')))
    assert len(tokids) == 3
    assert isinstance(tokids[0], np.ndarray)
    assert np.array_equal(tokids[0], np.array([0, 1, 2]))
    assert np.array_equal(tokids[1], np.array([0, 2, 0, 1]))
    assert np.array_equal(tokids[2], np.array([3, 4, 0]))


@given(tok=st.lists(st.integers(0, 100), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.text(), min_size=0, max_size=size[0]),
                          min_size=0, max_size=size[1])
    )
)
def test_utils_tokens2ids_and_ids2tokens(tok):
    tok = list(map(lambda x: np.array(x, dtype=np.str), tok))

    vocab, tokids = tokens2ids(tok)

    assert isinstance(vocab, np.ndarray)
    if tok:
        assert np.array_equal(vocab, np.unique(np.concatenate(tok)))
    else:
        assert np.array_equal(vocab, np.array([], dtype=np.str))

    assert len(tokids) == len(tok)

    tok2 = ids2tokens(vocab, tokids)
    assert len(tok2) == len(tok)

    for orig_tok, tokid, inversed_tokid_tok in zip(tok, tokids, tok2):
        assert isinstance(tokid, np.ndarray)
        assert len(tokid) == len(orig_tok)
        assert np.array_equal(orig_tok, inversed_tokid_tok)
