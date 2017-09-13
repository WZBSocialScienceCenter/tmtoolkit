from __future__ import division
import sys
from random import sample
from copy import deepcopy
import string

import nltk
import pytest
import hypothesis.strategies as st
from hypothesis import given

from tmtoolkit.preprocess import TMPreproc, str_multisplit, expand_compound_token, remove_special_chars_in_tokens,\
    create_ngrams
from tmtoolkit.corpus import Corpus


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
def test_remove_special_chars_in_tokens(tokens, special_chars):
    if len(special_chars) == 0:
        with pytest.raises(ValueError):
            remove_special_chars_in_tokens(tokens, special_chars)
    else:
        tokens_ = remove_special_chars_in_tokens(tokens, special_chars)
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
N_DOCS_EN = 6
N_DOCS_DE = 3   # given from corpus size

all_docs_en = {f_id: nltk.corpus.gutenberg.raw(f_id) for f_id in nltk.corpus.gutenberg.fileids()}
smaller_docs_en = [(y[0], y[1][:min(y[2], MAX_DOC_LEN)])
                   for y in map(lambda x: (x[0], x[1], len(x[1])), all_docs_en.items())]

corpus_en = Corpus(dict(sample(smaller_docs_en, N_DOCS_EN)))
#corpus_en = Corpus(dict(smaller_docs_en))
corpus_de = Corpus.from_folder('examples/data/gutenberg', read_size=MAX_DOC_LEN)


@pytest.fixture
def tmpreproc_en():
    return TMPreproc(corpus_en.docs, language='english')


@pytest.fixture
def tmpreproc_de():
    return TMPreproc(corpus_de.docs, language='german')


def test_fixtures(tmpreproc_en, tmpreproc_de):
    assert len(tmpreproc_en.docs) == N_DOCS_EN
    assert len(tmpreproc_de.docs) == N_DOCS_DE

    assert all(0 < len(doc) <= MAX_DOC_LEN for doc in tmpreproc_en.docs.values())
    assert all(0 < len(doc) <= MAX_DOC_LEN for doc in tmpreproc_de.docs.values())


def _check_save_load_state(preproc, repeat=1, recreate_from_state=False):
    # copy simple attribute states
    simple_state_attrs = ('language', 'stopwords', 'punctuation', 'special_chars', 'n_workers',
                   'tokenized', 'pos_tagged', 'ngrams_generated', 'ngrams_as_tokens')
    pre_state = {attr: deepcopy(getattr(preproc, attr)) for attr in simple_state_attrs}

    # copy complex attribute states
    pre_state['docs'] = deepcopy(preproc.docs)

    if preproc.tokenized:
        pre_state['tokens'] = preproc.tokens
        pre_state['vocabulary'] = preproc.vocabulary
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

    assert set(pre_state['docs'].keys()) == set(preproc.docs.keys())
    assert preproc.n_docs == len(pre_state['docs'])
    assert all(pre_state['docs'][k] == preproc.docs[k] for k in preproc.docs.keys())

    if preproc.tokenized:
        assert set(pre_state['tokens'].keys()) == set(preproc.tokens.keys())
        assert all(pre_state['tokens'][k] == preproc.tokens[k] for k in preproc.tokens.keys())

        assert pre_state['vocabulary'] == preproc.vocabulary

    if preproc.pos_tagged:
        assert set(pre_state['tokens_with_pos_tags'].keys()) == set(preproc.tokens_with_pos_tags.keys())
        assert all(pre_state['tokens_with_pos_tags'][k] == preproc.tokens_with_pos_tags[k]
                   for k in preproc.tokens_with_pos_tags.keys())

    if preproc.ngrams_generated:
        assert set(pre_state['ngrams'].keys()) == set(preproc.ngrams.keys())
        assert all(pre_state['ngrams'][k] == preproc.ngrams[k] for k in preproc.ngrams.keys())


#
# Tests with English corpus
#


def test_tmpreproc_en_init(tmpreproc_en):
    assert tmpreproc_en.docs == corpus_en.docs
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en)

    with pytest.raises(ValueError):    # because not tokenized
        assert tmpreproc_en.tokens

    with pytest.raises(ValueError):    # same
        assert tmpreproc_en.tokens_with_pos_tags

    with pytest.raises(ValueError):    # same
        assert tmpreproc_en.ngrams


def test_tmpreproc_en_save_load_state_several_times(tmpreproc_en):
    assert tmpreproc_en.docs == corpus_en.docs
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en, 5)


def test_tmpreproc_en_save_load_state_recreate_from_state(tmpreproc_en):
    assert tmpreproc_en.docs == corpus_en.docs
    assert tmpreproc_en.language == 'english'

    _check_save_load_state(tmpreproc_en, recreate_from_state=True)


def test_tmpreproc_no_tokens_fail(tmpreproc_en):
    with pytest.raises(ValueError):   # because not tokenized
        tmpreproc_en.generate_ngrams(2)

    with pytest.raises(ValueError):   # same
        tmpreproc_en.transform_tokens(lambda x: x)

    with pytest.raises(ValueError):   # same
        tmpreproc_en.stem()

    with pytest.raises(ValueError):   # same
        tmpreproc_en.expand_compound_tokens()

    with pytest.raises(ValueError):   # same
        tmpreproc_en.remove_special_chars_in_tokens()

    with pytest.raises(ValueError):   # same
        tmpreproc_en.clean_tokens()

    with pytest.raises(ValueError):   # same
        tmpreproc_en.pos_tag()

    with pytest.raises(ValueError):   # same
        tmpreproc_en.filter_for_tokenpattern(r'.*')

    with pytest.raises(ValueError):   # same
        tmpreproc_en.get_dtm()


def test_tmpreproc_en_tokenize(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    assert set(tokens.keys()) == set(tmpreproc_en.docs.keys())

    for dt in tokens.values():
        assert type(dt) in (tuple, list)
        assert len(dt) > 0
        assert any(len(t) > 1 for t in dt)  # make sure that not all tokens only consist of a single character

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_vocabulary(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    vocab = tmpreproc_en.tokenize().vocabulary
    assert len(vocab) <= sum(len(dt) for dt in tokens.values())

    for dt in tokens.values():
        assert all(t in vocab for t in dt)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_ngrams(tmpreproc_en):
    bigrams = tmpreproc_en.tokenize().generate_ngrams(2).ngrams
    assert set(bigrams.keys()) == set(tmpreproc_en.docs.keys())

    for dt in bigrams.values():
        assert type(dt) in (tuple, list)
        assert len(dt) > 0

    first_doc = next(iter(bigrams.keys()))

    bigrams_unjoined = tmpreproc_en.tokenize().generate_ngrams(2, join=False).ngrams
    first_bigram = bigrams_unjoined[first_doc][0]
    assert type(first_bigram) in (tuple, list)
    assert len(first_bigram) == 2
    assert ' '.join(first_bigram) == bigrams[first_doc][0]

    tokens = tmpreproc_en.generate_ngrams(2, reassign_tokens=True).tokens
    for dl, dt in tokens.items():
        dt_ = tmpreproc_en.ngrams[dl]
        assert all(t == t_ for t, t_ in zip(dt, dt_))
    assert tmpreproc_en.ngrams_as_tokens is True
    assert tmpreproc_en.pos_tagged is False

    bigrams = tmpreproc_en.tokenize().generate_ngrams(2).ngrams
    tokens = tmpreproc_en.use_ngrams_as_tokens().tokens
    for dl, dt in tokens.items():
        dt_ = bigrams[dl]
        assert all(t == t_ for t, t_ in zip(dt, dt_))
    assert tmpreproc_en.ngrams_as_tokens is True
    assert tmpreproc_en.pos_tagged is False

    _check_save_load_state(tmpreproc_en)

    # fail when reassign_tokens was set to true:
    with pytest.raises(ValueError):
        tmpreproc_en.stem()
    with pytest.raises(ValueError):
        tmpreproc_en.lemmatize()
    with pytest.raises(ValueError):
        tmpreproc_en.expand_compound_tokens()
    with pytest.raises(ValueError):
        tmpreproc_en.pos_tag()


def test_tmpreproc_en_transform_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    fn = string.upper if sys.version_info[0] < 3 else str.upper
    tokens_upper = tmpreproc_en.transform_tokens(fn).tokens

    for dl, dt in tokens.items():
        dt_ = tokens_upper[dl]
        assert len(dt) == len(dt_)
        assert all(t.upper() == t_ for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_transform_tokens_lambda(tmpreproc_en):
    with pytest.raises(ValueError):
        tmpreproc_en.tokenize().transform_tokens(lambda x: x.upper())

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_tokens_to_lowercase(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    tokens_upper = tmpreproc_en.tokens_to_lowercase().tokens

    assert set(tokens.keys()) == set(tokens_upper.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_upper[dl]
        assert len(dt) == len(dt_)
        assert all(t.lower() == t_ for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_stem(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    stems = tmpreproc_en.stem().tokens

    assert set(tokens.keys()) == set(stems.keys())

    for dl, dt in tokens.items():
        dt_ = stems[dl]
        assert len(dt) == len(dt_)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_pos_tag(tmpreproc_en):
    tmpreproc_en.tokenize().pos_tag()
    tokens = tmpreproc_en.tokens
    tokens_with_pos_tags = tmpreproc_en.tokens_with_pos_tags

    assert set(tokens.keys()) == set(tokens_with_pos_tags.keys())

    for dl, dt in tokens.items():
        tok_pos = tokens_with_pos_tags[dl]
        assert len(dt) == len(tok_pos)
        for t, (t_, pos) in zip(dt, tok_pos):
            assert t == t_
            assert pos

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_lemmatize_fail_no_pos_tags(tmpreproc_en):
    with pytest.raises(ValueError):
        tmpreproc_en.tokenize().lemmatize()


def test_tmpreproc_en_lemmatize(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    lemmata = tmpreproc_en.pos_tag().lemmatize().tokens

    assert set(tokens.keys()) == set(lemmata.keys())

    for dl, dt in tokens.items():
        dt_ = lemmata[dl]
        assert len(dt) == len(dt_)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_expand_compound_tokens(tmpreproc_en):
    tmpreproc_en.tokenize().clean_tokens()
    tokens = tmpreproc_en.tokens
    tokens_exp = tmpreproc_en.expand_compound_tokens().tokens

    assert set(tokens.keys()) == set(tokens_exp.keys())

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_expand_compound_tokens_same(tmpreproc_en):
    tmpreproc_en.tokenize().remove_special_chars_in_tokens().clean_tokens()
    tokens = tmpreproc_en.tokens
    tokens_exp = tmpreproc_en.expand_compound_tokens().tokens

    assert set(tokens.keys()) == set(tokens_exp.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_exp[dl]
        assert all(t == t_ for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_remove_special_chars_in_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    tokens_ = tmpreproc_en.remove_special_chars_in_tokens().tokens

    assert set(tokens.keys()) == set(tokens_.keys())

    for dl, dt in tokens.items():
        dt_ = tokens_[dl]
        assert len(dt) == len(dt_)
        assert all(len(t) >= len(t_) for t, t_ in zip(dt, dt_))

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_clean_tokens(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    cleaned = tmpreproc_en.clean_tokens().tokens

    assert set(tokens.keys()) == set(cleaned.keys())

    for dl, dt in tokens.items():
        dt_ = cleaned[dl]
        assert len(dt) >= len(dt_)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_token(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    filtered = tmpreproc_en.filter_for_token('the').tokens

    assert len(tokens) >= len(filtered)

    for dl, dt in tokens.items():
        dt_ = filtered[dl]
        assert dt == dt_
        assert any(t == 'the' for t in dt_)

    filtered_pttrn = tmpreproc_en.filter_for_tokenpattern(r'^the$').tokens
    assert len(tokens) >= len(filtered_pttrn)

    for dl, dt in filtered.items():
        dt_ = filtered_pttrn[dl]
        assert dt == dt_

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_tokenpattern(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    filtered = tmpreproc_en.filter_for_tokenpattern(r'^the.+').tokens

    assert len(tokens) == len(filtered)

    for dl, dt in tokens.items():
        dt_ = filtered[dl]
        assert dt == dt_
        assert any(t.startswith('the') and len(t) >= 4 for t in dt_)

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_tokenpattern_and_reset(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    tokens_reset = tmpreproc_en.filter_for_tokenpattern(r'^the.+').reset_filter().tokens

    assert len(tokens) == len(tokens_reset)

    for dl, dt in tokens.items():
        dt_ = tokens_reset[dl]
        assert dt == dt_

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_tokenpattern_2nd_pass(tmpreproc_en):
    tokens = tmpreproc_en.tokenize().tokens
    filtered = tmpreproc_en.filter_for_tokenpattern(r'^the.+').filter_for_tokenpattern(r'^un.+').tokens

    assert len(tokens) >= len(filtered)

    for dl, dt in tokens.items():
        dt_ = filtered.get(dl, None)
        if dt_ is not None:
            assert dt == dt_
            assert any(t.startswith('un') and len(t) >= 3 for t in dt_)

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


def test_tmpreproc_en_filter_for_pos_and_reset(tmpreproc_en):
    all_tok = tmpreproc_en.tokenize().pos_tag().tokens_with_pos_tags
    reset_tok = tmpreproc_en.filter_for_pos('N').reset_filter().tokens_with_pos_tags

    assert set(all_tok.keys()) == set(reset_tok.keys())

    for dl, tok_pos in all_tok.items():
        assert tok_pos == reset_tok[dl]

    _check_save_load_state(tmpreproc_en)


def test_tmpreproc_en_filter_for_pos_and_2nd_pass(tmpreproc_en):
    all_tok = tmpreproc_en.tokenize().pos_tag().tokens_with_pos_tags
    filtered_tok = tmpreproc_en.filter_for_pos('N').filter_for_pos('V').tokens_with_pos_tags

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


def test_tmpreproc_de_vocabulary_doc_frequency(tmpreproc_en):
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
