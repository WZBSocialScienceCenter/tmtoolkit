from random import sample
import string

import nltk
import pytest
import hypothesis.strategies as st
from hypothesis import given

from tmtoolkit.preprocess import TMPreproc, str_multisplit, expand_compound_token, remove_special_chars_in_tokens,\
    create_ngrams
from tmtoolkit.corpus import Corpus


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


corpus_en = Corpus({f_id: nltk.corpus.gutenberg.raw(f_id) for f_id in sample(nltk.corpus.gutenberg.fileids(), 3)})


@pytest.fixture
def tmpreproc_en():
    return TMPreproc(corpus_en.docs, language='english')


def test_tmpreproc_en_init(tmpreproc_en):
    assert tmpreproc_en.docs == corpus_en.docs
    assert tmpreproc_en.language == 'english'

    with pytest.raises(ValueError):    # because not tokenized
        assert tmpreproc_en.tokens

    with pytest.raises(ValueError):    # same
        assert tmpreproc_en.tokens_with_pos_tags

    with pytest.raises(ValueError):    # same
        assert tmpreproc_en.ngrams


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

    # fail when reassign_tokens was set to true:
    with pytest.raises(ValueError):
        tmpreproc_en.stem()
    with pytest.raises(ValueError):
        tmpreproc_en.lemmatize()
    with pytest.raises(ValueError):
        tmpreproc_en.expand_compound_tokens()
    with pytest.raises(ValueError):
        tmpreproc_en.pos_tag()


