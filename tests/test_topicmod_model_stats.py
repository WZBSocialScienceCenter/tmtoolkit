from __future__ import division

import re
import itertools
import math
import random
import string

import six
import lda
import numpy as np
import pytest
from hypothesis import given, strategies as st
from scipy.sparse import coo_matrix

from tmtoolkit.topicmod import model_stats


@given(n=st.integers(0, 10),
       distrib=st.lists(st.integers(0, 9), min_size=2, max_size=2).flatmap(
           lambda size: st.lists(st.lists(st.floats(0, 1, allow_nan=False, allow_infinity=False),
                                          min_size=size[0], max_size=size[0]),
                                 min_size=size[1], max_size=size[1])
       ))
def test_top_n_from_distribution(n, distrib):
    distrib = np.array(distrib)
    if len(distrib) == 0:
        with pytest.raises(ValueError):
            model_stats.top_n_from_distribution(distrib, n)
    else:
        if n < 1 or n > distrib.shape[1]:
            with pytest.raises(ValueError):
                model_stats.top_n_from_distribution(distrib, n)
        else:
            df = model_stats.top_n_from_distribution(distrib, n)

            assert len(df) == len(distrib)

            for _, row in df.iterrows():
                assert len(row) == n
                assert list(sorted(row, reverse=True)) == list(row)


@given(topic_word_distrib=st.lists(st.integers(0, 9), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.floats(0, 1, allow_nan=False, allow_infinity=False),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    vocab=st.lists(st.text(string.printable), min_size=0, max_size=9),
    top_n=st.integers(0, 10))
def test_top_words_for_topics(topic_word_distrib, vocab, top_n):
    topic_word_distrib = np.array(topic_word_distrib)
    vocab = np.array(vocab)

    if len(topic_word_distrib) == 0 or len(vocab) == 0 or topic_word_distrib.shape[1] != len(vocab):
        with pytest.raises(ValueError):
            model_stats.top_words_for_topics(topic_word_distrib, top_n, vocab)
        return

    if top_n < 1 or top_n > topic_word_distrib.shape[1]:
        with pytest.raises(ValueError):
            model_stats.top_words_for_topics(topic_word_distrib, top_n, vocab)
        return

    top_words = model_stats.top_words_for_topics(topic_word_distrib, top_n, vocab)
    assert isinstance(top_words, list)
    assert len(top_words) == topic_word_distrib.shape[0]
    assert all(l == top_n for l in map(len, top_words))
    assert all(w in vocab for w in sum(map(list, top_words), []))

    top_words = model_stats.top_words_for_topics(topic_word_distrib, top_n)  # no vocab -> return word indices
    assert isinstance(top_words, list)
    assert len(top_words) == topic_word_distrib.shape[0]
    assert all(l == top_n for l in map(len, top_words))
    assert all(w_idx in range(len(vocab)) for w_idx in sum(map(list, top_words), []))


def test_top_words_for_topics2():
    distrib = np.array([
        [3, 2, 1],
        [1, 3, 2],
        [1, 0, 1],
    ])

    vocab = np.array(['a', 'b', 'c'])

    top_words = model_stats.top_words_for_topics(distrib, 2, vocab)
    assert len(top_words) == len(distrib)
    top_words_lists = list(map(list, top_words))
    assert top_words_lists[0] == ['a', 'b']
    assert top_words_lists[1] == ['b', 'c']
    assert top_words_lists[2] in (['a', 'c'], ['c', 'a'])

    top_words, top_probs = model_stats.top_words_for_topics(distrib, 2, vocab, return_prob=True)
    assert len(top_words) == len(top_probs)
    assert np.allclose(top_probs[0], np.array([3, 2]))
    assert np.allclose(top_probs[1], np.array([3, 2]))
    assert np.allclose(top_probs[2], np.array([1, 1]))

    top_words = model_stats.top_words_for_topics(distrib, 2)  # no vocab -> return word indices
    assert len(top_words) == len(distrib)
    top_words_lists = list(map(list, top_words))
    assert top_words_lists[0] == [0, 1]
    assert top_words_lists[1] == [1, 2]
    assert top_words_lists[2] in ([0, 2], [2, 0])

    top_words = model_stats.top_words_for_topics(distrib)
    assert all(len(top_words[i]) == len(vocab) for i in range(len(distrib)))


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=2))
def test_get_doc_lengths(dtm, matrix_type):
    if matrix_type == 1:
        dtm = np.matrix(dtm)
        dtm_arr = dtm.A
    elif matrix_type == 2:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm = np.array(dtm)
        dtm_arr = dtm

    if dtm_arr.ndim != 2:
        with pytest.raises(ValueError):
            model_stats.get_doc_lengths(dtm)
    else:
        doc_lengths = model_stats.get_doc_lengths(dtm)
        assert doc_lengths.ndim == 1
        assert doc_lengths.shape == (dtm_arr.shape[0],)
        assert doc_lengths.tolist() == [sum(row) for row in dtm_arr]


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=2))
def test_get_doc_frequencies(dtm, matrix_type):
    if matrix_type == 1:
        dtm = np.matrix(dtm)
        dtm_arr = dtm.A
    elif matrix_type == 2:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm = np.array(dtm)
        dtm_arr = dtm

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            model_stats.get_doc_frequencies(dtm)
    else:
        n_docs = dtm.shape[0]

        df_abs = model_stats.get_doc_frequencies(dtm)
        assert isinstance(df_abs, np.ndarray)
        assert df_abs.ndim == 1
        assert df_abs.shape == (dtm_arr.shape[1],)
        assert all([0 <= v <= n_docs for v in df_abs])

        df_rel = model_stats.get_doc_frequencies(dtm, proportions=True)
        assert isinstance(df_rel, np.ndarray)
        assert df_rel.ndim == 1
        assert df_rel.shape == (dtm_arr.shape[1],)
        assert all([0 <= v <= 1 for v in df_rel])


def test_get_doc_frequencies2():
    dtm = np.array([
        [0, 2, 3, 0, 0],
        [1, 2, 0, 5, 0],
        [0, 1, 0, 3, 1],
    ])

    df = model_stats.get_doc_frequencies(dtm)

    assert df.tolist() == [1, 3, 1, 2, 1]


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=2),
    proportions=st.booleans())
def test_get_codoc_frequencies(dtm, matrix_type, proportions):
    if matrix_type == 1:
        dtm = np.matrix(dtm)
    elif matrix_type == 2:
        dtm = coo_matrix(dtm)
    else:
        dtm = np.array(dtm)

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            model_stats.get_codoc_frequencies(dtm, proportions=proportions)
        return

    n_docs, n_vocab = dtm.shape

    if n_vocab < 2:
        with pytest.raises(ValueError):
            model_stats.get_codoc_frequencies(dtm, proportions=proportions)
        return

    df = model_stats.get_codoc_frequencies(dtm, proportions=proportions)
    assert isinstance(df, dict)
    assert len(df) == math.factorial(n_vocab) / math.factorial(2) / math.factorial(n_vocab - 2)
    for w1, w2 in itertools.combinations(range(n_vocab), 2):
        n = df[(w1, w2)]
        if proportions:
            assert 0 <= n <= 1
        else:
            assert 0 <= n <= n_docs


def test_get_codoc_frequencies2():
    dtm = np.array([
        [0, 2, 3, 0, 0],
        [1, 2, 0, 5, 0],
        [0, 1, 0, 3, 1],
    ])

    df = model_stats.get_codoc_frequencies(dtm)

    assert len(df) == math.factorial(5) / math.factorial(2) / math.factorial(3)
    # just check a few
    assert df.get((0, 1), df.get((1, 0))) == 1
    assert df.get((1, 3), df.get((3, 1))) == 2
    assert df.get((0, 2), df.get((2, 0))) == 0


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=2))
def test_get_term_frequencies(dtm, matrix_type):
    if matrix_type == 1:
        dtm = np.matrix(dtm)
        dtm_arr = dtm.A
    elif matrix_type == 2:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm = np.array(dtm)
        dtm_arr = dtm

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            model_stats.get_term_frequencies(dtm)
    else:
        tf = model_stats.get_term_frequencies(dtm)
        assert tf.ndim == 1
        assert tf.shape == (dtm_arr.shape[1],)
        assert tf.tolist() == [sum(row) for row in dtm_arr.T]


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=2))
def test_get_term_proportions(dtm, matrix_type):
    if matrix_type == 1:
        dtm = np.matrix(dtm)
        dtm_arr = dtm.A
        dtm_flat = dtm.A1
    elif matrix_type == 2:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
        dtm_flat = dtm.A.flatten()
    else:
        dtm = np.array(dtm)
        dtm_arr = dtm
        dtm_flat = dtm.flatten()

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            model_stats.get_term_proportions(dtm)
    else:
        if dtm.sum() == 0:
            with pytest.raises(ValueError):
                model_stats.get_term_proportions(dtm)
        else:
            tp = model_stats.get_term_proportions(dtm)
            assert tp.ndim == 1
            assert tp.shape == (dtm_arr.shape[1],)

            if len(dtm_flat) > 0:
                assert np.isclose(tp.sum(), 1.0)
                assert all(0 <= v <= 1 for v in tp)


@given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    n_topics=st.integers(2, 10))
def test_get_marginal_topic_distrib(dtm, n_topics):
    dtm = np.array(dtm)
    if dtm.sum() == 0:  # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = model_stats.get_doc_lengths(dtm)
    marginal_topic_distr = model_stats.get_marginal_topic_distrib(model.doc_topic_, doc_lengths)

    assert marginal_topic_distr.shape == (n_topics,)
    assert np.isclose(marginal_topic_distr.sum(), 1.0)
    assert all(0 <= v <= 1 for v in marginal_topic_distr)


@given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    n_topics=st.integers(2, 10))
def test_get_marginal_word_distrib(dtm, n_topics):
    dtm = np.array(dtm)
    if dtm.sum() == 0:  # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = model_stats.get_doc_lengths(dtm)
    p_t = model_stats.get_marginal_topic_distrib(model.doc_topic_, doc_lengths)

    p_w = model_stats.get_marginal_word_distrib(model.topic_word_, p_t)
    assert p_w.shape == (dtm.shape[1],)
    assert np.isclose(p_w.sum(), 1.0)
    assert all(0 <= v <= 1 for v in p_w)


@given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    n_topics=st.integers(2, 10))
def test_get_word_distinctiveness(dtm, n_topics):
    dtm = np.array(dtm)
    if dtm.sum() == 0:  # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = model_stats.get_doc_lengths(dtm)
    p_t = model_stats.get_marginal_topic_distrib(model.doc_topic_, doc_lengths)

    w_distinct = model_stats.get_word_distinctiveness(model.topic_word_, p_t)

    assert w_distinct.shape == (dtm.shape[1],)
    assert all(v >= 0 for v in w_distinct)


@given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    n_topics=st.integers(2, 10))
def test_get_word_saliency(dtm, n_topics):
    dtm = np.array(dtm)
    if dtm.sum() == 0:  # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = model_stats.get_doc_lengths(dtm)

    w_sal = model_stats.get_word_saliency(model.topic_word_, model.doc_topic_, doc_lengths)
    assert w_sal.shape == (dtm.shape[1],)
    assert all(v >= 0 for v in w_sal)


@given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    n_topics=st.integers(2, 10),
    n_salient_words=st.integers(2, 10))
def test_get_most_or_least_salient_words(dtm, n_topics, n_salient_words):
    dtm = np.array(dtm)
    if dtm.sum() == 0:  # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    n_salient_words = min(n_salient_words, dtm.shape[1])

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = model_stats.get_doc_lengths(dtm)
    vocab = np.array([chr(65 + i) for i in range(dtm.shape[1])])  # this only works for few words

    most_salient = model_stats.get_most_salient_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths)
    least_salient = model_stats.get_least_salient_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths)
    assert most_salient.shape == least_salient.shape == (len(vocab),) == (dtm.shape[1],)
    assert all(a == b for a, b in zip(most_salient, least_salient[::-1]))

    most_salient_n = model_stats.get_most_salient_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths,
                                                        n=n_salient_words)
    least_salient_n = model_stats.get_least_salient_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths,
                                                          n=n_salient_words)
    assert most_salient_n.shape == least_salient_n.shape == (n_salient_words,)
    assert all(a == b for a, b in zip(most_salient_n, most_salient[:n_salient_words]))
    assert all(a == b for a, b in zip(least_salient_n, least_salient[:n_salient_words]))


@given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    n_topics=st.integers(2, 10),
    n_distinct_words=st.integers(2, 10))
def test_get_most_or_least_distinct_words(dtm, n_topics, n_distinct_words):
    dtm = np.array(dtm)
    if dtm.sum() == 0:  # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    n_distinct_words = min(n_distinct_words, dtm.shape[1])

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = model_stats.get_doc_lengths(dtm)
    vocab = np.array([chr(65 + i) for i in range(dtm.shape[1])])  # this only works for few words

    most_distinct = model_stats.get_most_distinct_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths)
    least_distinct = model_stats.get_least_distinct_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths)
    assert most_distinct.shape == least_distinct.shape == (len(vocab),) == (dtm.shape[1],)
    assert all(a == b for a, b in zip(most_distinct, least_distinct[::-1]))

    most_distinct_n = model_stats.get_most_distinct_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths,
                                                          n=n_distinct_words)
    least_distinct_n = model_stats.get_least_distinct_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths,
                                                            n=n_distinct_words)
    assert most_distinct_n.shape == least_distinct_n.shape == (n_distinct_words,)
    assert all(a == b for a, b in zip(most_distinct_n, most_distinct[:n_distinct_words]))
    assert all(a == b for a, b in zip(least_distinct_n, least_distinct[:n_distinct_words]))


@given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    n_topics=st.integers(2, 10),
    lambda_=st.floats(0, 1))
def test_get_topic_word_relevance(dtm, n_topics, lambda_):
    dtm = np.array(dtm)
    if dtm.sum() == 0:  # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = model_stats.get_doc_lengths(dtm)

    rel_mat = model_stats.get_topic_word_relevance(model.topic_word_, model.doc_topic_, doc_lengths, lambda_)

    assert rel_mat.shape == (n_topics, dtm.shape[1])
    assert all(isinstance(x, float) and not np.isnan(x) for x in rel_mat.flatten())


@given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    n_topics=st.integers(2, 10),
    lambda_=st.floats(0, 1),
    n_relevant_words=st.integers(2, 10))
def test_get_most_or_least_relevant_words_for_topic(dtm, n_topics, lambda_, n_relevant_words):
    dtm = np.array(dtm)
    if dtm.sum() == 0:  # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    n_relevant_words = min(n_relevant_words, dtm.shape[1])
    topic = random.randint(0, n_topics - 1)

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    vocab = np.array([chr(65 + i) for i in range(dtm.shape[1])])  # this only works for few words
    doc_lengths = model_stats.get_doc_lengths(dtm)

    rel_mat = model_stats.get_topic_word_relevance(model.topic_word_, model.doc_topic_, doc_lengths, lambda_)

    most_rel = model_stats.get_most_relevant_words_for_topic(vocab, rel_mat, topic)
    least_rel = model_stats.get_least_relevant_words_for_topic(vocab, rel_mat, topic)
    assert most_rel.shape == least_rel.shape == (len(vocab),) == (dtm.shape[1],)
    assert all(a == b for a, b in zip(most_rel, least_rel[::-1]))

    most_rel_n = model_stats.get_most_relevant_words_for_topic(vocab, rel_mat, topic, n=n_relevant_words)
    least_rel_n = model_stats.get_least_relevant_words_for_topic(vocab, rel_mat, topic, n=n_relevant_words)
    assert most_rel_n.shape == least_rel_n.shape == (n_relevant_words,)
    assert all(a == b for a, b in zip(most_rel_n, most_rel[:n_relevant_words]))
    assert all(a == b for a, b in zip(least_rel_n, least_rel[:n_relevant_words]))


@given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    n_topics=st.integers(2, 10),
    lambda_=st.floats(0, 1))
def test_generate_topic_labels_from_top_words(dtm, n_topics, lambda_):
    dtm = np.array(dtm)
    if dtm.sum() == 0:  # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    vocab = np.array([chr(65 + i) for i in range(dtm.shape[1])])  # this only works for few words
    doc_lengths = model_stats.get_doc_lengths(dtm)

    topic_labels = model_stats.generate_topic_labels_from_top_words(model.topic_word_, model.doc_topic_,
                                                                    doc_lengths, vocab, lambda_=lambda_)
    assert isinstance(topic_labels, list)
    assert len(topic_labels) == n_topics

    for i, l in enumerate(topic_labels):
        assert isinstance(l, six.string_types)
        parts = l.split('_')
        assert len(parts) >= 2
        assert int(parts[0]) == i + 1
        assert all(w in vocab for w in parts[1:])

    topic_labels_2 = model_stats.generate_topic_labels_from_top_words(model.topic_word_, model.doc_topic_,
                                                                      doc_lengths, vocab, lambda_=lambda_,
                                                                      n_words=2)
    assert isinstance(topic_labels_2, list)
    assert len(topic_labels_2) == n_topics

    for i, l in enumerate(topic_labels_2):
        assert isinstance(l, six.string_types)
        parts = l.split('_')
        assert len(parts) == 3
        assert int(parts[0]) == i + 1
        assert all(w in vocab for w in parts[1:])


def test_filter_topics():
    vocab = np.array(['abc', 'abcd', 'cde', 'efg', 'xyz'])
    distrib = np.array([
        [0.6, 0.3, 0.05, 0.025, 0.025],   # abc, abcd, cde
        [0.2, 0.1, 0.3, 0.3, 0.1],        # cde, efg, abc
        [0.05, 0.05, 0.2, 0.3, 0.4],      # xyz, efg, cde
    ])

    # simple exact match within top list of words
    topic_ind = model_stats.filter_topics('abc', vocab, distrib, top_n=3)
    assert list(topic_ind) == [0, 1]
    topic_ind = model_stats.filter_topics('xyz', vocab, distrib, top_n=3)
    assert list(topic_ind) == [2]

    # simple RE pattern match within top list of words
    topic_ind = model_stats.filter_topics(r'^ab', vocab, distrib, top_n=3, match='regex')
    assert list(topic_ind) == [0, 1]
    topic_ind = model_stats.filter_topics(r'(cd$|^x)', vocab, distrib, top_n=3, match='regex')
    assert list(topic_ind) == [0, 2]

    # simple glob pattern match within top list of words
    topic_ind = model_stats.filter_topics('ab*', vocab, distrib, top_n=3, match='glob')
    assert list(topic_ind) == [0, 1]
    topic_ind = model_stats.filter_topics('ab?d', vocab, distrib, top_n=3, match='glob')
    assert list(topic_ind) == [0]

    # multiple matches within top list of words
    topic_ind = model_stats.filter_topics(['abcd', 'xyz'], vocab, distrib, top_n=3)
    assert list(topic_ind) == [0, 2]
    topic_ind = model_stats.filter_topics(['abcd', 'xyz'], vocab, distrib, top_n=3, cond='all')
    assert list(topic_ind) == []
    topic_ind = model_stats.filter_topics(['cde', 'efg'], vocab, distrib, top_n=3, cond='all')
    assert list(topic_ind) == [1, 2]
    topic_ind = model_stats.filter_topics(['*cd', 'ef*'], vocab, distrib, top_n=3, match='glob', cond='all')
    assert list(topic_ind) == [1, 2]

    # simple exact threshold match
    topic_ind = model_stats.filter_topics('abc', vocab, distrib, thresh=0.6)
    assert list(topic_ind) == [0]
    topic_ind = model_stats.filter_topics('abc', vocab, distrib, thresh=0.2)
    assert list(topic_ind) == [0, 1]
    topic_ind = model_stats.filter_topics('xyz', vocab, distrib, thresh=0.5)
    assert list(topic_ind) == []

    # simple RE pattern threshold match
    topic_ind = model_stats.filter_topics(r'^ab', vocab, distrib, thresh=0.2, match='regex')
    assert list(topic_ind) == [0, 1]

    # multiple matches within top list of words
    topic_ind = model_stats.filter_topics(['abc', 'xyz'], vocab, distrib, thresh=0.4)
    assert list(topic_ind) == [0, 2]

    # simple match with combination of top words list and threshold
    topic_ind = model_stats.filter_topics('abc', vocab, distrib, top_n=1, thresh=0.6)
    assert list(topic_ind) == [0]
    topic_ind = model_stats.filter_topics('abc', vocab, distrib, top_n=3, thresh=0.6)
    assert list(topic_ind) == [0]
    topic_ind = model_stats.filter_topics('abc', vocab, distrib, top_n=1, thresh=0.9)
    assert list(topic_ind) == []
    topic_ind = model_stats.filter_topics('c*', vocab, distrib, top_n=3, thresh=0.3, match='glob')
    assert list(topic_ind) == [1]
    topic_ind = model_stats.filter_topics('*c*', vocab, distrib, top_n=3, thresh=0.3, match='glob')
    assert list(topic_ind) == [0, 1]
    topic_ind = model_stats.filter_topics('c*', vocab, distrib, top_n=3, thresh=0.3, match='glob', glob_method='search')
    assert list(topic_ind) == [0, 1]

    # multiple matches with combination of top words list and threshold
    topic_ind = model_stats.filter_topics(['*cd', 'ef*'], vocab, distrib, top_n=3, thresh=0.3, match='glob', cond='all')
    assert list(topic_ind) == [1]

    # return words and matches
    topic_ind, top_words, matches = model_stats.filter_topics([r'cd$', r'^x'], vocab, distrib, top_n=3, match='regex',
                                                              return_words_and_matches=True)
    assert list(topic_ind) == [0, 2]
    assert len(top_words) == 2
    assert list(top_words[0]) == ['abc', 'abcd', 'cde']
    assert list(top_words[1]) == ['xyz', 'efg', 'cde']
    assert list(matches[0]) == [False, True, False]
    assert list(matches[1]) == [True, False, False]

