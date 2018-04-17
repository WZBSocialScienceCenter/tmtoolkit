from __future__ import division

import random
import os
import string
import math
import itertools

import six
import pytest
import hypothesis.strategies as st
from hypothesis import given

import numpy as np
from scipy.sparse.coo import coo_matrix
import lda
import gensim
from sklearn.decomposition.online_lda import LatentDirichletAllocation
import PIL

from tmtoolkit import lda_utils


# common

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
            lda_utils.common.top_n_from_distribution(distrib, n)
    else:
        if n < 1 or n > distrib.shape[1]:
            with pytest.raises(ValueError):
                lda_utils.common.top_n_from_distribution(distrib, n)
        else:
            df = lda_utils.common.top_n_from_distribution(distrib, n)

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
            lda_utils.common.top_words_for_topics(topic_word_distrib, top_n, vocab)
        return

    if top_n < 1 or top_n > topic_word_distrib.shape[1]:
        with pytest.raises(ValueError):
            lda_utils.common.top_words_for_topics(topic_word_distrib, top_n, vocab)
        return

    top_words = lda_utils.common.top_words_for_topics(topic_word_distrib, top_n, vocab)
    assert isinstance(top_words, list)
    assert len(top_words) == topic_word_distrib.shape[0]
    assert all(l == top_n for l in map(len, top_words))
    assert all(w in vocab for w in sum(map(list, top_words), []))

    top_words = lda_utils.common.top_words_for_topics(topic_word_distrib, top_n)     # no vocab -> return word indices
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

    top_words = lda_utils.common.top_words_for_topics(distrib, 2, vocab)
    assert len(top_words) == len(distrib)
    top_words_lists = list(map(list, top_words))
    assert top_words_lists[0] == ['a', 'b']
    assert top_words_lists[1] == ['b', 'c']
    assert top_words_lists[2] in (['a', 'c'], ['c', 'a'])

    top_words = lda_utils.common.top_words_for_topics(distrib, 2)   # no vocab -> return word indices
    assert len(top_words) == len(distrib)
    top_words_lists = list(map(list, top_words))
    assert top_words_lists[0] == [0, 1]
    assert top_words_lists[1] == [1, 2]
    assert top_words_lists[2] in ([0, 2], [2, 0])


def test_save_load_ldamodel_pickle():
    pfile = 'tests/data/test_pickle_unpickle_ldamodel.pickle'

    dtm = np.array([[0, 1], [2, 3], [4, 5], [6, 0]])
    doc_labels = ['doc_' + str(i) for i in range(dtm.shape[0])]
    vocab = ['word_' + str(i) for i in range(dtm.shape[1])]

    model = lda.LDA(2, n_iter=1)
    model.fit(dtm)

    lda_utils.common.save_ldamodel_to_pickle(pfile, model, vocab, doc_labels)

    unpickled = lda_utils.common.load_ldamodel_from_pickle(pfile)

    assert np.array_equal(model.doc_topic_, unpickled['model'].doc_topic_)
    assert np.array_equal(model.topic_word_, unpickled['model'].topic_word_)
    assert vocab == unpickled['vocab']
    assert doc_labels == unpickled['doc_labels']


@given(n_param_sets=st.integers(0, 10), n_params=st.integers(1, 10), n_metrics=st.integers(1, 10))
def test_results_by_parameter_single_validation(n_param_sets, n_params, n_metrics):
    # TODO: implement a better test here

    param_names = ['param' + str(i) for i in range(n_params)]
    metric_names = ['metric' + str(i) for i in range(n_metrics)]
    res = []
    for _ in range(n_param_sets):
        param_set = dict(zip(param_names, np.random.randint(0, 100, n_params)))
        metric_results = dict(zip(metric_names, np.random.uniform(0, 1, n_metrics)))
        res.append((param_set, metric_results))

    p = random.choice(param_names)
    by_param = lda_utils.common.results_by_parameter(res, p)
    assert len(res) == len(by_param)
    assert all(x == 2 for x in map(len, by_param))


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
            lda_utils.common.get_doc_lengths(dtm)
    else:
        doc_lengths = lda_utils.common.get_doc_lengths(dtm)
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
            lda_utils.common.get_doc_frequencies(dtm)
    else:
        n_docs = dtm.shape[0]

        df_abs = lda_utils.common.get_doc_frequencies(dtm)
        assert isinstance(df_abs, np.ndarray)
        assert df_abs.ndim == 1
        assert df_abs.shape == (dtm_arr.shape[1],)
        assert all([0 <= v <= n_docs for v in df_abs])

        df_rel = lda_utils.common.get_doc_frequencies(dtm, proportions=True)
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

    df = lda_utils.common.get_doc_frequencies(dtm)

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
            lda_utils.common.get_codoc_frequencies(dtm, proportions=proportions)
        return

    n_docs, n_vocab = dtm.shape

    if n_vocab < 2:
        with pytest.raises(ValueError):
            lda_utils.common.get_codoc_frequencies(dtm, proportions=proportions)
        return

    df = lda_utils.common.get_codoc_frequencies(dtm, proportions=proportions)
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

    df = lda_utils.common.get_codoc_frequencies(dtm)

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
            lda_utils.common.get_term_frequencies(dtm)
    else:
        tf = lda_utils.common.get_term_frequencies(dtm)
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
            lda_utils.common.get_term_proportions(dtm)
    else:
        tp = lda_utils.common.get_term_proportions(dtm)
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
    if dtm.sum() == 0:   # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = lda_utils.common.get_doc_lengths(dtm)
    marginal_topic_distr = lda_utils.common.get_marginal_topic_distrib(model.doc_topic_, doc_lengths)

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
    if dtm.sum() == 0:   # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = lda_utils.common.get_doc_lengths(dtm)
    p_t = lda_utils.common.get_marginal_topic_distrib(model.doc_topic_, doc_lengths)

    p_w = lda_utils.common.get_marginal_word_distrib(model.topic_word_, p_t)
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
    if dtm.sum() == 0:   # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = lda_utils.common.get_doc_lengths(dtm)
    p_t = lda_utils.common.get_marginal_topic_distrib(model.doc_topic_, doc_lengths)

    w_distinct = lda_utils.common.get_word_distinctiveness(model.topic_word_, p_t)

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
    if dtm.sum() == 0:   # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = lda_utils.common.get_doc_lengths(dtm)

    w_sal = lda_utils.common.get_word_saliency(model.topic_word_, model.doc_topic_, doc_lengths)
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
    if dtm.sum() == 0:   # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    n_salient_words = min(n_salient_words, dtm.shape[1])

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = lda_utils.common.get_doc_lengths(dtm)
    vocab = np.array([chr(65 + i) for i in range(dtm.shape[1])])   # this only works for few words

    most_salient = lda_utils.common.get_most_salient_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths)
    least_salient = lda_utils.common.get_least_salient_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths)
    assert most_salient.shape == least_salient.shape == (len(vocab),) == (dtm.shape[1],)
    assert all(a == b for a, b in zip(most_salient, least_salient[::-1]))

    most_salient_n = lda_utils.common.get_most_salient_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths,
                                                             n=n_salient_words)
    least_salient_n = lda_utils.common.get_least_salient_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths,
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
    if dtm.sum() == 0:   # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    n_distinct_words = min(n_distinct_words, dtm.shape[1])

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = lda_utils.common.get_doc_lengths(dtm)
    vocab = np.array([chr(65 + i) for i in range(dtm.shape[1])])   # this only works for few words

    most_distinct = lda_utils.common.get_most_distinct_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths)
    least_distinct = lda_utils.common.get_least_distinct_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths)
    assert most_distinct.shape == least_distinct.shape == (len(vocab),) == (dtm.shape[1],)
    assert all(a == b for a, b in zip(most_distinct, least_distinct[::-1]))

    most_distinct_n = lda_utils.common.get_most_distinct_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths,
                                                               n=n_distinct_words)
    least_distinct_n = lda_utils.common.get_least_distinct_words(vocab, model.topic_word_, model.doc_topic_, doc_lengths,
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
    if dtm.sum() == 0:   # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    doc_lengths = lda_utils.common.get_doc_lengths(dtm)

    rel_mat = lda_utils.common.get_topic_word_relevance(model.topic_word_, model.doc_topic_, doc_lengths, lambda_)

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
    if dtm.sum() == 0:   # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    n_relevant_words = min(n_relevant_words, dtm.shape[1])
    topic = random.randint(0, n_topics-1)

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    vocab = np.array([chr(65 + i) for i in range(dtm.shape[1])])  # this only works for few words
    doc_lengths = lda_utils.common.get_doc_lengths(dtm)

    rel_mat = lda_utils.common.get_topic_word_relevance(model.topic_word_, model.doc_topic_, doc_lengths, lambda_)

    most_rel = lda_utils.common.get_most_relevant_words_for_topic(vocab, rel_mat, topic)
    least_rel = lda_utils.common.get_least_relevant_words_for_topic(vocab, rel_mat, topic)
    assert most_rel.shape == least_rel.shape == (len(vocab),) == (dtm.shape[1],)
    assert all(a == b for a, b in zip(most_rel, least_rel[::-1]))

    most_rel_n = lda_utils.common.get_most_relevant_words_for_topic(vocab, rel_mat, topic, n=n_relevant_words)
    least_rel_n = lda_utils.common.get_least_relevant_words_for_topic(vocab, rel_mat, topic, n=n_relevant_words)
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
    if dtm.sum() == 0:   # assure that we have at least one word in the DTM
        dtm[0, 0] = 1

    model = lda.LDA(n_topics, 1)
    model.fit(dtm)

    vocab = np.array([chr(65 + i) for i in range(dtm.shape[1])])  # this only works for few words
    doc_lengths = lda_utils.common.get_doc_lengths(dtm)

    topic_labels = lda_utils.common.generate_topic_labels_from_top_words(model.topic_word_, model.doc_topic_,
                                                                         doc_lengths, vocab, lambda_=lambda_)
    assert isinstance(topic_labels, list)
    assert len(topic_labels) == n_topics

    for i, l in enumerate(topic_labels):
        assert isinstance(l, six.string_types)
        parts = l.split('_')
        assert len(parts) >= 2
        assert int(parts[0]) == i+1
        assert all(w in vocab for w in parts[1:])

    topic_labels_2 = lda_utils.common.generate_topic_labels_from_top_words(model.topic_word_, model.doc_topic_,
                                                                           doc_lengths, vocab, lambda_=lambda_,
                                                                           n_words=2)
    assert isinstance(topic_labels_2, list)
    assert len(topic_labels_2) == n_topics

    for i, l in enumerate(topic_labels_2):
        assert isinstance(l, six.string_types)
        parts = l.split('_')
        assert len(parts) == 3
        assert int(parts[0]) == i+1
        assert all(w in vocab for w in parts[1:])


# parallel models and evaluation lda

EVALUATION_TEST_DTM = np.array([
        [1, 2, 3, 0, 0],
        [0, 0, 2, 2, 0],
        [3, 0, 1, 1, 3],
        [2, 1, 0, 2, 5],
])

EVALUATION_TEST_DTM_MULTI = {
    'test1': EVALUATION_TEST_DTM,
    'test2': np.array([
        [1, 0, 1, 0, 3],
        [0, 0, 2, 5, 0],
        [3, 0, 1, 2, 0],
        [2, 1, 3, 2, 4],
        [0, 0, 0, 1, 1],
        [3, 2, 5, 1, 1],
    ]),
    'test3': np.array([
        [0, 1, 3, 0, 4, 3],
        [3, 0, 2, 0, 0, 0],
        [0, 2, 1, 3, 3, 0],
        [2, 1, 5, 4, 0, 1],
    ]),
}


def test_compute_models_parallel_lda_multi_vs_singleproc():
    passed_params = {'n_topics', 'n_iter', 'random_state'}
    varying_params = [dict(n_topics=k) for k in range(2, 5)]
    const_params = dict(n_iter=3, random_state=1)

    models = lda_utils.tm_lda.compute_models_parallel(EVALUATION_TEST_DTM, varying_params, const_params)
    assert len(models) == len(varying_params)

    for param_set, model in models:
        assert set(param_set.keys()) == passed_params
        assert isinstance(model, lda.LDA)
        assert isinstance(model.doc_topic_, np.ndarray)
        assert isinstance(model.topic_word_, np.ndarray)

    models_singleproc = lda_utils.tm_lda.compute_models_parallel(EVALUATION_TEST_DTM, varying_params, const_params,
                                                                 n_max_processes=1)

    assert len(models_singleproc) == len(models)
    for param_set2, model2 in models_singleproc:
        for x, y in models:
            if x == param_set2:
                param_set1, model1 = x, y
                break
        else:
            assert False

        assert np.allclose(model1.doc_topic_, model2.doc_topic_)
        assert np.allclose(model1.topic_word_, model2.topic_word_)


def test_compute_models_parallel_lda_multiple_docs():
    # 1 doc, no varying params
    const_params = dict(n_topics=3, n_iter=3, random_state=1)
    models = lda_utils.tm_lda.compute_models_parallel(EVALUATION_TEST_DTM, constant_parameters=const_params)
    assert len(models) == 1
    assert type(models) is list
    assert len(models[0]) == 2
    param1, model1 = models[0]
    assert param1 == const_params
    assert isinstance(model1, lda.LDA)
    assert isinstance(model1.doc_topic_, np.ndarray)
    assert isinstance(model1.topic_word_, np.ndarray)

    # 1 *named* doc, some varying params
    passed_params = {'n_topics', 'n_iter', 'random_state'}
    const_params = dict(n_iter=3, random_state=1)
    varying_params = [dict(n_topics=k) for k in range(2, 5)]
    docs = {'test1': EVALUATION_TEST_DTM}
    models = lda_utils.tm_lda.compute_models_parallel(docs, varying_params,
                                                      constant_parameters=const_params)
    assert len(models) == len(docs)
    assert isinstance(models, dict)
    assert set(models.keys()) == {'test1'}

    param_match = False
    for d, m in models.items():
        assert d == 'test1'
        assert len(m) == len(varying_params)
        for param_set, model in m:
            assert set(param_set.keys()) == passed_params
            assert isinstance(model, lda.LDA)
            assert isinstance(model.doc_topic_, np.ndarray)
            assert isinstance(model.topic_word_, np.ndarray)

            if param_set == param1:
                assert np.allclose(model.doc_topic_, model1.doc_topic_)
                assert np.allclose(model.topic_word_, model1.topic_word_)
                param_match = True

    assert param_match

    # n docs, no varying params
    const_params = dict(n_topics=3, n_iter=3, random_state=1)
    models = lda_utils.tm_lda.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, constant_parameters=const_params)
    assert len(models) == len(EVALUATION_TEST_DTM_MULTI)
    assert isinstance(models, dict)
    assert set(models.keys()) == set(EVALUATION_TEST_DTM_MULTI.keys())

    for d, m in models.items():
        assert len(m) == 1
        for param_set, model in m:
            assert set(param_set.keys()) == set(const_params.keys())
            assert isinstance(model, lda.LDA)
            assert isinstance(model.doc_topic_, np.ndarray)
            assert isinstance(model.topic_word_, np.ndarray)

    # n docs, some varying params
    passed_params = {'n_topics', 'n_iter', 'random_state'}
    const_params = dict(n_iter=3, random_state=1)
    varying_params = [dict(n_topics=k) for k in range(2, 5)]
    models = lda_utils.tm_lda.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, varying_params,
                                                      constant_parameters=const_params)
    assert len(models) == len(EVALUATION_TEST_DTM_MULTI)
    assert isinstance(models, dict)
    assert set(models.keys()) == set(EVALUATION_TEST_DTM_MULTI.keys())

    for d, m in models.items():
        assert len(m) == len(varying_params)
        for param_set, model in m:
            assert set(param_set.keys()) == passed_params
            assert isinstance(model, lda.LDA)
            assert isinstance(model.doc_topic_, np.ndarray)
            assert isinstance(model.topic_word_, np.ndarray)


def test_evaluation_lda_all_metrics_multi_vs_singleproc():
    passed_params = {'n_topics', 'alpha', 'n_iter', 'refresh', 'random_state'}
    varying_params = [dict(n_topics=k, alpha=1/k) for k in range(2, 5)]
    const_params = dict(n_iter=10, refresh=1, random_state=1)

    eval_res = lda_utils.tm_lda.evaluate_topic_models(EVALUATION_TEST_DTM, varying_params, const_params)

    assert len(eval_res) == len(varying_params)

    for param_set, metric_results in eval_res:
        assert set(param_set.keys()) == passed_params
        assert set(metric_results.keys()) == set(lda_utils.tm_lda.AVAILABLE_METRICS)

        assert metric_results['loglikelihood'] < 0
        assert 0 <= metric_results['cao_juan_2009'] <= 1
        assert 0 <= metric_results['arun_2010']

        if 'griffiths_2004' in lda_utils.tm_lda.AVAILABLE_METRICS:  # only if gmpy2 is installed
            assert metric_results['griffiths_2004'] < 0

    eval_res_singleproc = lda_utils.tm_lda.evaluate_topic_models(EVALUATION_TEST_DTM, varying_params, const_params,
                                                                 n_max_processes=1)
    assert len(eval_res_singleproc) == len(eval_res)
    for param_set2, metric_results2 in eval_res_singleproc:
        for x, y in eval_res:
            if x == param_set2:
                param_set1, metric_results1 = x, y
                break
        else:
            assert False

        assert metric_results1 == metric_results2


# parallel models and evaluation gensim


def test_evaluation_gensim_all_metrics():
    passed_params = {'num_topics', 'update_every', 'passes', 'iterations'}
    varying_params = [dict(num_topics=k) for k in range(2, 5)]
    const_params = dict(update_every=0, passes=1, iterations=1)

    eval_res = lda_utils.tm_gensim.evaluate_topic_models(EVALUATION_TEST_DTM, varying_params, const_params)

    assert len(eval_res) == len(varying_params)

    for param_set, metric_results in eval_res:
        assert set(param_set.keys()) == passed_params
        assert set(metric_results.keys()) == set(lda_utils.tm_gensim.AVAILABLE_METRICS)

        assert metric_results['perplexity'] > 0
        assert 0 <= metric_results['cao_juan_2009'] <= 1


def test_compute_models_parallel_gensim():
    passed_params = {'num_topics', 'update_every', 'passes', 'iterations'}
    varying_params = [dict(num_topics=k) for k in range(2, 5)]
    const_params = dict(update_every=0, passes=1, iterations=1)

    models = lda_utils.tm_gensim.compute_models_parallel(EVALUATION_TEST_DTM, varying_params, const_params)

    assert len(models) == len(varying_params)

    for param_set, model in models:
        assert set(param_set.keys()) == passed_params
        assert isinstance(model, gensim.models.LdaModel)
        assert isinstance(model.state.get_lambda(), np.ndarray)


def test_compute_models_parallel_gensim_multiple_docs():
    # 1 doc, no varying params
    const_params = dict(num_topics=3, update_every=0, passes=1, iterations=1)
    models = lda_utils.tm_gensim.compute_models_parallel(EVALUATION_TEST_DTM, constant_parameters=const_params)
    assert len(models) == 1
    assert type(models) is list
    assert len(models[0]) == 2
    param1, model1 = models[0]
    assert param1 == const_params
    assert isinstance(model1, gensim.models.LdaModel)
    assert isinstance(model1.state.get_lambda(), np.ndarray)

    # 1 *named* doc, some varying params
    passed_params = {'num_topics', 'update_every', 'passes', 'iterations'}
    const_params = dict(update_every=0, passes=1, iterations=1)
    varying_params = [dict(num_topics=k) for k in range(2, 5)]
    docs = {'test1': EVALUATION_TEST_DTM}
    models = lda_utils.tm_gensim.compute_models_parallel(docs, varying_params,
                                                         constant_parameters=const_params)
    assert len(models) == len(docs)
    assert isinstance(models, dict)
    assert set(models.keys()) == {'test1'}

    for d, m in models.items():
        assert d == 'test1'
        assert len(m) == len(varying_params)
        for param_set, model in m:
            assert set(param_set.keys()) == passed_params
            assert isinstance(model, gensim.models.LdaModel)
            assert isinstance(model.state.get_lambda(), np.ndarray)

    # n docs, no varying params
    const_params = dict(num_topics=3, update_every=0, passes=1, iterations=1)
    models = lda_utils.tm_gensim.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, constant_parameters=const_params)
    assert len(models) == len(EVALUATION_TEST_DTM_MULTI)
    assert isinstance(models, dict)
    assert set(models.keys()) == set(EVALUATION_TEST_DTM_MULTI.keys())

    for d, m in models.items():
        assert len(m) == 1
        for param_set, model in m:
            assert set(param_set.keys()) == set(const_params.keys())
            assert isinstance(model, gensim.models.LdaModel)
            assert isinstance(model.state.get_lambda(), np.ndarray)

    # n docs, some varying params
    passed_params = {'num_topics', 'update_every', 'passes', 'iterations'}
    const_params = dict(update_every=0, passes=1, iterations=1)
    varying_params = [dict(num_topics=k) for k in range(2, 5)]
    models = lda_utils.tm_gensim.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, varying_params,
                                                         constant_parameters=const_params)
    assert len(models) == len(EVALUATION_TEST_DTM_MULTI)
    assert isinstance(models, dict)
    assert set(models.keys()) == set(EVALUATION_TEST_DTM_MULTI.keys())

    for d, m in models.items():
        assert len(m) == len(varying_params)
        for param_set, model in m:
            assert set(param_set.keys()) == passed_params
            assert isinstance(model, gensim.models.LdaModel)
            assert isinstance(model.state.get_lambda(), np.ndarray)


# parallel models and evaluation sklearn


def test_evaluation_sklearn_all_metrics():
    passed_params = {'n_components', 'learning_method', 'evaluate_every', 'max_iter', 'n_jobs'}
    varying_params = [dict(n_components=k) for k in range(2, 5)]
    const_params = dict(learning_method='batch', evaluate_every=1, max_iter=3, n_jobs=1)

    eval_res = lda_utils.tm_sklearn.evaluate_topic_models(EVALUATION_TEST_DTM, varying_params, const_params)

    assert len(eval_res) == len(varying_params)

    for param_set, metric_results in eval_res:
        assert set(param_set.keys()) == passed_params
        assert set(metric_results.keys()) == set(lda_utils.tm_sklearn.AVAILABLE_METRICS)

        assert metric_results['perplexity'] > 0
        assert 0 <= metric_results['cao_juan_2009'] <= 1
        assert 0 <= metric_results['arun_2010']


def test_compute_models_parallel_sklearn():
    passed_params = {'n_components', 'learning_method', 'evaluate_every', 'max_iter', 'n_jobs'}
    varying_params = [dict(n_components=k) for k in range(2, 5)]
    const_params = dict(learning_method='batch', evaluate_every=1, max_iter=3, n_jobs=1)

    models = lda_utils.tm_sklearn.compute_models_parallel(EVALUATION_TEST_DTM, varying_params, const_params)

    assert len(models) == len(varying_params)

    for param_set, model in models:
        assert set(param_set.keys()) == passed_params
        assert isinstance(model, LatentDirichletAllocation)
        assert isinstance(model.components_, np.ndarray)


def test_compute_models_parallel_sklearn_multiple_docs():
    # 1 doc, no varying params
    const_params = dict(n_components=3, learning_method='batch', evaluate_every=1, max_iter=3, n_jobs=1)
    models = lda_utils.tm_sklearn.compute_models_parallel(EVALUATION_TEST_DTM, constant_parameters=const_params)
    assert len(models) == 1
    assert type(models) is list
    assert len(models[0]) == 2
    param1, model1 = models[0]
    assert param1 == const_params
    assert isinstance(model1, LatentDirichletAllocation)
    assert isinstance(model1.components_, np.ndarray)

    # 1 *named* doc, some varying params
    passed_params = {'n_components', 'learning_method', 'evaluate_every', 'max_iter', 'n_jobs'}
    const_params = dict(learning_method='batch', evaluate_every=1, max_iter=3, n_jobs=1)
    varying_params = [dict(n_components=k) for k in range(2, 5)]
    docs = {'test1': EVALUATION_TEST_DTM}
    models = lda_utils.tm_sklearn.compute_models_parallel(docs, varying_params,
                                                          constant_parameters=const_params)
    assert len(models) == len(docs)
    assert isinstance(models, dict)
    assert set(models.keys()) == {'test1'}

    for d, m in models.items():
        assert d == 'test1'
        assert len(m) == len(varying_params)
        for param_set, model in m:
            assert set(param_set.keys()) == passed_params
            assert isinstance(model, LatentDirichletAllocation)
            assert isinstance(model.components_, np.ndarray)

    # n docs, no varying params
    const_params = dict(n_components=3, learning_method='batch', evaluate_every=1, max_iter=3, n_jobs=1)
    models = lda_utils.tm_sklearn.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, constant_parameters=const_params)
    assert len(models) == len(EVALUATION_TEST_DTM_MULTI)
    assert isinstance(models, dict)
    assert set(models.keys()) == set(EVALUATION_TEST_DTM_MULTI.keys())

    for d, m in models.items():
        assert len(m) == 1
        for param_set, model in m:
            assert set(param_set.keys()) == set(const_params.keys())
            assert isinstance(model, LatentDirichletAllocation)
            assert isinstance(model.components_, np.ndarray)

    # n docs, some varying params
    passed_params = {'n_components', 'learning_method', 'evaluate_every', 'max_iter', 'n_jobs'}
    const_params = dict(learning_method='batch', evaluate_every=1, max_iter=3, n_jobs=1)
    varying_params = [dict(n_components=k) for k in range(2, 5)]
    models = lda_utils.tm_sklearn.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, varying_params,
                                                          constant_parameters=const_params)
    assert len(models) == len(EVALUATION_TEST_DTM_MULTI)
    assert isinstance(models, dict)
    assert set(models.keys()) == set(EVALUATION_TEST_DTM_MULTI.keys())

    for d, m in models.items():
        assert len(m) == len(varying_params)
        for param_set, model in m:
            assert set(param_set.keys()) == passed_params
            assert isinstance(model, LatentDirichletAllocation)
            assert isinstance(model.components_, np.ndarray)

# visualize


try:
    from wordcloud import WordCloud

    def test_generate_wordclouds_for_topic_words():
        py3file = '.py3' if six.PY3 else ''
        data = lda_utils.common.load_ldamodel_from_pickle('tests/data/tiny_model_reuters_5_topics%s.pickle' % py3file)
        model = data['model']
        vocab = data['vocab']

        phi = model.topic_word_
        assert phi.shape == (5, len(vocab))

        topic_word_clouds = lda_utils.visualize.generate_wordclouds_for_topic_words(phi, vocab, 10)
        assert len(topic_word_clouds) == 5
        assert set(topic_word_clouds.keys()) == set('topic_%d' % i for i in range(1, 6))
        assert all(isinstance(wc, PIL.Image.Image) for wc in topic_word_clouds.values())

        topic_word_clouds = lda_utils.visualize.generate_wordclouds_for_topic_words(phi, vocab, 10,
                                                                                    which_topics=('topic_1', 'topic_2'),
                                                                                    return_images=False,
                                                                                    width=640, height=480)
        assert set(topic_word_clouds.keys()) == {'topic_1', 'topic_2'}
        assert all(isinstance(wc, WordCloud) for wc in topic_word_clouds.values())
        assert all(wc.width == 640 and wc.height == 480 for wc in topic_word_clouds.values())


    def test_generate_wordclouds_for_document_topics():
        py3file = '.py3' if six.PY3 else ''
        data = lda_utils.common.load_ldamodel_from_pickle('tests/data/tiny_model_reuters_5_topics%s.pickle' % py3file)
        model = data['model']
        doc_labels = data['doc_labels']

        theta = model.doc_topic_
        assert theta.shape == (len(doc_labels), 5)

        doc_topic_clouds = lda_utils.visualize.generate_wordclouds_for_document_topics(theta, doc_labels, 3)
        assert len(doc_topic_clouds) == len(doc_labels)
        assert set(doc_topic_clouds.keys()) == set(doc_labels)
        assert all(isinstance(wc, PIL.Image.Image) for wc in doc_topic_clouds.values())

        which_docs = doc_labels[:2]
        assert len(which_docs) == 2
        doc_topic_clouds = lda_utils.visualize.generate_wordclouds_for_document_topics(theta, doc_labels, 3,
                                                                                       which_documents=which_docs,
                                                                                       return_images=False,
                                                                                       width=640, height=480)
        assert set(doc_topic_clouds.keys()) == set(which_docs)
        assert all(isinstance(wc, WordCloud) for wc in doc_topic_clouds.values())
        assert all(wc.width == 640 and wc.height == 480 for wc in doc_topic_clouds.values())


    def test_write_wordclouds_to_folder(tmpdir):
        path = tmpdir.mkdir('wordclouds').dirname

        py3file = '.py3' if six.PY3 else ''
        data = lda_utils.common.load_ldamodel_from_pickle('tests/data/tiny_model_reuters_5_topics%s.pickle' % py3file)
        model = data['model']
        vocab = data['vocab']

        phi = model.topic_word_
        assert phi.shape == (5, len(vocab))

        topic_word_clouds = lda_utils.visualize.generate_wordclouds_for_topic_words(phi, vocab, 10)

        lda_utils.visualize.write_wordclouds_to_folder(topic_word_clouds, path, 'cloud_{label}.png')

        for label in topic_word_clouds.keys():
            assert os.path.exists(os.path.join(path, 'cloud_{label}.png'.format(label=label)))
except:
    # wordcloud module not found
    pass