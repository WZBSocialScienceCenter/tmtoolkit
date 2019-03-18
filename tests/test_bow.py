import itertools
import math

import numpy as np
import pytest
from hypothesis import given, strategies as st
from scipy.sparse import coo_matrix
import gensim

from tmtoolkit import bow


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=1))
def test_get_doc_lengths(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm = np.array(dtm)
        dtm_arr = dtm

    if dtm_arr.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.get_doc_lengths(dtm)
    else:
        doc_lengths = bow.bow_stats.get_doc_lengths(dtm)
        assert doc_lengths.ndim == 1
        assert doc_lengths.shape == (dtm_arr.shape[0],)
        assert doc_lengths.tolist() == [sum(row) for row in dtm_arr]


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=1))
def test_get_doc_frequencies(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm = np.array(dtm)
        dtm_arr = dtm

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.get_doc_frequencies(dtm)
    else:
        n_docs = dtm.shape[0]

        df_abs = bow.bow_stats.get_doc_frequencies(dtm)
        assert isinstance(df_abs, np.ndarray)
        assert df_abs.ndim == 1
        assert df_abs.shape == (dtm_arr.shape[1],)
        assert all([0 <= v <= n_docs for v in df_abs])

        df_rel = bow.bow_stats.get_doc_frequencies(dtm, proportions=True)
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

    df = bow.bow_stats.get_doc_frequencies(dtm)

    assert df.tolist() == [1, 3, 1, 2, 1]


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=1),
    proportions=st.booleans())
def test_get_codoc_frequencies(dtm, matrix_type, proportions):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
    else:
        dtm = np.array(dtm)

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.get_codoc_frequencies(dtm, proportions=proportions)
        return

    n_docs, n_vocab = dtm.shape

    if n_vocab < 2:
        with pytest.raises(ValueError):
            bow.bow_stats.get_codoc_frequencies(dtm, proportions=proportions)
        return

    df = bow.bow_stats.get_codoc_frequencies(dtm, proportions=proportions)
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

    df = bow.bow_stats.get_codoc_frequencies(dtm)

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
    matrix_type=st.integers(min_value=0, max_value=1))
def test_get_term_frequencies(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm = np.array(dtm)
        dtm_arr = dtm

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.get_term_frequencies(dtm)
    else:
        tf = bow.bow_stats.get_term_frequencies(dtm)
        assert tf.ndim == 1
        assert tf.shape == (dtm_arr.shape[1],)
        assert tf.tolist() == [sum(row) for row in dtm_arr.T]


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=1))
def test_get_term_proportions(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
        dtm_flat = dtm.A.flatten()
    else:
        dtm = np.array(dtm)
        dtm_arr = dtm
        dtm_flat = dtm.flatten()

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.get_term_proportions(dtm)
    else:
        if dtm.sum() == 0:
            with pytest.raises(ValueError):
                bow.bow_stats.get_term_proportions(dtm)
        else:
            tp = bow.bow_stats.get_term_proportions(dtm)
            assert tp.ndim == 1
            assert tp.shape == (dtm_arr.shape[1],)

            if len(dtm_flat) > 0:
                assert np.isclose(tp.sum(), 1.0)
                assert all(0 <= v <= 1 for v in tp)


@given(dtm=st.lists(st.integers(1, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=1))
def test_dtm_to_dataframe(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm = np.array(dtm)
        dtm_arr = dtm

    doc_labels = ['doc%d' % i for i in range(dtm.shape[0])]
    vocab = ['t%d' % i for i in range(dtm.shape[1])]

    # check invalid doc_labels
    if len(doc_labels) > 0:
        with pytest.raises(ValueError):
            bow.dtm.dtm_to_dataframe(dtm, doc_labels[:-1], vocab)

    # check invalid vocab
    if len(vocab) > 0:
        with pytest.raises(ValueError):
            bow.dtm.dtm_to_dataframe(dtm, doc_labels, vocab[:-1])

    # check with valid doc_labels and vocab
    df = bow.dtm.dtm_to_dataframe(dtm, doc_labels, vocab)
    assert df.shape == dtm.shape
    assert np.array_equal(df.to_numpy(), dtm_arr)
    assert np.array_equal(df.index.values, doc_labels)
    assert np.array_equal(df.columns.values, vocab)


@given(dtm=st.lists(st.integers(1, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=1))
def test_dtm_to_gensim_corpus_and_gensim_corpus_to_dtm(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
    else:
        dtm = np.array(dtm)

    gensim_corpus = bow.dtm.dtm_to_gensim_corpus(dtm)
    assert isinstance(gensim_corpus, gensim.matutils.Sparse2Corpus)
    assert len(gensim_corpus) == dtm.shape[0]

    # convert back
    dtm_ = bow.dtm.gensim_corpus_to_dtm(gensim_corpus)
    assert isinstance(dtm_, coo_matrix)


@given(dtm=st.lists(st.integers(1, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
),
    matrix_type=st.integers(min_value=0, max_value=1),
    as_gensim_dictionary=st.booleans())
def test_dtm_and_vocab_to_gensim_corpus_and_dict(dtm, matrix_type, as_gensim_dictionary):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
    else:
        dtm = np.array(dtm)

    vocab = ['t%d' % i for i in range(dtm.shape[1])]

    gensim_corpus, id2word = bow.dtm.dtm_and_vocab_to_gensim_corpus_and_dict(dtm, vocab,
                                                                             as_gensim_dictionary=as_gensim_dictionary)
    assert isinstance(gensim_corpus, gensim.matutils.Sparse2Corpus)
    assert len(gensim_corpus) == dtm.shape[0]

    if as_gensim_dictionary:
        assert isinstance(id2word, gensim.corpora.Dictionary)
    else:
        assert isinstance(id2word, dict)
