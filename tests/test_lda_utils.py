import random

import pytest
import hypothesis.strategies as st
from hypothesis import given

import numpy as np
import lda

from tmtoolkit import lda_utils


# common

@given(n=st.integers(0, 10),
       distrib=st.lists(st.integers(0, 9), min_size=2, max_size=2).flatmap(
           lambda size: st.lists(st.lists(st.floats(0, 1, allow_nan=False, allow_infinity=False),
                                          min_size=size[0], max_size=size[0]),
                                 min_size=size[1], max_size=size[1])
       ))
def test_common_top_n_from_distribution(n, distrib):
    distrib = np.array(distrib)
    if len(distrib) == 0:
        with pytest.raises(ValueError):
            lda_utils.top_n_from_distribution(distrib, n)
    else:
        if n < 1 or n > len(distrib[0]):
            with pytest.raises(ValueError):
                lda_utils.top_n_from_distribution(distrib, n)
        else:
            df = lda_utils.top_n_from_distribution(distrib, n)

            assert len(df) == len(distrib)

            for _, row in df.iterrows():
                assert len(row) == n
                assert list(sorted(row, reverse=True)) == list(row)


def test_save_load_ldamodel_pickle():
    pfile = 'tests/data/test_pickle_unpickle_ldamodel.pickle'

    dtm = np.array([[0, 1], [2, 3], [4, 5], [6, 0]])
    doc_labels = ['doc_' + str(i) for i in range(dtm.shape[0])]
    vocab = ['word_' + str(i) for i in range(dtm.shape[1])]

    model = lda.LDA(2, n_iter=1)
    model.fit(dtm)

    lda_utils.save_ldamodel_to_pickle(model, vocab, doc_labels, pfile)

    model_, vocab_, doc_labels_ = lda_utils.load_ldamodel_from_pickle(pfile)

    assert np.array_equal(model.doc_topic_, model_.doc_topic_)
    assert np.array_equal(model.topic_word_, model_.topic_word_)
    assert vocab == vocab_
    assert doc_labels == doc_labels_


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
    by_param = lda_utils.results_by_parameter(res, p)
    assert len(res) == len(by_param)
    assert all(x == 2 for x in map(len, by_param))


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
           lambda size: st.lists(st.lists(st.integers(0, 10),
                                          min_size=size[0], max_size=size[0]),
                                 min_size=size[1], max_size=size[1])
       ))
def test_get_doc_lengths(dtm):
    dtm = np.array(dtm)
    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            lda_utils.get_doc_lengths(dtm)
    else:
        doc_lengths = lda_utils.get_doc_lengths(dtm)
        assert doc_lengths.shape == (dtm.shape[0],)
        assert list(doc_lengths) == [sum(row) for row in dtm]


@given(dtm=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
           lambda size: st.lists(st.lists(st.integers(0, 10),
                                          min_size=size[0], max_size=size[0]),
                                 min_size=size[1], max_size=size[1])
       ))
def test_get_term_frequencies(dtm):
    dtm = np.array(dtm)
    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            lda_utils.get_term_frequencies(dtm)
    else:
        tf = lda_utils.get_term_frequencies(dtm)
        assert tf.shape == (dtm.shape[1],)
        assert list(tf) == [sum(row) for row in dtm.T]


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

    doc_lengths = lda_utils.get_doc_lengths(dtm)
    marginal_topic_distr = lda_utils.get_marginal_topic_distrib(model.doc_topic_, doc_lengths)

    assert marginal_topic_distr.shape == (n_topics,)
    assert np.isclose(marginal_topic_distr.sum(), 1.0)
