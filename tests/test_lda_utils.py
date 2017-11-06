from __future__ import division

import random

import pytest
import hypothesis.strategies as st
from hypothesis import given

import numpy as np
import lda
import gensim
from sklearn.decomposition.online_lda import LatentDirichletAllocation

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


# @given(dtm=st.lists(st.integers(2, 10), min_size=2, max_size=2).flatmap(
#            lambda size: st.lists(st.lists(st.integers(0, 10),
#                                           min_size=size[0], max_size=size[0]),
#                                  min_size=size[1], max_size=size[1])
#        ))
def test_evaluation_lda_all_metrics_multi_vs_singleproc():
    passed_params = {'n_topics', 'alpha', 'n_iter', 'refresh', 'random_state'}
    varying_params = [dict(n_topics=k, alpha=1/k) for k in range(2, 5)]
    const_params = dict(n_iter=3, refresh=1, random_state=1)

    eval_res = lda_utils.tm_lda.evaluate_topic_models(EVALUATION_TEST_DTM, varying_params, const_params,
                                                      griffiths_2004_burnin=1)

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
                                                                 n_max_processes=1, griffiths_2004_burnin=1)
    assert len(eval_res_singleproc) == len(eval_res)
    for param_set2, metric_results2 in eval_res_singleproc:
        for x, y in eval_res:
            if x == param_set2:
                param_set1, metric_results1 = x, y
                break
        else:
            assert False

        assert metric_results1 == metric_results2


# evaluation gensim


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


# evaluation sklearn


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
