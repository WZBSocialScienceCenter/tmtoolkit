import random

import gensim
import lda
import numpy as np
from hypothesis import given, strategies as st
from sklearn.decomposition import LatentDirichletAllocation

from tmtoolkit.topicmod import evaluate, tm_lda, tm_sklearn, tm_gensim


def test_metric_held_out_documents_wallach09():
    """
    Test with data from original MATLAB implementation by Ian Murray
    https://people.cs.umass.edu/~wallach/code/etm/
    """
    np.random.seed(0)

    alpha = np.array([
        0.11689,
        0.42451,
        0.45859
    ])

    alpha /= alpha.sum()   # normalize inexact numbers

    phi = np.array([
        [0.306800, 0.094071, 0.284774, 0.211957, 0.102399],
        [0.234192, 0.157973, 0.093717, 0.280588, 0.233528],
        [0.173420, 0.166972, 0.196522, 0.208105, 0.254981]
    ])
    phi /= phi.sum(axis=1)[:, np.newaxis]       # normalize inexact numbers

    dtm = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
    ])

    theta = np.array([
        [0.044671, 0.044671, 0.059036, 0.082889, 0.044671, 0.082889, 0.143070],
        [0.429325, 0.429325, 0.429297, 0.487980, 0.429325, 0.487980, 0.269092],
        [0.526004, 0.526004, 0.511666, 0.429130, 0.526004, 0.429130, 0.587838],
    ]).T
    theta /= theta.sum(axis=1)[:, np.newaxis]       # normalize inexact numbers

    res = evaluate.metric_held_out_documents_wallach09(dtm, theta, phi, alpha, n_samples=10000)

    assert round(res) == -11


EVALUATION_TEST_DTM = np.array([
        [1, 2, 3, 0, 0],
        [0, 0, 2, 2, 0],
        [3, 0, 1, 1, 3],
        [2, 1, 0, 2, 5],
])
EVALUATION_TEST_VOCAB = np.array(['a', 'b', 'c', 'd', 'e'])
EVALUATION_TEST_TOKENS = [
    ['a', 'b', 'b', 'c', 'c', 'c'],
    ['c', 'c', 'd', 'd'],
    ['a', 'a', 'a', 'c', 'd', 'e', 'e', 'e'],
    ['a', 'a', 'b', 'd', 'd', 'e', 'e', 'e', 'e', 'e'],
]
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

    models = tm_lda.compute_models_parallel(EVALUATION_TEST_DTM, varying_params, const_params)
    assert len(models) == len(varying_params)

    for param_set, model in models:
        assert set(param_set.keys()) == passed_params
        assert isinstance(model, lda.LDA)
        assert isinstance(model.doc_topic_, np.ndarray)
        assert isinstance(model.topic_word_, np.ndarray)

    models_singleproc = tm_lda.compute_models_parallel(EVALUATION_TEST_DTM, varying_params, const_params,
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
    models = tm_lda.compute_models_parallel(EVALUATION_TEST_DTM, constant_parameters=const_params)
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
    models = tm_lda.compute_models_parallel(docs, varying_params,
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
    models = tm_lda.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, constant_parameters=const_params)
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
    models = tm_lda.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, varying_params,
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

    evaluate_topic_models_kwargs = dict(
        metric=tm_lda.AVAILABLE_METRICS,
        held_out_documents_wallach09_n_samples=10,
        held_out_documents_wallach09_n_folds=2,
        coherence_gensim_vocab=EVALUATION_TEST_VOCAB,
        coherence_gensim_texts=EVALUATION_TEST_TOKENS,
        return_models=True
    )

    eval_res = tm_lda.evaluate_topic_models(EVALUATION_TEST_DTM, varying_params, const_params,
                                            **evaluate_topic_models_kwargs)

    assert len(eval_res) == len(varying_params)

    for param_set, metric_results in eval_res:
        assert set(param_set.keys()) == passed_params
        assert set(metric_results.keys()) == set(tm_lda.AVAILABLE_METRICS + ('model',))

        assert 0 <= metric_results['cao_juan_2009'] <= 1
        assert 0 <= metric_results['arun_2010']
        assert metric_results['coherence_mimno_2011'] < 0
        assert np.isclose(metric_results['coherence_gensim_u_mass'], metric_results['coherence_mimno_2011'])
        assert 0 <= metric_results['coherence_gensim_c_v'] <= 1
        assert metric_results['coherence_gensim_c_uci'] < 0
        assert metric_results['coherence_gensim_c_npmi'] < 0

        if 'griffiths_2004' in tm_lda.AVAILABLE_METRICS:  # only if gmpy2 is installed
            assert metric_results['griffiths_2004'] < 0

        if 'loglikelihood' in tm_lda.AVAILABLE_METRICS:
            assert metric_results['loglikelihood'] < 0

        if 'held_out_documents_wallach09' in tm_lda.AVAILABLE_METRICS:  # only if gmpy2 is installed
            assert metric_results['held_out_documents_wallach09'] < 0

        assert isinstance(metric_results['model'], lda.LDA)

    eval_res_singleproc = tm_lda.evaluate_topic_models(EVALUATION_TEST_DTM, varying_params, const_params,
                                                                n_max_processes=1, **evaluate_topic_models_kwargs)
    assert len(eval_res_singleproc) == len(eval_res)
    for param_set2, metric_results2 in eval_res_singleproc:
        for x, y in eval_res:
            if x == param_set2:
                param_set1, metric_results1 = x, y
                break
        else:
            assert False

        # exclude results that use metrics with random sampling
        if 'held_out_documents_wallach09' in tm_lda.AVAILABLE_METRICS:  # only if gmpy2 is installed
            del metric_results1['held_out_documents_wallach09']
            del metric_results2['held_out_documents_wallach09']

        del metric_results1['model']
        del metric_results2['model']

        assert metric_results1 == metric_results2


def test_evaluation_gensim_all_metrics():
    passed_params = {'num_topics', 'update_every', 'passes', 'iterations'}
    varying_params = [dict(num_topics=k) for k in range(2, 5)]
    const_params = dict(update_every=0, passes=1, iterations=1)

    eval_res = tm_gensim.evaluate_topic_models(EVALUATION_TEST_DTM, varying_params, const_params,
                                               metric=tm_gensim.AVAILABLE_METRICS,
                                               coherence_gensim_texts=EVALUATION_TEST_TOKENS,
                                               coherence_gensim_kwargs={
                                                   'dictionary': evaluate.FakedGensimDict.from_vocab(EVALUATION_TEST_VOCAB)
                                               },
                                               return_models=True)

    assert len(eval_res) == len(varying_params)

    for param_set, metric_results in eval_res:
        assert set(param_set.keys()) == passed_params
        assert set(metric_results.keys()) == set(tm_gensim.AVAILABLE_METRICS + ('model',))

        assert metric_results['perplexity'] > 0
        assert 0 <= metric_results['cao_juan_2009'] <= 1
        assert metric_results['coherence_mimno_2011'] < 0
        assert np.isclose(metric_results['coherence_gensim_u_mass'], metric_results['coherence_mimno_2011'])
        assert 0 <= metric_results['coherence_gensim_c_v'] <= 1
        assert metric_results['coherence_gensim_c_uci'] < 0
        assert metric_results['coherence_gensim_c_npmi'] < 0


def test_compute_models_parallel_gensim():
    passed_params = {'num_topics', 'update_every', 'passes', 'iterations'}
    varying_params = [dict(num_topics=k) for k in range(2, 5)]
    const_params = dict(update_every=0, passes=1, iterations=1)

    models = tm_gensim.compute_models_parallel(EVALUATION_TEST_DTM, varying_params, const_params)

    assert len(models) == len(varying_params)

    for param_set, model in models:
        assert set(param_set.keys()) == passed_params
        assert isinstance(model, gensim.models.LdaModel)
        assert isinstance(model.state.get_lambda(), np.ndarray)


def test_compute_models_parallel_gensim_multiple_docs():
    # 1 doc, no varying params
    const_params = dict(num_topics=3, update_every=0, passes=1, iterations=1)
    models = tm_gensim.compute_models_parallel(EVALUATION_TEST_DTM, constant_parameters=const_params)
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
    models = tm_gensim.compute_models_parallel(docs, varying_params,
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
    models = tm_gensim.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, constant_parameters=const_params)
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
    models = tm_gensim.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, varying_params,
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


def test_evaluation_sklearn_all_metrics():
    passed_params = {'n_components', 'learning_method', 'evaluate_every', 'max_iter', 'n_jobs'}
    varying_params = [dict(n_components=k) for k in range(2, 5)]
    const_params = dict(learning_method='batch', evaluate_every=1, max_iter=3, n_jobs=1)

    evaluate_topic_models_kwargs = dict(
        metric=tm_sklearn.AVAILABLE_METRICS,
        held_out_documents_wallach09_n_samples=10,
        held_out_documents_wallach09_n_folds=2,
        coherence_gensim_vocab=EVALUATION_TEST_VOCAB,
        coherence_gensim_texts=EVALUATION_TEST_TOKENS,
        return_models=True,
    )

    eval_res = tm_sklearn.evaluate_topic_models(EVALUATION_TEST_DTM, varying_params, const_params,
                                                **evaluate_topic_models_kwargs)

    assert len(eval_res) == len(varying_params)

    for param_set, metric_results in eval_res:
        assert set(param_set.keys()) == passed_params
        assert set(metric_results.keys()) == set(tm_sklearn.AVAILABLE_METRICS + ('model',))

        assert metric_results['perplexity'] > 0
        assert 0 <= metric_results['cao_juan_2009'] <= 1
        assert 0 <= metric_results['arun_2010']
        assert metric_results['coherence_mimno_2011'] < 0
        assert np.isclose(metric_results['coherence_gensim_u_mass'], metric_results['coherence_mimno_2011'])
        assert 0 <= metric_results['coherence_gensim_c_v'] <= 1
        assert metric_results['coherence_gensim_c_uci'] < 0
        assert metric_results['coherence_gensim_c_npmi'] < 0

        if 'held_out_documents_wallach09' in tm_lda.AVAILABLE_METRICS:  # only if gmpy2 is installed
            assert metric_results['held_out_documents_wallach09'] < 0

        assert isinstance(metric_results['model'], LatentDirichletAllocation)


def test_compute_models_parallel_sklearn():
    passed_params = {'n_components', 'learning_method', 'evaluate_every', 'max_iter', 'n_jobs'}
    varying_params = [dict(n_components=k) for k in range(2, 5)]
    const_params = dict(learning_method='batch', evaluate_every=1, max_iter=3, n_jobs=1)

    models = tm_sklearn.compute_models_parallel(EVALUATION_TEST_DTM, varying_params, const_params)

    assert len(models) == len(varying_params)

    for param_set, model in models:
        assert set(param_set.keys()) == passed_params
        assert isinstance(model, LatentDirichletAllocation)
        assert isinstance(model.components_, np.ndarray)


def test_compute_models_parallel_sklearn_multiple_docs():
    # 1 doc, no varying params
    const_params = dict(n_components=3, learning_method='batch', evaluate_every=1, max_iter=3, n_jobs=1)
    models = tm_sklearn.compute_models_parallel(EVALUATION_TEST_DTM, constant_parameters=const_params)
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
    models = tm_sklearn.compute_models_parallel(docs, varying_params,
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
    models = tm_sklearn.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, constant_parameters=const_params)
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
    models = tm_sklearn.compute_models_parallel(EVALUATION_TEST_DTM_MULTI, varying_params,
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
    by_param = evaluate.results_by_parameter(res, p)
    assert len(res) == len(by_param)
    assert all(x == 2 for x in map(len, by_param))
