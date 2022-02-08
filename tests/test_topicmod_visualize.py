import os
import random

import pytest
from hypothesis import given, strategies as st, settings

import numpy as np
import matplotlib.pyplot as plt

from ._testtools import strategy_2d_prob_distribution

from tmtoolkit.utils import empty_chararray
from tmtoolkit.topicmod import model_io, visualize, evaluate


def test_generate_wordclouds_for_topic_words():
    try:
        import lda
        import PIL
        from wordcloud import WordCloud
    except ImportError:
        pytest.skip('at least one of lda, Pillow, wordcloud not installed')

    data = model_io.load_ldamodel_from_pickle(os.path.join('tests', 'data', 'tiny_model_reuters_5_topics.pickle'))
    model = data['model']
    vocab = data['vocab']

    phi = model.topic_word_
    assert phi.shape == (5, len(vocab))

    topic_word_clouds = visualize.generate_wordclouds_for_topic_words(phi, vocab, 10)
    assert len(topic_word_clouds) == 5
    assert set(topic_word_clouds.keys()) == set('topic_%d' % i for i in range(1, 6))
    assert all(isinstance(wc, PIL.Image.Image) for wc in topic_word_clouds.values())

    topic_word_clouds = visualize.generate_wordclouds_for_topic_words(phi, vocab, 10,
                                                                      which_topics=('topic_1', 'topic_2'),
                                                                      return_images=False,
                                                                      width=640, height=480)
    assert set(topic_word_clouds.keys()) == {'topic_1', 'topic_2'}
    assert all(isinstance(wc, WordCloud) for wc in topic_word_clouds.values())
    assert all(wc.width == 640 and wc.height == 480 for wc in topic_word_clouds.values())


def test_generate_wordclouds_for_document_topics():
    try:
        import lda
        import PIL
        from wordcloud import WordCloud
    except ImportError:
        pytest.skip('at least one of lda, Pillow, wordcloud not installed')

    data = model_io.load_ldamodel_from_pickle(os.path.join('tests', 'data', 'tiny_model_reuters_5_topics.pickle'))
    model = data['model']
    doc_labels = data['doc_labels']

    theta = model.doc_topic_
    assert theta.shape == (len(doc_labels), 5)

    doc_topic_clouds = visualize.generate_wordclouds_for_document_topics(theta, doc_labels, 3)
    assert len(doc_topic_clouds) == len(doc_labels)
    assert set(doc_topic_clouds.keys()) == set(doc_labels)
    assert all(isinstance(wc, PIL.Image.Image) for wc in doc_topic_clouds.values())

    which_docs = doc_labels[:2]
    assert len(which_docs) == 2
    doc_topic_clouds = visualize.generate_wordclouds_for_document_topics(theta, doc_labels, 3,
                                                                         which_documents=which_docs,
                                                                         return_images=False,
                                                                         width=640, height=480)
    assert set(doc_topic_clouds.keys()) == set(which_docs)
    assert all(isinstance(wc, WordCloud) for wc in doc_topic_clouds.values())
    assert all(wc.width == 640 and wc.height == 480 for wc in doc_topic_clouds.values())


def test_write_wordclouds_to_folder(tmpdir):
    try:
        import lda
        import PIL
        from wordcloud import WordCloud
    except ImportError:
        pytest.skip('at least one of lda, Pillow, wordcloud not installed')

    path = tmpdir.mkdir('wordclouds').dirname

    data = model_io.load_ldamodel_from_pickle(os.path.join('tests', 'data', 'tiny_model_reuters_5_topics.pickle'))
    model = data['model']
    vocab = data['vocab']

    phi = model.topic_word_
    assert phi.shape == (5, len(vocab))

    topic_word_clouds = visualize.generate_wordclouds_for_topic_words(phi, vocab, 10)

    visualize.write_wordclouds_to_folder(topic_word_clouds, path, 'cloud_{label}.png')

    for label in topic_word_clouds.keys():
        assert os.path.exists(os.path.join(path, 'cloud_{label}.png'.format(label=label)))


@settings(deadline=5000)
@given(
    doc_topic=strategy_2d_prob_distribution(),
    make_topic_labels=st.booleans()
)
def test_plot_doc_topic_heatmap(doc_topic, make_topic_labels):
    doc_topic = np.array(doc_topic)
    doc_labels = ['d%d' % i for i in range(doc_topic.shape[0])]

    if make_topic_labels and doc_topic.ndim == 2:
        topic_labels = ['t%d' % i for i in range(doc_topic.shape[1])]
    else:
        topic_labels = None

    fig, ax = plt.subplots(figsize=(8, 6))

    if doc_topic.ndim != 2 or 0 in set(doc_topic.shape):
        with pytest.raises(ValueError):
            visualize.plot_doc_topic_heatmap(fig, ax, doc_topic, doc_labels=doc_labels, topic_labels=topic_labels)
    else:
        visualize.plot_doc_topic_heatmap(fig, ax, doc_topic, doc_labels=doc_labels, topic_labels=topic_labels)

    plt.close(fig)


@settings(deadline=5000)
@given(topic_word=strategy_2d_prob_distribution())
def test_plot_topic_word_heatmap(topic_word):
    topic_word = np.array(topic_word)

    if topic_word.ndim == 2:
        vocab = np.array(['t%d' % i for i in range(topic_word.shape[1])])
    else:
        vocab = empty_chararray()

    fig, ax = plt.subplots(figsize=(8, 6))

    if topic_word.ndim != 2 or 0 in set(topic_word.shape):
        with pytest.raises(ValueError):
            visualize.plot_topic_word_heatmap(fig, ax, topic_word, vocab)
    else:
        visualize.plot_topic_word_heatmap(fig, ax, topic_word, vocab)

    plt.close(fig)


# TODO: check how eval. results are generated and reenable this
# @settings(deadline=5000)
# @given(n_param_sets=st.integers(0, 10),
#        n_params=st.integers(1, 3),
#        n_metrics=st.integers(1, 3),
#        plot_specific_metric=st.booleans())
# def test_plot_eval_results(n_param_sets, n_params, n_metrics, plot_specific_metric):
#     param_names = ['param' + str(i) for i in range(n_params)]
#     metric_names = ['metric' + str(i) for i in range(n_metrics)]
#     res = []
#     for _ in range(n_param_sets):
#         param_set = dict(zip(param_names, np.random.randint(0, 100, n_params)))
#         metric_results = dict(zip(metric_names, np.random.uniform(0, 1, n_metrics)))
#         res.append((param_set, metric_results))
#
#     p = random.sample(param_names, random.randint(1, len(param_names)))
#     by_param = evaluate.results_by_parameter(res, p)
#
#     if not by_param:
#         with pytest.raises(ValueError):
#             visualize.plot_eval_results(by_param)
#     else:
#         if plot_specific_metric:
#             metric = random.choice(metric_names)
#         else:
#             metric = None
#
#         fig, _, _ = visualize.plot_eval_results(by_param, metric=metric, param=p)
#         plt.close(fig)
