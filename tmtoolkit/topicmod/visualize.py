"""
Functions to visualize topic models and topic model evaluation results.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""
import itertools
import math
import os
import logging
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from tmtoolkit.topicmod.model_stats import top_n_from_distribution
from tmtoolkit.bow.bow_stats import doc_lengths, term_frequencies
from tmtoolkit.topicmod import evaluate
from tmtoolkit.utils import mat2d_window_from_indices

logger = logging.getLogger('tmtoolkit')


#%% word clouds from topic models


def _wordcloud_color_func_black(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'rgb(0,0,0)'


#: Default wordcloud settings for transparent background and black font; will be passed to :class:`wordcloud.WordCloud`
DEFAULT_WORDCLOUD_KWARGS = {
    'width': 800,
    'height': 600,
    'mode': 'RGBA',
    'background_color': None,
    'color_func': _wordcloud_color_func_black
}


def write_wordclouds_to_folder(wordclouds, folder, file_name_fmt='{label}.png', **save_kwargs):
    """
    Save all wordcloud image objects in `wordclouds` to `folder`.

    :param wordclouds: dict mapping wordcloud label to wordcloud object
    :param folder: target path
    :param file_name_fmt: file name string format with placeholder ``"{label}"``
    :param save_kwargs: additional options passed to `save` method of each wordcloud image object
    """

    if not os.path.exists(folder):
        raise ValueError('target folder `%s` does not exist' % folder)

    for label, wc in wordclouds.items():
        file_name = file_name_fmt.format(label=label)
        file_path = os.path.join(folder, file_name)
        logger.info(f'writing wordcloud to file "{file_path}"')

        wc.save(file_path, **save_kwargs)


def generate_wordclouds_for_topic_words(topic_word_distrib, vocab, top_n, topic_labels='topic_{i1}', which_topics=None,
                                        return_images=True, **wordcloud_kwargs):
    """
    Generate wordclouds for the top `top_n` words of each topic in `topic_word_distrib`.

    :param topic_word_distrib: topic-word distribution; shape KxM, where K is number of topics, M is vocabulary size
    :param vocab: vocabulary array of length M
    :param top_n: number of top values to take from each row of `distrib`
    :param topic_labels: labels used for each row; determine keys in in result dict; either single format string with
                         placeholders ``"{i0}"`` (zero-based topic index) or ``"{i1}"`` (one-based topic index), or
                         list of topic label strings
    :param which_topics: if not None, a sequence of indices into rows of `topic_word_distrib` to select only these
                         topics to generate wordclouds from
    :param return_images: if True, store image objects instead of :class:`wordcloud.WordCloud` objects in the result
                          dict
    :param wordcloud_kwargs: pass additional options to :class:`wordcloud.WordCloud`; updates options in
           :data:`~tmtoolkit.topicmod.visualize.DEFAULT_WORDCLOUD_KWARGS`
    :return: dict mapping row labels to wordcloud images or instances generated from each topic
    """
    return generate_wordclouds_from_distribution(topic_word_distrib, row_labels=topic_labels, val_labels=vocab,
                                                 top_n=top_n, which_rows=which_topics, return_images=return_images,
                                                 **wordcloud_kwargs)


def generate_wordclouds_for_document_topics(doc_topic_distrib, doc_labels, top_n, topic_labels='topic_{i1}',
                                            which_documents=None, return_images=True, **wordcloud_kwargs):
    """
    Generate wordclouds for the top `top_n` topics of each document in `doc_topic_distrib`.

    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents, K is the
                              number of topics
    :param doc_labels: list/array of length N with a string label for each document
    :param top_n: number of top values to take from each row of `distrib`
    :param topic_labels: labels used for each row; determine keys in in result dict; either single format string with
                         placeholders ``"{i0}"`` (zero-based topic index) or ``"{i1}"`` (one-based topic index), or
                         list of topic label strings
    :param which_documents: if not None, a sequence of indices into rows of `doc_topic_distrib` to select only these
                            topics to generate wordclouds from
    :param return_images: if True, store image objects instead of :class:`wordcloud.WordCloud` objects in the result
                          dict
    :param wordcloud_kwargs: pass additional options to :class:`wordcloud.WordCloud`; updates options in
           :data:`~tmtoolkit.topicmod.visualize.DEFAULT_WORDCLOUD_KWARGS`
    :return: dict mapping row labels to wordcloud images or instances generated from each document
    """
    return generate_wordclouds_from_distribution(doc_topic_distrib, row_labels=doc_labels, val_labels=topic_labels, top_n=top_n,
                                                 which_rows=which_documents, return_images=return_images,
                                                 **wordcloud_kwargs)


def generate_wordclouds_from_distribution(distrib, row_labels, val_labels, top_n, which_rows=None, return_images=True,
                                          **wordcloud_kwargs):
    """
    Generate wordclouds for each row in a given probability distribution `distrib`.

    .. note:: Use :func:`~tmtoolkit.topicmod.visualize.generate_wordclouds_for_topic_words` or
              :func:`~tmtoolkit.topicmod.visualize.generate_wordclouds_for_document_topics` as shortcuts for creating
              wordclouds for a topic-word or document-topic distribution.

    :param distrib: 2D (sparse) array/matrix probability distribution
    :param row_labels: labels for rows in probability distribution; these are used as keys in the return dict
    :param val_labels: labels for values in probability distribution (e.g. vocabulary)
    :param top_n: number of top values to take from each row of `distrib`
    :param which_rows: if not None, select only the rows from this sequence of indices from `distrib`
    :param return_images: if True, store image objects instead of :class:`wordcloud.WordCloud` objects in the result
                          dict
    :param wordcloud_kwargs: pass additional options to :class:`wordcloud.WordCloud`; updates options in
           :data:`~tmtoolkit.topicmod.visualize.DEFAULT_WORDCLOUD_KWARGS`
    :return: dict mapping row labels to wordcloud images or instances generated from each distribution row
    """

    prob = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=None)
    words = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=val_labels)

    if which_rows:
        prob = prob.loc[which_rows, :]
        words = words.loc[which_rows, :]

        assert prob.shape == words.shape

    wordclouds = {}
    for (p_row_name, p), (w_row_name, w) in zip(prob.iterrows(), words.iterrows()):
        assert p_row_name == w_row_name
        logger.info(f'generating wordcloud for "{p_row_name}"')
        wc = generate_wordcloud_from_probabilities_and_words(p, w,
                                                             return_image=return_images,
                                                             **wordcloud_kwargs)
        wordclouds[p_row_name] = wc

    return wordclouds


def generate_wordcloud_from_probabilities_and_words(prob, words, return_image=True, wordcloud_instance=None,
                                                    **wordcloud_kwargs):
    """
    Generate a single wordcloud for given probabilities (weights) `prob` of the respective `words`.

    :param prob: 1D array or sequence of probabilities for `words`
    :param words: 1D array or sequence of word strings
    :param return_images: if True, store image objects instead of :class:`wordcloud.WordCloud` objects in the result
                          dict
    :param wordcloud_instance: optionally pass an already initialized :class:`wordcloud.WordCloud` instance
    :param wordcloud_kwargs: pass additional options to :class:`wordcloud.WordCloud`; updates options in
           :data:`~tmtoolkit.topicmod.visualize.DEFAULT_WORDCLOUD_KWARGS`
    :return: either a wordcloud image if `return_images` is True, otherwise a :class:`wordcloud.WordCloud` instance
    """

    if len(prob) != len(words):
        raise ValueError('`prob` and `words` must have the name length')
    if hasattr(prob, 'ndim') and prob.ndim != 1:
        raise ValueError('`prob` must be a 1D array or sequence')
    if hasattr(words, 'ndim') and words.ndim != 1:
        raise ValueError('`words` must be a 1D array or sequence')

    weights = dict(zip(words, prob))

    return generate_wordcloud_from_weights(weights, return_image=return_image,
                                           wordcloud_instance=wordcloud_instance, **wordcloud_kwargs)


def generate_wordcloud_from_weights(weights, return_image=True, wordcloud_instance=None, **wordcloud_kwargs):
    """
    Generate a single wordcloud for a `weights` dict that maps words to "weights" (e.g. probabilities) which determine
    their size in the wordcloud.

    :param weights: dict that maps words to weights
    :param return_images: if True, store image objects instead of :class:`wordcloud.WordCloud` objects in the result
                          dict
    :param wordcloud_instance: optionally pass an already initialized :class:`wordcloud.WordCloud` instance
    :param wordcloud_kwargs: pass additional options to :class:`wordcloud.WordCloud`; updates options in
           :data:`~tmtoolkit.topicmod.visualize.DEFAULT_WORDCLOUD_KWARGS`
    :return: either a wordcloud image if `return_images` is True, otherwise a :class:`wordcloud.WordCloud` instance
    """

    if not isinstance(weights, dict) or not weights:
        raise ValueError('`weights` must be a non-empty dictionary')

    if not wordcloud_instance:
        from wordcloud import WordCloud

        use_wc_kwargs = DEFAULT_WORDCLOUD_KWARGS.copy()
        use_wc_kwargs.update(wordcloud_kwargs)
        wordcloud_instance = WordCloud(**use_wc_kwargs)

    wordcloud_instance.generate_from_frequencies(weights)

    if return_image:
        return wordcloud_instance.to_image()
    else:
        return wordcloud_instance


#%% plot 2D probability distribution rankings


def plot_topic_word_ranked_prob(fig, ax, topic_word_distrib, n,
                                highlight_label_fmt='topic {i0}',
                                highlight_label_other='other topics',
                                title='Ranked word probability per topic',
                                xaxislabel='word rank',
                                yaxislabel='word probability',
                                **kwargs):
    """
    Plot a topic-word probability distribution by ranking the probabilities in each row. This is for example useful
    in order to examine how many top words usually describe most of a topic.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param topic_word_distrib: topic-word probability distribution
    :param n: limit max. shown word rank on x-axis
    :param highlight_label_fmt: if `highlight` is given, use this format for labeling the highlighted rows
    :param highlight_label_other: if `highlight` is given, use this as label for non-highlighted rows
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param kwargs: further arguments passed to :func:`~tmtoolkit.topicmod.visualize.plot_prob_distrib_ranked_prob`
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """
    return plot_prob_distrib_ranked_prob(fig, ax, topic_word_distrib, x_limit=n,
                                         highlight_label_fmt=highlight_label_fmt,
                                         highlight_label_other=highlight_label_other,
                                         title=title, xaxislabel=xaxislabel, yaxislabel=yaxislabel, **kwargs)


def plot_doc_topic_ranked_prob(fig, ax, doc_topic_distrib, n,
                               highlight_label_fmt='document {i0}',
                               highlight_label_other='other documents',
                               title='Ranked topic probability per document',
                               xaxislabel='topic rank',
                               yaxislabel='topic probability',
                               **kwargs):
    """
    Plot a document-topic probability distribution by ranking the probabilities in each row. This is for example useful
    in order to examine how many top topics usually describe most of a document.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param doc_topic_distrib: document-topic probability distribution
    :param n: limit max. shown topic rank on x-axis
    :param highlight_label_fmt: if `highlight` is given, use this format for labeling the highlighted rows
    :param highlight_label_other: if `highlight` is given, use this as label for non-highlighted rows
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param kwargs: further arguments passed to :func:`~tmtoolkit.topicmod.visualize.plot_prob_distrib_ranked_prob`
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """
    return plot_prob_distrib_ranked_prob(fig, ax, doc_topic_distrib, x_limit=n,
                                         highlight_label_fmt=highlight_label_fmt,
                                         highlight_label_other=highlight_label_other,
                                         title=title, xaxislabel=xaxislabel, yaxislabel=yaxislabel, **kwargs)


def plot_prob_distrib_ranked_prob(fig, ax, data, x_limit, log_scale=True, lw=1, alpha=0.1,
                                  highlight=None, highlight_label_fmt='{i0}', highlight_label_other='other',
                                  highlight_lw=3, highlight_alpha=0.3,
                                  title=None, xaxislabel='rank', yaxislabel='probability'):
    """
    Plot a 2D probability distribution (one distribution for each row which should add up to 1) by ranking the
    probabilities in each row.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param data: a 2D probability distribution (one distribution for each row which should add up to 1)
    :param x_limit: limit max. shown rank on x-axis
    :param log_scale: if True, apply log scale on y-axis
    :param lw: line width
    :param alpha: line transparency
    :param highlight: if given, pass a sequence or NumPy array with *indices* of rows in `data`, which should be
                      highlighted
    :param highlight_label_fmt: if `highlight` is given, use this format for labeling the highlighted rows
    :param highlight_label_other: if `highlight` is given, use this as label for non-highlighted rows
    :param highlight_lw: line width for highlighted distributions
    :param highlight_alpha: line transparency for highlighted distributions
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.ndim != 2:
        raise ValueError('`data` must be a 2D matrix/array')

    if data.shape[1] == 0:
        raise ValueError('`data` must have at least one column')

    if not 1 <= x_limit <= data.shape[1]:
        raise ValueError('`x_limit` must be strictly positive and no larger than the number of columns in `data`')

    if highlight:
        if not isinstance(highlight, np.ndarray):
            highlight = np.array(highlight)

        palette = plt.get_cmap('Dark2')
        highlight_handles = []
    else:
        palette = None
        highlight_handles = None

    # log transform
    if log_scale:
        data = np.log10(data)

    # set title
    if title:
        ax.set_title(title)

    # rowwise sorting (NumPy still doesn't support descending order, hence the "-" hack)
    data_desc = -np.sort(-data, axis=1)

    ranks = np.arange(1, x_limit + 1)
    for i, row in enumerate(data_desc):  # each row is a prob. distrib. with descending prob. values
        if highlight is not None:
            highlight_index = np.where(i == highlight)[0]
        else:
            highlight_index = []

        if len(highlight_index) > 0:
            color = palette(highlight_index[0])
            label = highlight_label_fmt.format(i0=i, i1=i+1)
            lw_ = highlight_lw
            alpha_ = highlight_alpha
        else:
            color = 'black'
            label = highlight_label_other
            lw_ = lw
            alpha_ = alpha

        res = ax.plot(ranks, row[:x_limit], color=color, label=label, lw=lw_, alpha=alpha_)

        if len(highlight_index) > 0:
            highlight_handles.append(res[0])

    # customize axes
    if xaxislabel:
        ax.set_xlabel(xaxislabel)
    if yaxislabel:
        if log_scale:
            yaxislabel += ' (log10 scale)'
        ax.set_ylabel(yaxislabel)

    if highlight_handles:
        ax.legend(handles=highlight_handles, loc='best')

    return fig, ax


#%% plot heatmaps (especially for doc-topic distribution)


def plot_doc_topic_heatmap(fig, ax, doc_topic_distrib, doc_labels, topic_labels=None,
                           which_documents=None, which_document_indices=None,
                           which_topics=None, which_topic_indices=None,
                           xaxislabel=None, yaxislabel=None,
                           **kwargs):
    """
    Plot a heatmap for a document-topic distribution `doc_topic_distrib` to a matplotlib Figure `fig` and Axes `ax`
    using `doc_labels` as document labels on the y-axis and topics from 1 to K (number of topics) on
    the x-axis.

    .. note:: It is almost always necessary to select a subset of your document-topic distribution with the
              `which_documents` or `which_topics` parameters, as otherwise the amount of data to be plotted will be too
              high to give a reasonable picture.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents, K is the
                              number of topics
    :param doc_labels: list/array of length N with a string label for each document
    :param topic_labels: labels used for each row; either single format string with
                         placeholders ``"{i0}"`` (zero-based topic index) or ``"{i1}"`` (one-based topic index), or
                         list of topic label strings
    :param which_documents: select documents via document label strings
    :param which_document_indices: alternatively, select documents with zero-based document index in [0, N-1]
    :param which_topics: select topics via topic label strings (when string array or list) or with
                         one-based topic index in [1, K] (when integer array or list)
    :param which_topic_indices:  alternatively, select topics with zero-based topic index in [0, K-1]
    :param xaxislabel: x axis label string
    :param yaxislabel: y axis label string
    :param kwargs: additional arguments passed to :func:`~tmtoolkit.topicmod.visualize.plot_heatmap`
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    if not isinstance(doc_topic_distrib, np.ndarray) or doc_topic_distrib.ndim != 2:
        raise ValueError('`mat` must be a 2D NumPy array')

    if doc_topic_distrib.shape[0] == 0 or doc_topic_distrib.shape[1] == 0:
        raise ValueError('invalid shape for `mat`: %s' % str(doc_topic_distrib.shape))

    if which_documents is not None and which_document_indices is not None:
        raise ValueError('only `which_documents` or `which_document_indices` can be set, not both')

    if which_topics is not None and which_topic_indices is not None:
        raise ValueError('only `which_topics` or `which_topic_indices` can be set, not both')

    if which_documents is not None:
        which_document_indices = np.where(np.isin(doc_labels, which_documents))[0]

    select_distrib_subset = False

    if topic_labels is None:
        topic_labels = np.array(range(1, doc_topic_distrib.shape[1]+1))
    elif not isinstance(topic_labels, np.ndarray):
        topic_labels = np.array(topic_labels)

    if which_topics is not None:
        which_topics = np.array(which_topics)
        if which_topics.dtype.kind == 'U':
            which_topic_indices = np.where(np.isin(topic_labels, which_topics))[0]
        else:
            which_topic_indices = which_topics - 1

    if which_document_indices is not None:
        select_distrib_subset = True
        doc_labels = np.array(doc_labels)[which_document_indices]

    if which_topic_indices is not None:
        select_distrib_subset = True
        topic_labels = topic_labels[which_topic_indices]

    if select_distrib_subset:
        doc_topic_distrib = mat2d_window_from_indices(doc_topic_distrib, which_document_indices, which_topic_indices)

    return plot_heatmap(fig, ax, doc_topic_distrib,
                        xaxislabel=xaxislabel or 'topic',
                        yaxislabel=yaxislabel or 'document',
                        xticklabels=topic_labels,
                        yticklabels=doc_labels,
                        **kwargs)


def plot_topic_word_heatmap(fig, ax, topic_word_distrib, vocab, topic_labels=None,
                            which_topics=None, which_topic_indices=None,
                            which_words=None, which_word_indices=None,
                            xaxislabel=None, yaxislabel=None,
                            **kwargs):
    """
    Plot a heatmap for a topic-word distribution `topic_word_distrib` to a matplotlib Figure `fig` and Axes `ax`
    using `vocab` as vocabulary on the x-axis and topics from 1 to `n_topics=doc_topic_distrib.shape[1]` on
    the y-axis.


    .. note:: It is almost always necessary to select a subset of your topic-word distribution with the
              `which_words` or `which_topics` parameters, as otherwise the amount of data to be plotted will be too high
              to give a reasonable picture.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param topic_word_distrib: topic-word distribution; shape KxM, where K is number of topics, M is vocabulary size
    :param vocab: vocabulary array of length M
    :param topic_labels: labels used for each row; either single format string with
                         placeholders ``"{i0}"`` (zero-based topic index) or ``"{i1}"`` (one-based topic index), or
                         list of topic label strings
    :param which_topics: select topics via topic label strings (when string array or list and `topic_labels` is given)
                         or with one-based topic index in [1, K] (when integer array or list)
    :param which_topic_indices:  alternatively, select topics with zero-based topic index in [0, K-1]
    :param which_words: select words with one-based word index in [1, M]
    :param which_word_indices: alternatively, select words with zero-based word index in [0, K-1]
    :param xaxislabel: x axis label string
    :param yaxislabel: y axis label string
    :param kwargs: additional arguments passed to :func:`~tmtoolkit.topicmod.visualize.plot_heatmap`
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """
    if not isinstance(topic_word_distrib, np.ndarray) or topic_word_distrib.ndim != 2:
        raise ValueError('`mat` must be a 2D NumPy array')

    if topic_word_distrib.shape[0] == 0 or topic_word_distrib.shape[1] == 0:
        raise ValueError('invalid shape for `mat`: %s' % str(topic_word_distrib.shape))

    if which_topics is not None and which_topic_indices is not None:
        raise ValueError('only `which_topics` or `which_topic_indices` can be set, not both')

    if which_words is not None and which_word_indices is not None:
        raise ValueError('only `which_words` or `which_word_indices` can be set, not both')

    if which_words is not None:
        which_word_indices = np.where(np.isin(vocab, which_words))[0]

    select_distrib_subset = False

    if topic_labels is None:
        topic_labels = np.array(range(1, topic_word_distrib.shape[0]+1))
    elif not isinstance(topic_labels, np.ndarray):
        topic_labels = np.array(topic_labels)

    if which_topics is not None:
        which_topics = np.array(which_topics)
        if which_topics.dtype.kind == 'U':
            which_topic_indices = np.where(np.isin(topic_labels, which_topics))[0]
        else:
            which_topic_indices = which_topics - 1

    if which_topic_indices is not None:
        select_distrib_subset = True
        topic_labels = topic_labels[which_topic_indices]

    if which_word_indices is not None:
        select_distrib_subset = True
        vocab = np.array(vocab)[which_word_indices]

    if select_distrib_subset:
        topic_word_distrib = mat2d_window_from_indices(topic_word_distrib, which_topic_indices, which_word_indices)

    return plot_heatmap(fig, ax, topic_word_distrib,
                        xaxislabel=xaxislabel or 'vocab',
                        yaxislabel=yaxislabel or 'topic',
                        xticklabels=vocab,
                        yticklabels=topic_labels,
                        **kwargs)


def plot_heatmap(fig, ax, data,
                 xaxislabel=None, yaxislabel=None,
                 xticklabels=None, yticklabels=None,
                 title=None, grid=True,
                 values_in_cells=True, round_values_in_cells=2,
                 legend=False,
                 fontsize_axislabel=None,
                 fontsize_axisticks=None,
                 fontsize_cell_values=None):
    """
    Generic heatmap plotting function for 2D matrix `data`.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param data: 2D array/matrix to be plotted as heatmap
    :param xaxislabel: x axis label string
    :param yaxislabel: y axis label string
    :param xticklabels: list of x axis tick labels
    :param yticklabels: list of y axis tick labels
    :param title: plot title
    :param grid: draw grid if True
    :param values_in_cells: draw values of `data` in heatmap cells
    :param round_values_in_cells: round these values to the given number of digits
    :param legend: if True, draw a legend
    :param fontsize_axislabel: font size for axis label
    :param fontsize_axisticks: font size for axis ticks
    :param fontsize_cell_values: font size for values in cells
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.ndim != 2:
        raise ValueError('`data` must be a 2D matrix/array')

    # draw basic heatmap
    cax = ax.matshow(data)

    # draw legend
    if legend:
        fig.colorbar(cax)

    # set title
    if title:
        ax.set_title(title, y=1.25)

    n_rows, n_cols = data.shape

    # draw values in cells
    if values_in_cells:
        textcol_thresh = data.min() + (data.max() - data.min()) / 2
        x_indices, y_indices = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        for x, y in zip(x_indices.flatten(), y_indices.flatten()):
            val = data[y, x]
            # lower values get white text color for better visibility
            textcol = 'white' if val < textcol_thresh else 'black'
            disp_val = round(val, round_values_in_cells) if round_values_in_cells is not None else val
            ax.text(x, y, disp_val, va='center', ha='center', color=textcol, fontsize=fontsize_cell_values)

    # customize axes
    if xaxislabel:
        ax.set_xlabel(xaxislabel)
    if yaxislabel:
        ax.set_ylabel(yaxislabel)

    if fontsize_axislabel:
        for item in (ax.xaxis.label, ax.yaxis.label):
            item.set_fontsize(fontsize_axislabel)

    ax.set_xticks(np.arange(0, n_cols))
    ax.set_yticks(np.arange(0, n_rows))

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=45, ha='left')
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if fontsize_axisticks:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(fontsize_axisticks)

    # gridlines based on minor ticks
    if grid:
        ax.set_xticks(np.arange(-.5, n_cols), minor=True)
        ax.set_yticks(np.arange(-.5, n_rows), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    return fig, ax


#%% plotting of evaluation results


def plot_eval_results(eval_results, metric=None, param=None,
                      xaxislabel=None, yaxislabel=None,
                      title=None,
                      title_fontsize='xx-large',
                      subfig_fontsize='large',
                      axes_title_fontsize='medium',
                      show_metric_direction=True,
                      metric_direction_font_size='medium',
                      subplots_adjust_opts=None,
                      figsize='auto',
                      fig_opts=None,
                      subfig_opts=None,
                      subplots_opts=None):
    """
    Plot the evaluation results from `eval_results`, which must be a sequence containing
    `(param_0, ..., param_N, metric results)` tuples, where `param_N` is the parameter value to appear on the x axis
    and all parameter combinations before are used to create a small multiples plot (if there are more than one param.).
    The metric results can be a dict structure containing the evaluation results for each metric. `eval_results` can be
    created using :func:`tmtoolkit.topicmod.evaluate.results_by_parameter`.

    .. note:: Due to a bug in matplotlib, it seems that it's not possible to display a plot title when plotting small
              multiples and adjusting the positioning of the subplots. Hence you must set `show_metric_direction` to
              False when you're displaying small multiples and need want to display a plot title.

    :param eval_results: topic evaluation results as sequence containing `(param_0, ..., param_N, metric results)`
    :param metric: either single string or list of strings; plot only this/these specific metric/s
    :param param: names of the parameters used in `eval_results`
    :param xaxislabel: x axis label string
    :param yaxislabel: y axis label string
    :param title: plot title
    :param title_fontsize: font size for the figure title
    :param axes_title_fontsize: font size for the plot titles
    :param show_metric_direction: if True, show whether the shown metric should be minimized or maximized for
                                  optimization
    :param metric_direction_font_size: font size for the metric optimization direction indicator
    :param subplots_opts: options passed to Matplotlib's ``plt.subplots()``
    :param subplots_adjust_opts: options passed to Matplotlib's ``fig.subplots_adjust()``
    :param figsize: tuple ``(width, height)`` or ``"auto"`` (default)
    :param fig_opts: additional parameters passed to Matplotlib's ``plt.figure()``
    :param subfig_opts: additional parameters passed to Matplotlib's ``fig.subfigures()``
    :param subplots_opts: additional parameters passed to Matplotlib's ``subfig.subplots()``
    :return: tuple of generated (matplotlib Figure object, matplotlib Subfigures, matplotlib Axes)
    """
    if type(eval_results) not in (list, tuple) or not eval_results:
        raise ValueError('`eval_results` must be a list or tuple with at least one element')

    first_row = next(iter(eval_results))

    if type(first_row) not in (list, tuple):
        raise ValueError('`eval_results` must be a list or tuple containing a (param, values) tuple. '
                         'Maybe `eval_results` must be converted with `results_by_parameter`.')

    n_params = len(first_row) - 1

    if n_params < 1:
        raise ValueError('each entry in `eval_results` must contain at least two values '
                         '(n parameter values and evaluation results)')

    if isinstance(param, str):
        param = [param]

    if param and len(param) != n_params:
        raise ValueError('if `param` is given, its length must equal the number of parameters in the eval. results')

    eval_colwise = list(zip(*eval_results))
    n_param_combinations = 1
    for p in range(0, n_params-1):   # we don't count the last level as this will go on the x-axis
        n_param_combinations *= len(set(eval_colwise[p]))

    if metric is not None and type(metric) not in (list, tuple):
        metric = [metric]
    elif metric is None:
        # remove special evaluation result 'model': the calculated model itself
        metric = sorted(set(first_row[-1].keys()) - {'model'})

    metric = sorted(metric)

    metric_direction = []
    for m in metric:
        if m == 'perplexity':
            metric_direction.append('minimize')
        else:
            m_fn_name = 'metric_%s' % (m[:16] if m.startswith('coherence_gensim') else m)
            m_fn = getattr(evaluate, m_fn_name, None)
            if m_fn:
                metric_direction.append(getattr(m_fn, 'direction', 'unknown'))
            else:
                metric_direction.append('unknown')

    n_metrics = len(metric)

    assert n_metrics == len(metric_direction)

    metrics_ordered = []
    for m_dir in sorted(set(metric_direction), reverse=True):
        metrics_ordered.extend([(m, d) for m, d in zip(metric, metric_direction) if d == m_dir])

    assert n_metrics == len(metrics_ordered)

    if n_param_combinations > 3:
        n_fig_rows = math.ceil(math.sqrt(n_param_combinations))
        n_fig_cols = n_fig_rows

        n_fig_rows -= (n_fig_rows**2 - n_param_combinations) // n_fig_rows
    else:
        n_fig_rows = 1
        n_fig_cols = n_param_combinations

    # get figures and subplots (axes)
    if figsize == 'auto':
        figsize = (6 * n_fig_cols, 2 * n_fig_rows * n_metrics)

    fig = plt.figure(layout='constrained', figsize=figsize, **(fig_opts or {}))

    subfigs = fig.subfigures(nrows=n_fig_rows, ncols=n_fig_cols, **(subfig_opts or {}))
    if isinstance(subfigs, np.ndarray):
        subfigs = subfigs.flatten()
    else:
        subfigs = [subfigs]

    #unique_param_values_param_index = []
    unique_param_values = []
    for col in eval_colwise[:-2]:
        unique_vals = set(col)
        #unique_param_values_param_index.append([i] * len(unique_vals))
        unique_param_values.append(sorted(unique_vals))

    param_combinations = list(itertools.product(*unique_param_values))
    assert len(param_combinations) == n_param_combinations

    x = np.array(sorted(set(eval_colwise[-2])))
    all_metrics_results = np.array(eval_colwise[-1])

    subfigs_axes = []

    for i_subfig, subfig in enumerate(subfigs):
        if len(subfigs) > 1:
            if i_subfig >= len(param_combinations):
                break
            param_vals = param_combinations[i_subfig]
            if param:
                subfig_titles = [f'{param[i]} = {v}' for i, v in enumerate(param_vals)]
            else:
                subfig_titles = [str(v) for v in param_vals]

            subfig.suptitle('\n'.join(subfig_titles), fontsize=subfig_fontsize)
            which_results = np.repeat(True, len(all_metrics_results))
            for i, v in enumerate(param_vals):
                which_results &= np.isclose(np.array(eval_colwise[i]), v)

            metrics_results = all_metrics_results[which_results]
        else:
            metrics_results = all_metrics_results

        axes = subfig.subplots(nrows=n_metrics, ncols=1, sharex=True, **(subplots_opts or {}))
        subfigs_axes.append(axes)

        # draw subplot for each metric
        axes_pos_per_dir = defaultdict(list)
        axes_sequence = axes.flatten() if n_metrics > 1 else [axes]
        assert len(axes_sequence) == len(metrics_ordered)
        for i, (ax, (m, m_dir)) in enumerate(zip(axes_sequence, metrics_ordered)):
            if show_metric_direction:
                axes_pos_per_dir[m_dir].append(ax.get_position())

            y = [mres[m] for mres in metrics_results]
            ax.plot(x, y, label=m)

            ax.set_title(m, fontsize=axes_title_fontsize)

            # set axis labels
            if (param or xaxislabel) and i == len(metric)-1:
                if xaxislabel:
                    ax.set_xlabel(xaxislabel)
                else:
                    ax.set_xlabel(param[-1])
            if yaxislabel:
                ax.set_ylabel(yaxislabel)

        # show grouped metric direction on the left
        if axes_pos_per_dir:   # = if show_metric_direction
            left_xs = []
            ys = []
            for m_dir, bboxes in axes_pos_per_dir.items():
                left_xs.append(min(bb.x0 for bb in bboxes))
                min_y = min(bb.y0 for bb in bboxes)
                max_y = max(bb.y1 for bb in bboxes)
                ys.append((min_y, max_y))

            left_x = min(left_xs) / 2.5

            for (min_y, max_y), m_dir in zip(ys, axes_pos_per_dir.keys()):
                center_y = min_y + (max_y - min_y) / 2

                subfig.text(left_x / 1.5, center_y, m_dir, fontsize=metric_direction_font_size, rotation='vertical',
                            horizontalalignment='right', verticalalignment='center')

    # set adjustments
    subplots_adjust_kwargs = {}

    if show_metric_direction:
        subplots_adjust_kwargs.update({'left': 0.15})

    subplots_adjust_kwargs.update(subplots_adjust_opts or {})

    if subplots_adjust_kwargs:
        fig.subplots_adjust(**subplots_adjust_kwargs)

    if title:
        fig.suptitle(title, fontsize=title_fontsize)

    return fig, subfigs, subfigs_axes


#%% Other functions


def parameters_for_ldavis(topic_word_distrib, doc_topic_distrib, dtm, vocab, sort_topics=False):
    """
    Create a parameters dict that can be used with the
    `pyLDAVis package <https://pyldavis.readthedocs.io/en/latest/readme.html>`_ by passing the dict ``params`` like
    ``pyLDAVis.prepare(**params)``.

    :param topic_word_distrib: topic-word distribution; shape KxM, where K is number of topics, M is vocabulary size
    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents, K is the
                              number of topics
    :param dtm: document-term-matrix; shape NxM
    :param vocab: vocabulary array/list of length M
    :param sort_topics: if True, sort the topics
    :return: dict with parameters ready to use with pyLDAVis
    """
    return dict(
        topic_term_dists=topic_word_distrib,
        doc_topic_dists=doc_topic_distrib,
        vocab=vocab,
        doc_lengths=doc_lengths(dtm),
        term_frequency=term_frequencies(dtm),
        sort_topics=sort_topics,
    )