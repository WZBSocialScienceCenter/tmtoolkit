# -*- coding: utf-8 -*-
"""
Functions to visualize topic models and topic model evaluation results.

Markus Konrad <markus.konrad@wzb.eu>
"""

from __future__ import division
import os
import logging
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tmtoolkit.topicmod.model_stats import get_doc_lengths, get_term_frequencies, top_n_from_distribution
from tmtoolkit.topicmod import evaluate
from tmtoolkit.utils import mat2d_window_from_indices

logger = logging.getLogger('tmtoolkit')


#%% word clouds from topic models


def _wordcloud_color_func_black(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'rgb(0,0,0)'


DEFAULT_WORDCLOUD_KWARGS = {   # default wordcloud settings for transparent background and black font
    'width': 800,
    'height': 600,
    'mode': 'RGBA',
    'background_color': None,
    'color_func': _wordcloud_color_func_black
}


def write_wordclouds_to_folder(wordclouds, folder, file_name_fmt='{label}.png', **save_kwargs):
    if not os.path.exists(folder):
        raise ValueError('target folder `%s` does not exist' % folder)

    for label, wc in wordclouds.items():
        file_name = file_name_fmt.format(label=label)
        file_path = os.path.join(folder, file_name)
        logger.info('writing wordcloud to file `%s`' % file_path)

        wc.save(file_path, **save_kwargs)


def generate_wordclouds_for_topic_words(phi, vocab, top_n, topic_labels='topic_{i1}', which_topics=None,
                                        return_images=True, **wordcloud_kwargs):
    return generate_wordclouds_from_distribution(phi, row_labels=topic_labels, val_labels=vocab, top_n=top_n,
                                                 which_rows=which_topics, return_images=return_images,
                                                 **wordcloud_kwargs)


def generate_wordclouds_for_document_topics(theta, doc_labels, top_n, topic_labels='topic_{i1}', which_documents=None,
                                            return_images=True, **wordcloud_kwargs):
    return generate_wordclouds_from_distribution(theta, row_labels=doc_labels, val_labels=topic_labels, top_n=top_n,
                                                 which_rows=which_documents, return_images=return_images,
                                                 **wordcloud_kwargs)


def generate_wordclouds_from_distribution(distrib, row_labels, val_labels, top_n, which_rows=None, return_images=True,
                                          **wordcloud_kwargs):
    prob = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=None)
    words = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=val_labels)

    if which_rows:
        prob = prob.loc[which_rows, :]
        words = words.loc[which_rows, :]

        assert prob.shape == words.shape

    wordclouds = {}
    for (p_row_name, p), (w_row_name, w) in zip(prob.iterrows(), words.iterrows()):
        assert p_row_name == w_row_name
        logger.info('generating wordcloud for `%s`' % p_row_name)
        wc = generate_wordcloud_from_probabilities_and_words(p, w,
                                                             return_image=return_images,
                                                             **wordcloud_kwargs)
        wordclouds[p_row_name] = wc

    return wordclouds


def generate_wordcloud_from_probabilities_and_words(prob, words, return_image=True, wordcloud_instance=None,
                                                    **wordcloud_kwargs):
    if len(prob) != len(words):
        raise ValueError('`distrib` and `labels` must have the name length')
    if hasattr(prob, 'ndim') and prob.ndim != 1:
        raise ValueError('`distrib` must be a 1D array or sequence')
    if hasattr(words, 'ndim') and words.ndim != 1:
        raise ValueError('`labels` must be a 1D array or sequence')

    weights = dict(zip(words, prob))

    return generate_wordcloud_from_weights(weights, return_image=return_image,
                                           wordcloud_instance=wordcloud_instance, **wordcloud_kwargs)


def generate_wordcloud_from_weights(weights, return_image=True, wordcloud_instance=None, **wordcloud_kwargs):
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

#%% plot heatmaps (especially for doc-topic distribution)


def plot_doc_topic_heatmap(fig, ax, doc_topic_distrib, doc_labels, topic_labels=None,
                           which_documents=None, which_document_indices=None,
                           which_topics=None, which_topic_indices=None,
                           xaxislabel=None, yaxislabel=None,
                           **kwargs):
    """
    Plot a heatmap for a document-topic distribution `doc_topic_distrib` to a matplotlib Figure `fig` and Axes `ax`
    using `doc_labels` as document labels on the y-axis and topics from 1 to `n_topics=doc_topic_distrib.shape[1]` on
    the x-axis.
    Custom topic labels can be passed as `topic_labels`.
    A subset of documents can be specified either with a sequence `which_documents` containing a subset of document
    labels from `doc_labels` or `which_document_indices` containing a sequence of document indices.
    A subset of topics can be specified either with a sequence `which_topics` containing sequence of numbers between
    [1, n_topics] or `which_topic_indices` which is a number between [0, n_topics-1]
    Additional arguments can be passed via `kwargs` to `plot_heatmap`.

    Please note that it is almost always necessary to select a subset of your document-topic distribution with the
    `which_documents` or `which_topics` parameters, as otherwise the amount of data to be plotted will be too high
    to give a reasonable picture.
    """
    if which_documents is not None and which_document_indices is not None:
        raise ValueError('only `which_documents` or `which_document_indices` can be set, not both')

    if which_topics is not None and which_topic_indices is not None:
        raise ValueError('only `which_topics` or `which_topic_indices` can be set, not both')

    if which_documents is not None:
        which_document_indices = np.where(np.isin(doc_labels, which_documents))[0]

    if which_topics is not None:
        which_topic_indices = np.array(which_topics) - 1

    select_distrib_subset = False

    if topic_labels is None:
        topic_labels = np.array(range(1, doc_topic_distrib.shape[1]+1))
    elif not isinstance(topic_labels, np.ndarray):
        topic_labels = np.array(topic_labels)

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


def plot_topic_word_heatmap(fig, ax, topic_word_distrib, vocab,
                            which_topics=None, which_topic_indices=None,
                            which_words=None, which_word_indices=None,
                            xaxislabel=None, yaxislabel=None,
                            **kwargs):
    """
    Plot a heatmap for a topic-word distribution `topic_word_distrib` to a matplotlib Figure `fig` and Axes `ax`
    using `vocab` as vocabulary on the x-axis and topics from 1 to `n_topics=doc_topic_distrib.shape[1]` on
    the y-axis.
    A subset of words from `vocab` can be specified either directly with a sequence `which_words` or
    `which_document_indices` containing a sequence of word indices in `vocab`.
    A subset of topics can be specified either with a sequence `which_topics` containing sequence of numbers between
    [1, n_topics] or `which_topic_indices` which is a number between [0, n_topics-1]
    Additional arguments can be passed via `kwargs` to `plot_heatmap`.

    Please note that it is almost always necessary to select a subset of your topic-word distribution with the
    `which_words` or `which_topics` parameters, as otherwise the amount of data to be plotted will be too high
    to give a reasonable picture.
    """
    if which_topics is not None and which_topic_indices is not None:
        raise ValueError('only `which_topics` or `which_topic_indices` can be set, not both')

    if which_words is not None and which_word_indices is not None:
        raise ValueError('only `which_words` or `which_word_indices` can be set, not both')

    if which_topics is not None:
        which_topic_indices = np.array(which_topics) - 1

    if which_words is not None:
        which_word_indices = np.where(np.isin(vocab, which_words))[0]

    select_distrib_subset = False
    topic_labels = np.array(range(1, topic_word_distrib.shape[0]+1))

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
    """"
    helper function to plot a heatmap for a 2D matrix `data` using matplotlib's "matshow" function
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


def plot_eval_results(eval_results, metric=None, xaxislabel=None, yaxislabel=None,
                      title=None, title_fontsize='x-large', axes_title_fontsize='large',
                      show_metric_direction=True, metric_direction_font_size='large',
                      subplots_opts=None, subplots_adjust_opts=None, figsize='auto',
                      **fig_kwargs):
    """
    Plot the evaluation results from `eval_results`. `eval_results` must be a sequence containing `(param, values)`
    tuples, where `param` is the parameter value to appear on the x axis and `values` can be a dict structure
    containing the metric values. `eval_results` can be created using the `results_by_parameter` function from the
    `topicmod.common` module.
    Set `metric` to plot only a specific metric.
    Set `xaxislabel` for a label on the x-axis.
    Set `yaxislabel` for a label on the y-axis.
    Set `title` for a plot title.
    Options in a dict `subplots_opts` will be passed to `plt.subplots(...)`.
    Options in a dict `subplots_adjust_opts` will be passed to `fig.subplots_adjust(...)`.
    `figsize` can be set to a tuple `(width, height)` or to `"auto"` (default) which will set the size to
    `(8, 2 * <num. of metrics>)`.
    """
    if type(eval_results) not in (list, tuple) or not eval_results:
        raise ValueError('`eval_results` must be a list or tuple with at least one element')

    if type(eval_results[0]) not in (list, tuple) or len(eval_results[0]) != 2:
        raise ValueError('`eval_results` must be a list or tuple containing a (param, values) tuple. '
                         'Maybe `eval_results` must be converted with `results_by_parameter`.')

    if metric is not None and type(metric) not in (list, tuple):
        metric = [metric]
    elif metric is None:
        # remove special evaluation result 'model': the calculated model itself
        metric = list(set(next(iter(eval_results))[1].keys()) - {'model'})

    metric = sorted(metric)

    metric_direction = []
    for m in metric:
        m_fn_name = 'metric_%s' % (m[:16] if m.startswith('coherence_gensim') else m)
        m_fn = getattr(evaluate, m_fn_name, None)
        if m_fn:
            metric_direction.append(getattr(m_fn, 'direction', None))
        else:
            metric_direction.append(None)

    n_metrics = len(metric)

    assert n_metrics == len(metric_direction)

    metrics_ordered = []
    for m_dir in sorted(set(metric_direction), reverse=True):
        metrics_ordered.extend([(m, d) for m, d in zip(metric, metric_direction) if d == m_dir])

    assert n_metrics == len(metrics_ordered)

    # get figure and subplots (axes)
    if figsize == 'auto':
        figsize = (8, 2*n_metrics)

    subplots_kwargs = dict(nrows=n_metrics, ncols=1, sharex=True, constrained_layout=True, figsize=figsize)
    subplots_kwargs.update(subplots_opts or {})
    subplots_kwargs.update(fig_kwargs)

    fig, axes = plt.subplots(**subplots_kwargs)

    # set title
    if title:
        fig.suptitle(title, fontsize=title_fontsize)

    x = list(zip(*eval_results))[0]

    # set adjustments
    if title:
        subplots_adjust_kwargs = dict(top=0.9, hspace=0.3)
    else:
        subplots_adjust_kwargs = {}

    subplots_adjust_kwargs.update(subplots_adjust_opts or {})

    if subplots_adjust_kwargs:
        fig.subplots_adjust(**subplots_adjust_kwargs)

    # draw subplot for each metric
    axes_pos_per_dir = defaultdict(list)
    for i, (ax, (m, m_dir)) in enumerate(zip(axes.flatten(), metrics_ordered)):
        if show_metric_direction:
            axes_pos_per_dir[m_dir].append(ax.get_position())

        y = [metric_res[m] for _, metric_res in eval_results]
        ax.plot(x, y, label=m)

        ax.set_title(m, fontsize=axes_title_fontsize)

        # set axis labels
        if xaxislabel and i == len(metric)-1:
            ax.set_xlabel(xaxislabel)
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

        fig.lines = []
        for (min_y, max_y), m_dir in zip(ys, axes_pos_per_dir.keys()):
            center_y = min_y + (max_y - min_y) / 2

            fig.lines.append(Line2D((left_x, left_x), (min_y, max_y), transform=fig.transFigure, linewidth=5,
                                    color='lightgray'))

            fig.text(left_x / 1.5, center_y, m_dir, fontsize=metric_direction_font_size, rotation='vertical',
                     horizontalalignment='right', verticalalignment='center')

    return fig, axes


#%% Other functions


def parameters_for_ldavis(topic_word_distrib, doc_topic_distrib, dtm, vocab, sort_topics=False):
    return dict(
        topic_term_dists=topic_word_distrib,
        doc_topic_dists=doc_topic_distrib,
        vocab=vocab,
        doc_lengths=get_doc_lengths(dtm),
        term_frequency=get_term_frequencies(dtm),
        sort_topics=sort_topics,
    )