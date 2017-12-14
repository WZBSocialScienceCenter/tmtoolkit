import os
import logging

import numpy as np

from tmtoolkit.utils import mat2d_window_from_indices
from tmtoolkit.lda_utils.common import top_n_from_distribution


logger = logging.getLogger('tmtoolkit')


#
# word clouds from topic models
#

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

#
# plot heatmaps (especially for doc-topic distribution)
#


def plot_doc_topic_heatmap(fig, ax, doc_topic_distrib, doc_labels,
                           which_documents=None, which_document_indices=None,
                           which_topics=None, which_topic_indices=None,
                           xaxislabel=None, yaxislabel=None,
                           **kwargs):
    """
    Plot a heatmap for a document-topic distribution `doc_topic_distrib` to a matplotlib Figure `fig` and Axes `ax`
    using `doc_labels` as document labels on the y-axis and topics from 1 to `n_topics=doc_topic_distrib.shape[1]` on
    the x-axis.
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
    topic_labels = np.array(range(1, doc_topic_distrib.shape[1]+1))

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
        ax.set_xticklabels(xticklabels, rotation=45)
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


#
# plotting of evaluation results
#


def plot_eval_results(fig, ax, eval_results, metric=None, normalize_y=None,
                      xaxislabel=None, yaxislabel=None, title=None):
    """
    Plot the evaluation results from `eval_results` to a matplotlib Figure `fig` and Axes `ax`. `eval_results` must be
    a sequence containing `(param, values)` tuples, where `param` is the parameter value to appear on the x axis and
    `values` can be a dict structure containing the metric values. `eval_results` can be created using the
    `results_by_parameter` function from the lda_utils.common module.
    Set `metric` to plot only a specific metric.
    Set `normalize_y` to True or False to either normalize metric values to [0,1] (or [-1,0] if all-negative) or not.
    Set `xaxislabel` for a label on the x-axis.
    Set `yaxislabel` for a label on the y-axis.
    Set `title` for a plot title.
    """
    if type(eval_results) not in (list, tuple) or not eval_results:
        raise ValueError('`eval_results` must be a list or tuple with at least one element')

    if type(eval_results[0]) not in (list, tuple) or len(eval_results[0]) != 2:
        raise ValueError('`eval_results` must be a list or tuple containing a (param, values) tuple. '
                         'Maybe `eval_results` must be converted with `results_by_parameter`.')

    if normalize_y is None:
        normalize_y = metric is None

    if metric == 'cross_validation':   # this is currently not really supported
        plotting_res = []
        for k, folds in eval_results:
            plotting_res.extend([(k, val, f) for f, val in enumerate(folds)])
        x, y, f = zip(*plotting_res)
        ax.scatter(x, y, c=f, alpha=0.5)
    else:
        if metric is not None and type(metric) not in (list, tuple):
            metric = [metric]
        elif metric is None:
            # remove special evaluation result 'model': the calculated model itself
            all_metrics = set(next(iter(eval_results))[1].keys()) - {'model'}
            metric = sorted(all_metrics)

        if normalize_y:
            res_per_metric = {}
            for m in metric:
                params = list(zip(*eval_results))[0]
                unnorm = np.array([metric_res[m] for _, metric_res in eval_results])
                unnorm_nonnan = unnorm[~np.isnan(unnorm)]
                vals_max = np.max(unnorm_nonnan)
                vals_min = np.min(unnorm_nonnan)

                if vals_max != vals_min:
                    rng = vals_max - vals_min
                else:
                    rng = 1.0   # avoid division by zero

                if vals_max < 0:
                    norm = -(vals_max - unnorm) / rng
                else:
                    norm = (unnorm - vals_min) / rng
                res_per_metric[m] = dict(zip(params, norm))

            eval_results_tmp = []
            for k, _ in eval_results:
                metric_res = {}
                for m in metric:
                    metric_res[m] = res_per_metric[m][k]
                eval_results_tmp.append((k, metric_res))
            eval_results = eval_results_tmp

        x = list(zip(*eval_results))[0]
        for m in metric:
            y = [metric_res[m] for _, metric_res in eval_results]
            ax.plot(x, y, label=m)

        # set axis labels
        if xaxislabel:
            ax.set_xlabel(xaxislabel)
        if yaxislabel:
            ax.set_ylabel(yaxislabel)

        # set title
        if title:
            ax.set_title(title)

        # set legend
        ax.legend(loc='best')

    return fig, ax
