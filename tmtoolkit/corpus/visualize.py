"""
Functions to visualize corpus summary statistics.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import logging
import re
from typing import Optional, Union, Collection, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from ._corpus import Corpus
from ._corpusfuncs import doc_lengths, vocabulary_counts, doc_frequencies, doc_num_sents, doc_sent_lengths, \
    doc_token_lengths
from ..types import Proportion
from ..utils import flatten_list

logger = logging.getLogger('tmtoolkit')

PTTRN_NUMPY_UFUNC = re.compile(r"^<ufunc '(\w+)'>$")


#%%

def plot_doc_lengths_hist(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                          select: Optional[Union[str, Collection[str]]] = None,
                          y_log: bool = True,
                          title: Optional[str] = 'Histogram of document lengths',
                          xaxislabel: Optional[str] = 'document lengths',
                          yaxislabel: Optional[str] = 'count',
                          **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of document lengths for corpus `docs`.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param y_log: if True, scale y-axis via log10 transformation
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param kwargs: additional keyword arguments passed on to matplotlib histogram plotting function ``ax.hist``
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    # get document lengths
    logger.info('processing document lengths')
    dlengths = doc_lengths(docs, select=select)

    if not dlengths:
        raise ValueError('cannot produce histogram for empty corpus or empty corpus selection')

    x = np.fromiter(dlengths.values(), dtype='uint32', count=len(dlengths))

    # generate plot
    logger.info('producing plot')

    return _plot_hist(fig, ax, x, y_log=y_log,
                      title=title, xaxislabel=xaxislabel, yaxislabel=yaxislabel,
                      **kwargs)


def plot_vocab_counts_hist(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                           select: Optional[Union[str, Collection[str]]] = None,
                           y_log: bool = True,
                           title: Optional[str] = 'Histogram for number of occurrences per token type',
                           xaxislabel: Optional[str] = 'number of occurrences per token type',
                           yaxislabel: Optional[str] = 'count',
                           **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of vocabulary counts (i.e. number of occurrences per token type) for corpus `docs`.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param y_log: if True, scale y-axis via log10 transformation
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param kwargs: additional keyword arguments passed on to matplotlib histogram plotting function ``ax.hist``
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    # get document lengths
    logger.info('processing vocabulary counts')
    vocabcounts = vocabulary_counts(docs, select=select, tokens_as_hashes=not docs.uses_unigrams,
                                    convert_uint64hashes=False)

    if not vocabcounts:
        raise ValueError('cannot produce histogram for empty corpus or empty corpus selection')

    x = np.fromiter(vocabcounts.values(), dtype='uint32', count=len(vocabcounts))

    # generate plot
    logger.info('producing plot')

    return _plot_hist(fig, ax, x, y_log=y_log,
                      title=title, xaxislabel=xaxislabel, yaxislabel=yaxislabel,
                      **kwargs)


def plot_doc_frequencies_hist(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                              select: Optional[Union[str, Collection[str]]] = None,
                              proportions: Proportion = Proportion.NO,
                              y_log: bool = True,
                              title: Optional[str] = 'Histogram of document frequencies',
                              xaxislabel: Optional[str] = 'document frequency',
                              yaxislabel: Optional[str] = 'count',
                              **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of document frequencies for corpus `docs`.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param proportions: one of :attr:`~tmtoolkit.types.Proportion`: ``NO (0)`` – return counts; ``YES (1)`` – return
                        proportions; ``LOG (2)`` – return log10 of proportions
    :param y_log: if True, scale y-axis via log10 transformation
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param kwargs: additional keyword arguments passed on to matplotlib histogram plotting function ``ax.hist``
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    # get document lengths
    logger.info('processing document frequencies')
    dfreqs = doc_frequencies(docs, select=select, proportions=proportions)

    if not dfreqs:
        raise ValueError('cannot produce histogram for empty corpus or empty corpus selection')

    x = np.fromiter(dfreqs.values(), dtype='uint32' if proportions == Proportion.NO else 'float64',
                    count=len(dfreqs))

    # generate plot
    logger.info('producing plot')

    if proportions == Proportion.LOG:
        xaxislabel += ' (log scale)'

    return _plot_hist(fig, ax, x, y_log=y_log, title=title, xaxislabel=xaxislabel, yaxislabel=yaxislabel,
                      **kwargs)


def plot_num_sents_hist(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                        select: Optional[Union[str, Collection[str]]] = None,
                        y_log: bool = True,
                        title: Optional[str] = 'Histogram of number of sentences per document',
                        xaxislabel: Optional[str] = 'number of sentences',
                        yaxislabel: Optional[str] = 'count',
                        **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of number of sentences per document of corpus `docs`.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param y_log: if True, scale y-axis via log10 transformation
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param kwargs: additional keyword arguments passed on to matplotlib histogram plotting function ``ax.hist``
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    # get document lengths
    logger.info('processing number of sentences')
    dfreqs = doc_num_sents(docs, select=select)

    if not dfreqs:
        raise ValueError('cannot produce histogram for empty corpus or empty corpus selection')

    x = np.fromiter(dfreqs.values(), dtype='uint32', count=len(dfreqs))

    # generate plot
    logger.info('producing plot')

    return _plot_hist(fig, ax, x, y_log=y_log, title=title, xaxislabel=xaxislabel, yaxislabel=yaxislabel,
                      **kwargs)


def plot_sent_lengths_hist(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                           select: Optional[Union[str, Collection[str]]] = None,
                           y_log: bool = True,
                           title: Optional[str] = 'Histogram of sentence lengths',
                           xaxislabel: Optional[str] = 'sentence length',
                           yaxislabel: Optional[str] = 'count',
                           **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of sentence lengths in corpus `docs`.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param y_log: if True, scale y-axis via log10 transformation
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param kwargs: additional keyword arguments passed on to matplotlib histogram plotting function ``ax.hist``
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    # get document lengths
    logger.info('processing sentence lengths')
    sentlengths = doc_sent_lengths(docs, select=select)

    if not sentlengths:
        raise ValueError('cannot produce histogram for empty corpus or empty corpus selection')

    x = np.array(flatten_list(sentlengths.values()), dtype='uint32')

    # generate plot
    logger.info('producing plot')

    return _plot_hist(fig, ax, x, y_log=y_log, title=title, xaxislabel=xaxislabel, yaxislabel=yaxislabel,
                      **kwargs)


def plot_token_lengths_hist(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                            select: Optional[Union[str, Collection[str]]] = None,
                            y_log: bool = True,
                            title: Optional[str] = 'Histogram of token lengths',
                            xaxislabel: Optional[str] = 'token length',
                            yaxislabel: Optional[str] = 'count',
                            **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of sentence lengths in corpus `docs`.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param y_log: if True, scale y-axis via log10 transformation
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param kwargs: additional keyword arguments passed on to matplotlib histogram plotting function ``ax.hist``
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    # get document lengths
    logger.info('processing token lengths')
    toklengths = doc_token_lengths(docs, select=select)

    if not toklengths:
        raise ValueError('cannot produce histogram for empty corpus or empty corpus selection')

    x = np.array(flatten_list(toklengths.values()), dtype='uint32')

    # generate plot
    logger.info('producing plot')

    return _plot_hist(fig, ax, x, y_log=y_log, title=title, xaxislabel=xaxislabel, yaxislabel=yaxislabel,
                      **kwargs)


def plot_num_sents_vs_sent_length(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                                  select: Optional[Union[str, Collection[str]]] = None,
                                  min_n_sents: int = 0,
                                  x_log: bool = False,
                                  y_log: bool = False,
                                  title: Optional[str] = 'Number of sentences vs. mean sentence length',
                                  xaxislabel: Optional[str] = 'number of documents',
                                  yaxislabel: Optional[str] = 'sentence length',
                                  **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Make scatter plot of number of sentences vs. mean sentence length in corpus `docs`.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param min_n_sents: plot only mean sentence lengths for documents with at least `min_n_sents` sentences
    :param x_log: if True, scale x-axis via log10 transformation
    :param y_log: if True, scale y-axis via log10 transformation
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param kwargs: additional keyword arguments passed on to matplotlib histogram plotting function ``ax.hist``
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    # get document lengths
    logger.info('processing sentence lengths')
    sentlengths = doc_sent_lengths(docs, select=select)

    if not sentlengths:
        raise ValueError('cannot produce histogram for empty corpus or empty corpus selection')

    x = []
    y = []
    for sents in sentlengths.values():
        n_sents = len(sents)
        if n_sents > min_n_sents:
            x.append(n_sents)
            y.append(np.mean(sents))

    if x_log:
        x = np.log10(x)
    else:
        x = np.array(x, dtype='uint32')

    if y_log:
        y = np.log10(y)
    else:
        y = np.array(y, dtype='float64')

    # generate plot
    logger.info('producing plot')

    # set title
    if title:
        ax.set_title(title)

    # plot as scatter plot
    _plot_opts = {'alpha': 0.3, 'edgecolors': 'white'}
    if kwargs:
        _plot_opts.update(kwargs)
    ax.scatter(x, y, **_plot_opts)

    # customize axes
    if xaxislabel:
        ax.set_xlabel(_add_axis_scale_info(xaxislabel, x_log))

    if yaxislabel:
        ax.set_ylabel(_add_axis_scale_info(yaxislabel, y_log))

    return fig, ax


def plot_ranked_vocab_counts(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                             select: Optional[Union[str, Collection[str]]] = None,
                             x_log: bool = True,
                             y_log: bool = True,
                             zipf: bool = False,
                             title: Optional[str] = 'Scatter plot for vocabulary term count vs. rank',
                             xaxislabel: Optional[str] = 'rank',
                             yaxislabel: Optional[str] = 'count',
                             hist_opts: Optional[Dict[str, Any]] = None,
                             plot_opts: Optional[Dict[str, Any]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Make scatter plot for vocabulary term count vs. rank and optionally overlay with theoretical distribution from
    Zipf's law via `zipf=True`.

    :param fig: matplotlib Figure object
    :param ax: matplotlib Axes object
    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param x_log: if True, scale x-axis via log10 transformation
    :param y_log: if True, scale y-axis via log10 transformation
    :param zipf: if True, add red dashed line indicating theoretical frequencies according to Zipf's law
    :param title: plot title
    :param xaxislabel: x-axis label
    :param yaxislabel: y-axis label
    :param hist_opts: additional keyword arguments passed on to histogram binning function ``np.histogram``
    :param plot_opts: additional keyword arguments passed on to respective matplotlib plotting function
    :return: tuple of generated (matplotlib Figure object, matplotlib Axes object)
    """

    _plot_opts = {'alpha': 0.3, 'edgecolors': 'white'}
    if plot_opts:
        _plot_opts.update(plot_opts)

    _hist_opts = {'bins': 'auto'}
    if hist_opts:
        _hist_opts.update(hist_opts)

    del plot_opts, hist_opts

    # get document lengths
    logger.info('processing vocabulary counts')
    vocabcounts = vocabulary_counts(docs, select=select, tokens_as_hashes=not docs.uses_unigrams,
                                    convert_uint64hashes=False)

    if not vocabcounts:
        raise ValueError('cannot produce histogram for empty corpus or empty corpus selection')

    y = np.fromiter(vocabcounts.values(), dtype='uint32', count=len(vocabcounts))
    y = -np.sort(-y)   # sort in descending order

    x = np.arange(1, len(y)+1)   # rank

    if x_log:
        x = np.log10(x)

    if y_log:
        y = np.log10(y)

    # generate plot
    logger.info('producing plot')

    # set title
    if title:
        ax.set_title(title)

    # plot as scatter plot
    ax.scatter(x, y, **_plot_opts)

    if zipf:
        # Zipf's law scaled to natural frequencies with `N_max` being the maximum count and rank `k`:
        # y = N_max/k  <=> log y = log N_max - log k

        if x_log and not y_log:   # sequence in log space (otherwise to few points on lower end of x-axis)
            zipf_x = np.logspace(0, np.log10(len(y)), num=1000)
        else:
            zipf_x = np.linspace(1, len(y)+1, num=1000)

        if x_log and y_log:
            zipf_x = np.log10(zipf_x)

        if y_log:
            zipf_y = np.max(y) - (zipf_x if x_log and y_log else np.log10(zipf_x))
        else:
            zipf_y = np.max(y) / zipf_x

        if x_log and not y_log:
            zipf_x = np.log10(zipf_x)

        # plot as line
        ax.plot(zipf_x, zipf_y, c='red', linestyle='dashed')

    # customize axes
    if xaxislabel:
        ax.set_xlabel(_add_axis_scale_info(xaxislabel, x_log))

    if yaxislabel:
        ax.set_ylabel(_add_axis_scale_info(yaxislabel, y_log))

    return fig, ax


#%% helper functions


def _add_axis_scale_info(axislbl: str, log: bool):
    if log:
        return axislbl + ' (log10 scale)'

    return axislbl


def _plot_hist(fig: plt.Figure, ax: plt.Axes, x: np.ndarray,
               y_log: bool = False,
               title: Optional[str] = None,
               xaxislabel: Optional[str] = None,
               yaxislabel: Optional[str] = None,
               **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Helper function for producting a histogram.
    """

    kwargs = kwargs or {}
    kwargs.update({'log': y_log})

    # set title
    if title:
        ax.set_title(title)

    ax.hist(x, **kwargs)

    # customize axes
    if xaxislabel:
        ax.set_xlabel(xaxislabel)

    if yaxislabel:
        ax.set_ylabel(_add_axis_scale_info(yaxislabel, y_log))

    return fig, ax
