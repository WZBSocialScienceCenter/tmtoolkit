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
from ._corpusfuncs import doc_lengths, vocabulary_counts

logger = logging.getLogger('tmtoolkit')

PTTRN_NUMPY_UFUNC = re.compile(r"^<ufunc '(\w+)'>$")


#%%

def plot_doc_lengths_hist(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                          select: Optional[Union[str, Collection[str]]] = None,
                          y_log: bool = False,
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


def plot_vocab_counts_scatter(fig: plt.Figure, ax: plt.Axes, docs: Corpus,
                              select: Optional[Union[str, Collection[str]]] = None,
                              x_log: bool = True,
                              y_log: bool = True,
                              zipf: bool = False,
                              title: Optional[str] = 'Scatter plot for number of occurrences per token type',
                              xaxislabel: Optional[str] = 'number of occurrences per token type',
                              yaxislabel: Optional[str] = 'count',
                              hist_opts: Optional[Dict[str, Any]] = None,
                              plot_opts: Optional[Dict[str, Any]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of vocabulary counts (i.e. number of occurrences per token type) for corpus `docs`.

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

    x = np.fromiter(vocabcounts.values(), dtype='uint32', count=len(vocabcounts))
    max_x = np.max(x)

    if x_log:
        x = np.log10(x)

    # bin the counts using histogram
    counts, bins = np.histogram(x, **_hist_opts)
    bins, counts = bins[:-1][counts > 0], counts[counts > 0]   # only keep filled bins

    if y_log:
        counts = np.log10(counts)

    # generate plot
    logger.info('producing plot')

    # set title
    if title:
        ax.set_title(title)

    # plot as scatter plot
    ax.scatter(bins, counts, **_plot_opts)

    if zipf:
        # Zipf's law scaled to natural frequencies with `max(H)` being the maximum count in histogram H and rank `k`:
        # y = max(H)/k  <=>
        # log y = log max(H) - log k

        if x_log and not y_log:   # sequence in log space (otherwise to few points on lower end of x-axis)
            zipf_x = np.logspace(0, np.log10(max_x), num=1000)
        else:
            zipf_x = np.linspace(1, max_x, num=1000)

        if x_log and y_log:
            zipf_x = np.log10(zipf_x)

        if y_log:
            zipf_y = np.max(counts) - (zipf_x if x_log and y_log else np.log10(zipf_x))
        else:
            zipf_y = np.max(counts) / zipf_x

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
