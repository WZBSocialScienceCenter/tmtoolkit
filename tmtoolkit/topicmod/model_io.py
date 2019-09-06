"""
Functions for printing/exporting topic model results.
"""

from collections import OrderedDict

import numpy as np
import datatable as dt

from tmtoolkit.topicmod.model_stats import get_marginal_topic_distrib, top_n_from_distribution, \
    _join_value_and_label_dfs
from tmtoolkit.bow.bow_stats import doc_lengths
from tmtoolkit.utils import pickle_data, unpickle_file
from tmtoolkit.topicmod._common import DEFAULT_RANK_NAME_FMT, DEFAULT_TOPIC_NAME_FMT


def ldamodel_top_topic_words(topic_word_distrib, vocab, top_n=10, val_fmt=None, col_labels=None, index_name=None):
    """
    Retrieve the top (i.e. most probable) `top_n` words for each topic in the topic-word distribution
    `topic_word_distrib` as pandas DataFrame.

    .. seealso:: :func:`~tmtoolkit.topicmod.model_io.ldamodel_full_topic_words` to retrieve the full distribution as
                 formatted pandas DataFrame; :func:`~tmtoolkit.topicmod.model_io.ldamodel_top_doc_topics` to retrieve
                 the top topics per document from a document-topic distribution

    :param topic_word_distrib: topic-word distribtion; shape KxM, where K is number of topics, M is vocabulary size
    :param vocab: vocabulary list/array of length K
    :param top_n: number of most probable words per topic to select
    :param val_fmt: format string for table cells where ``{lbl}`` is replaced by the respective word from `vocab` and
                    ``{val}`` is replaced by the word's probability given the topic
    :param col_labels: format string for the columns where ``{i0}`` or ``{i1}`` are replaced by the respective zero- or
                       one-indexed rank
    :param index_name: name of the table index
    :return: pandas DataFrame
    """
    df_values = top_n_from_distribution(topic_word_distrib, top_n=top_n,
                                        row_labels=DEFAULT_TOPIC_NAME_FMT, val_labels=None)
    df_labels = top_n_from_distribution(topic_word_distrib, top_n=top_n,
                                        row_labels=DEFAULT_TOPIC_NAME_FMT, val_labels=vocab)
    return _join_value_and_label_dfs(df_values, df_labels, top_n, row_labels=DEFAULT_TOPIC_NAME_FMT,
                                     val_fmt=val_fmt, col_labels=col_labels, index_name=index_name or 'topic')


def ldamodel_top_doc_topics(doc_topic_distrib, doc_labels, top_n=3, val_fmt=None, col_labels=None, index_name=None):
    """
    Retrieve the top (i.e. most probable) `top_n` topics for each document in the document-topic distribution
    `doc_topic_distrib` as pandas DataFrame.

    .. seealso:: :func:`~tmtoolkit.topicmod.model_io.ldamodel_full_doc_topics` to retrieve the full distribution as
                 formatted pandas DataFrame; :func:`~tmtoolkit.topicmod.model_io.ldamodel_top_topic_words` to retrieve
                 the top words per topic from a topic-word distribution

    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents, K is the
                              number of topics
    :param doc_labels: list/array of length N with a string label for each document
    :param top_n: number of most probable topics per document to select
    :param val_fmt: format string for table cells where ``{lbl}`` is replaced by the respective topic name and
                    ``{val}`` is replaced by the topic's probability given the document
    :param col_labels: format string for the columns where ``{i0}`` or ``{i1}`` are replaced by the respective zero- or
                       one-indexed rank
    :param index_name: name of the table index
    :return: pandas DataFrame
    """
    df_values = top_n_from_distribution(doc_topic_distrib, top_n=top_n,
                                        row_labels=doc_labels, val_labels=None)
    df_labels = top_n_from_distribution(doc_topic_distrib, top_n=top_n,
                                        row_labels=doc_labels, val_labels=DEFAULT_TOPIC_NAME_FMT)
    return _join_value_and_label_dfs(df_values, df_labels, top_n, row_labels=doc_labels,
                                     val_fmt=val_fmt, col_labels=col_labels, index_name=index_name)


def ldamodel_full_topic_words(topic_word_distrib, vocab, colname_rowindex='_topic',
                              fmt_rowindex=DEFAULT_TOPIC_NAME_FMT):
    """
    Generate a datatable Frame for the full topic-word distribution `topic_word_distrib`.

    .. note:: This generates a datatable Frame instead of a pandas DataFrame because datatable is much faster when
              dealing with large data as it is often the case with the full distribution.

    .. seealso:: :func:`~tmtoolkit.topicmod.model_io.ldamodel_top_topic_words` to retrieve only the most probable words
                 in the distribution as formatted pandas DataFrame;
                 :func:`~tmtoolkit.topicmod.model_io.ldamodel_full_doc_topics` to retrieve the full document-topic
                 distribution as datatable Frame

    :param topic_word_distrib: topic-word distribtion; shape KxM, where K is number of topics, M is vocabulary size
    :param vocab: vocabulary list/array of length K
    :param colname_rowindex: column name for the "row index", i.e. the column that identifies each row
    :param fmt_rowindex: format string for each row index where ``{i0}`` or ``{i1}`` are replaced by the respective
                         zero- or one-indexed topic numbers
    :return: datatable Frame
    """
    if fmt_rowindex:
        rownames = [fmt_rowindex.format(i0=i, i1=i + 1) for i in range(topic_word_distrib.shape[0])]
    else:
        rownames = None

    return dt.cbind(dt.Frame({colname_rowindex: rownames}),
                    dt.Frame(topic_word_distrib, names=list(vocab)))


def ldamodel_full_doc_topics(doc_topic_distrib, doc_labels, colname_rowindex='_doc',
                             fmt_colnames=DEFAULT_TOPIC_NAME_FMT):
    """
    Generate a datatable Frame for the full doc-topic distribution `doc_topic_distrib`.

    .. note:: This generates a datatable Frame instead of a pandas DataFrame because datatable is much faster when
              dealing with large data as it is often the case with the full distribution.

    .. seealso:: :func:`~tmtoolkit.topicmod.model_io.ldamodel_top_doc_topics` to retrieve only the most probable topics
                 in the distribution as formatted pandas DataFrame;
                 :func:`~tmtoolkit.topicmod.model_io.ldamodel_full_topic_words` to retrieve the full topic-word
                 distribution as datatable Frame

    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents, K is the
                              number of topics
    :param doc_labels: list/array of length N with a string label for each document
    :param colname_rowindex: column name for the "row index", i.e. the column that identifies each row
    :param fmt_colnames: format string for each column name where ``{i0}`` or ``{i1}`` are replaced by the respective
                         zero- or one-indexed topic numbers
    :return: datatable Frame
    """
    if fmt_colnames:
        colnames = [fmt_colnames.format(i0=i, i1=i+1) for i in range(doc_topic_distrib.shape[1])]
    else:
        colnames = None

    return dt.cbind(dt.Frame({colname_rowindex: doc_labels}),
                    dt.Frame(doc_topic_distrib, names=list(colnames)))


def print_ldamodel_distribution(distrib, row_labels, val_labels, top_n=10):
    """
    Print `top_n` top values from a LDA model's distribution `distrib`. This is a general function to print top values
    of any multivariate distribution given as matrix `distrib` with H rows and I columns, each identified by
    H `row_labels` and I `val_labels`.

    .. seealso:: :func:`~tmtoolkit.topicmod.model_io.print_ldamodel_topic_words` to print the top values of a
                 topic-word distribution or :func:`~tmtoolkit.topicmod.model_io.print_ldamodel_doc_topics`
                 to print the top values of a document-topic distribution.

    :param distrib: either a topic-word or a document-topic distribution of shape HxI
    :param row_labels: list/array of length H with label string for each row of `distrib` or format string
    :param val_labels: list/array of length I with label string for each column of `distrib` or format string
    :param top_n: number of top values to print
    """

    df_values = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=None)
    df_labels = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=val_labels)

    for i, (ind, row) in enumerate(df_labels.iterrows()):
        print(ind)
        for j, label in enumerate(row):
            val = df_values.iloc[i, j]
            print('> #%d. %s (%f)' % (j + 1, label, val))


def print_ldamodel_topic_words(topic_word_distrib, vocab, top_n=10, row_labels=DEFAULT_TOPIC_NAME_FMT):
    """
    Print `top_n` values from an LDA model's topic-word distribution `topic_word_distrib`.

    .. seealso:: :func:`~tmtoolkit.topicmod.model_io.print_ldamodel_doc_topics`
                 to print the top values of a document-topic distribution.

    :param topic_word_distrib: topic-word distribtion; shape KxM, where K is number of topics, M is vocabulary size
    :param vocab: vocabulary list/array of length K
    :param top_n: number of top values to print
    :param row_labels: format string for each row index where ``{i0}`` or ``{i1}`` are replaced by the respective
                       zero- or one-indexed topic numbers
    """
    print_ldamodel_distribution(topic_word_distrib, row_labels=row_labels, val_labels=vocab,
                                top_n=top_n)


def print_ldamodel_doc_topics(doc_topic_distrib, doc_labels, top_n=3, val_labels=DEFAULT_TOPIC_NAME_FMT):
    """
    Print `top_n` values from an LDA model's document-topic distribution `doc_topic_distrib`.

    .. seealso:: :func:`~tmtoolkit.topicmod.model_io.print_ldamodel_topic_words`
                 to print the top values of a topic-word distribution.

    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents, K is the
                              number of topics
    :param doc_labels: list/array of length N with a string label for each document
    :param top_n: number of top values to print
    :param val_labels: format string for each value where ``{i0}`` or ``{i1}`` are replaced by the respective
                       zero- or one-indexed topic numbers
    """
    print_ldamodel_distribution(doc_topic_distrib, row_labels=doc_labels, val_labels=val_labels,
                                top_n=top_n)


def save_ldamodel_summary_to_excel(excel_file, topic_word_distrib, doc_topic_distrib, doc_labels, vocab,
                                   top_n_topics=10, top_n_words=10, dtm=None,
                                   rank_label_fmt=None, topic_label_fmt=None):
    """
    Save a summary derived from an LDA model's topic-word and document-topic distributions (`topic_word_distrib` and
    `doc_topic_distrib` to an Excel file `excel_file`. Return the generated Excel sheets as dict of pandas DataFrames.

    The resulting Excel file will consist of 6 or optional 7 sheets:

    - ``top_doc_topics_vals``: document-topic distribution with probabilities of top topics per document
    - ``top_doc_topics_labels``: document-topic distribution with labels (e.g. ``"topic_12"``) of top topics per
      document
    - ``top_doc_topics_labelled_vals``: document-topic distribution combining probabilities and labels of top topics per
      document (e.g. ``"topic_12 (0.21)"``)
    - ``top_topic_word_vals``: topic-word distribution with probabilities of top words per topic
    - ``top_topic_word_labels``: topic-word distribution with top words per (e.g. ``"politics"``) topic
    - ``top_topic_words_labelled_vals``: topic-word distribution combining probabilities and top words per topic
      (e.g. ``"politics (0.08)"``)
    - optional if `dtm` is given – ``marginal_topic_distrib``: marginal topic distribution

    :param excel_file: target Excel file
    :param topic_word_distrib: topic-word distribtion; shape KxM, where K is number of topics, M is vocabulary size
    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents, K is the
                              number of topics
    :param doc_labels: list/array of length N with a string label for each document
    :param vocab: vocabulary list/array of length K
    :param top_n_topics: number of most probable topics per document to include in the summary
    :param top_n_words: number of most probable words per topic to include in the summary
    :param dtm: document-term matrix; shape NxM; if this is given, a sheet for the marginal topic distribution will
                be included
    :param rank_label_fmt: format string for the rank labels where ``{i0}`` or ``{i1}`` are replaced by the respective
                       zero- or one-indexed rank numbers (leave to None for default)
    :param topic_label_fmt: format string for the topic labels where ``{i0}`` or ``{i1}`` are replaced by the respective
                       zero- or one-indexed topic numbers (leave to None for default)
    :return: dict mapping sheet name to pandas DataFrame
    """

    # we'll use pandas instead of datatable here, because datatable doesn't support Excel export yet
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError('package `pandas` must be installed to use this function')

    rank_label_fmt = rank_label_fmt or DEFAULT_RANK_NAME_FMT
    topic_label_fmt = topic_label_fmt or DEFAULT_TOPIC_NAME_FMT
    sheets = OrderedDict()

    # must convert NumPy string array to lists of Python strings, because OpenPyXL can't handle them
    if isinstance(doc_labels, np.ndarray):
        doc_labels = list(map(str, doc_labels))

    if isinstance(vocab, np.ndarray):
        vocab = list(map(str, vocab))

    # doc-topic distribution sheets
    sheets['top_doc_topics_vals'] = top_n_from_distribution(doc_topic_distrib, top_n=top_n_topics,
                                                            row_labels=doc_labels,
                                                            col_labels=rank_label_fmt)
    sheets['top_doc_topics_labels'] = top_n_from_distribution(doc_topic_distrib, top_n=top_n_topics,
                                                              row_labels=doc_labels,
                                                              col_labels=rank_label_fmt,
                                                              val_labels=topic_label_fmt)
    sheets['top_doc_topics_labelled_vals'] = ldamodel_top_doc_topics(doc_topic_distrib, doc_labels, top_n=top_n_topics)

    # topic-word distribution sheets
    sheets['top_topic_word_vals'] = top_n_from_distribution(topic_word_distrib, top_n=top_n_words,
                                                            row_labels=topic_label_fmt,
                                                            col_labels=rank_label_fmt)
    sheets['top_topic_word_labels'] = top_n_from_distribution(topic_word_distrib, top_n=top_n_words,
                                                              row_labels=topic_label_fmt,
                                                              col_labels=rank_label_fmt,
                                                              val_labels=vocab)
    sheets['top_topic_words_labelled_vals'] = ldamodel_top_topic_words(topic_word_distrib, vocab, top_n=top_n_words)

    if dtm is not None:
        doc_len = doc_lengths(dtm)
        marg_topic_distr = get_marginal_topic_distrib(doc_topic_distrib, doc_len)
        row_names = [DEFAULT_TOPIC_NAME_FMT.format(i0=i, i1=i + 1) for i in range(len(marg_topic_distr))]
        sheets['marginal_topic_distrib'] = pd.DataFrame(marg_topic_distr, columns=['marginal_topic_distrib'],
                                                        index=row_names)

    excel_writer = pd.ExcelWriter(excel_file)

    for sh_name, sh_data in sheets.items():
        sh_data.to_excel(excel_writer, sh_name)

    excel_writer.save()

    return sheets


def save_ldamodel_to_pickle(picklefile, model, vocab, doc_labels, dtm=None, **kwargs):
    """
    Save an LDA model object `model` as pickle file to `picklefile`.

    .. seealso:: :func:`~tmtoolkit.topicmod.model_io.load_ldamodel_from_pickle` to load the saved model.

    :param picklefile: target file
    :param model: LDA model instance
    :param vocab: vocabulary list/array of length M
    :param doc_labels: document labels list/array of length N
    :param dtm: optional document-term matrix of shape NxM
    :param kwargs: additional options for :func:`tmtoolkit.utils.pickle_data`
    """
    pickle_data({'model': model, 'vocab': vocab, 'doc_labels': doc_labels, 'dtm': dtm}, picklefile, **kwargs)


def load_ldamodel_from_pickle(picklefile, **kwargs):
    """
    Load an LDA model object from a pickle file `picklefile`.

    .. seealso:: :func:`~tmtoolkit.topicmod.model_io.save_ldamodel_to_pickle` to save a model.

    :param picklefile: target file
    :param kwargs: additional options for :func:`tmtoolkit.utils.unpickle_file`
    :return: dict with keys: ``'model'`` – model instance; ``'vocab'`` – vocabulary; ``'doc_labels'`` – document labels;
                             ``'dtm'`` – optional document-term matrix;
    """
    return unpickle_file(picklefile, **kwargs)
