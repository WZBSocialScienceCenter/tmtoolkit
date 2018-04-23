# -*- coding: utf-8 -*-
"""
Functions for printing/exporting topic models.

Markus Konrad <markus.konrad@wzb.eu>
"""

from __future__ import unicode_literals
from collections import OrderedDict

import six
import numpy as np
import pandas as pd

from tmtoolkit.topicmod.model_stats import get_doc_lengths, get_marginal_topic_distrib
from tmtoolkit.utils import pickle_data, unpickle_file


DEFAULT_TOPIC_NAME_FMT = 'topic_{i1}'
DEFAULT_RANK_NAME_FMT = 'rank_{i1}'


def top_n_from_distribution(distrib, top_n=10, row_labels=None, col_labels=None, val_labels=None):
    """
    Get `top_n` values from LDA model's distribution `distrib` as DataFrame. Can be used for topic-word distributions
    and document-topic distributions. Set `row_labels` to a format string or a list. Set `col_labels` to a format
    string for the column names. Set `val_labels` to return value labels instead of pure values (probabilities).
    """
    if len(distrib) == 0:
        raise ValueError('`distrib` must contain values')

    if top_n < 1:
        raise ValueError('`top_n` must be at least 1')
    elif top_n > distrib.shape[1]:
        raise ValueError('`top_n` cannot be larger than num. of values in `distrib` rows')

    if row_labels is None:
        row_label_fixed = None
    elif isinstance(row_labels, six.string_types):
        row_label_fixed = row_labels
    else:
        row_label_fixed = None

    if val_labels is not None and type(val_labels) in (list, tuple):
        val_labels = np.array(val_labels)

    if col_labels is None:
        columns = range(top_n)
    else:
        columns = [col_labels.format(i0=i, i1=i+1) for i in range(top_n)]

    series = []

    for i, row_distrib in enumerate(distrib):
        if row_label_fixed:
            row_name = row_label_fixed.format(i0=i, i1=i+1)
        else:
            if row_labels is not None:
                row_name = row_labels[i]
            else:
                row_name = None

        # `sorter_arr` is an array of indices that would sort another array by `row_distrib` (from low to high!)
        sorter_arr = np.argsort(row_distrib)

        if val_labels is None:
            sorted_vals = row_distrib[sorter_arr][:-(top_n + 1):-1]
        else:
            if isinstance(val_labels, six.string_types):
                sorted_vals = [val_labels.format(i0=i, i1=i+1, val=row_distrib[i]) for i in sorter_arr[::-1]][:top_n]
            else:
                # first brackets: sort vocab by `sorter_arr`
                # second brackets: slice operation that reverts ordering (:-1) and then selects only `n_top` number of
                # elements
                sorted_vals = val_labels[sorter_arr][:-(top_n + 1):-1]

        series_kwargs = dict(index=columns)
        if row_name is not None:
            series_kwargs['name'] = row_name

        series.append(pd.Series(sorted_vals, **series_kwargs))

    return pd.DataFrame(series)


def top_words_for_topics(topic_word_distrib, top_n, vocab=None):
    if not isinstance(topic_word_distrib, np.ndarray) or topic_word_distrib.ndim != 2:
        raise ValueError('`topic_word_distrib` must be a 2D NumPy array')

    if len(topic_word_distrib) == 0:
        raise ValueError('`topic_word_distrib` cannot be empty')

    if vocab is not None:
        if not isinstance(vocab, np.ndarray) or vocab.ndim != 1:
            raise ValueError('`vocab` must be a 1D NumPy array')

        if len(vocab) == 0:
            raise ValueError('`vocab` cannot be empty')

        if topic_word_distrib.shape[1] != len(vocab):
            raise ValueError('shapes of provided `topic_word_distrib` and `vocab` do not match (vocab sizes differ)')

    if top_n < 1:
        raise ValueError('`top_n` must be at least 1')
    elif top_n > topic_word_distrib.shape[1]:
        raise ValueError('`top_n` cannot be larger than vocab size')

    topic_words = []

    for topic in topic_word_distrib:
        sorter_arr = np.argsort(topic)
        if vocab is None:
            topic_words.append(sorter_arr[:-(top_n+1):-1])
        else:
            topic_words.append(vocab[sorter_arr][:-(top_n+1):-1])

    return topic_words


def _join_value_and_label_dfs(vals, labels, top_n, val_fmt=None, row_labels=None, col_labels=None, index_name=None):
    val_fmt = val_fmt or '{lbl} ({val:.4})'
    col_labels = col_labels or DEFAULT_RANK_NAME_FMT
    index_name = index_name or 'document'

    if col_labels is None:
        columns = range(top_n)
    else:
        columns = [col_labels.format(i0=i, i1=i+1) for i in range(top_n)]

    df = pd.DataFrame(columns=columns)

    for i, (_, row) in enumerate(labels.iterrows()):
        joined = []
        for j, lbl in enumerate(row):
            val = vals.iloc[i, j]
            joined.append(val_fmt.format(lbl=lbl, val=val))

        if row_labels is not None:
            if isinstance(row_labels, six.string_types):
                row_name = row_labels.format(i0=i, i1=i+1)
            else:
                row_name = row_labels[i]
        else:
            row_name = None

        row_data = pd.Series(joined, name=row_name, index=columns)
        df = df.append(row_data)

    df.index.name = index_name

    return df


def ldamodel_top_topic_words(topic_word_distrib, vocab, top_n=10, val_fmt=None, col_labels=None, index_name=None):
    df_values = top_n_from_distribution(topic_word_distrib, top_n=top_n,
                                        row_labels=DEFAULT_TOPIC_NAME_FMT, val_labels=None)
    df_labels = top_n_from_distribution(topic_word_distrib, top_n=top_n,
                                        row_labels=DEFAULT_TOPIC_NAME_FMT, val_labels=vocab)
    return _join_value_and_label_dfs(df_values, df_labels, top_n, row_labels=DEFAULT_TOPIC_NAME_FMT,
                                     val_fmt=val_fmt, col_labels=col_labels, index_name=index_name)


def ldamodel_top_doc_topics(doc_topic_distrib, doc_labels, top_n=3, val_fmt=None, col_labels=None, index_name=None):
    df_values = top_n_from_distribution(doc_topic_distrib, top_n=top_n,
                                        row_labels=doc_labels, val_labels=None)
    df_labels = top_n_from_distribution(doc_topic_distrib, top_n=top_n,
                                        row_labels=doc_labels, val_labels=DEFAULT_TOPIC_NAME_FMT)
    return _join_value_and_label_dfs(df_values, df_labels, top_n, row_labels=doc_labels,
                                     val_fmt=val_fmt, col_labels=col_labels, index_name=index_name)


def ldamodel_full_topic_words(topic_word_distrib, vocab, fmt_rownames=DEFAULT_TOPIC_NAME_FMT):
    if fmt_rownames:
        rownames = [fmt_rownames.format(i0=i, i1=i+1) for i in range(topic_word_distrib.shape[0])]
    else:
        rownames = None

    return pd.DataFrame(topic_word_distrib, columns=vocab, index=rownames)


def ldamodel_full_doc_topics(doc_topic_distrib, doc_labels, fmt_colnames=DEFAULT_TOPIC_NAME_FMT):
    if fmt_colnames:
        colnames = [fmt_colnames.format(i0=i, i1=i+1) for i in range(doc_topic_distrib.shape[1])]
    else:
        colnames = None

    return pd.DataFrame(doc_topic_distrib, columns=colnames, index=doc_labels)


def print_ldamodel_distribution(distrib, row_labels, val_labels, top_n=10):
    """
    Print `n_top` top values from a LDA model's distribution `distrib`. Can be used for topic-word distributions and
    document-topic distributions.
    """

    df_values = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=None)
    df_labels = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=val_labels)

    for i, (ind, row) in enumerate(df_labels.iterrows()):
        print(ind)
        for j, label in enumerate(row):
            val = df_values.iloc[i, j]
            print('> #%d. %s (%f)' % (j + 1, label, val))


def print_ldamodel_topic_words(topic_word_distrib, vocab, n_top=10):
    """Print `n_top` values from a LDA model's topic-word distributions."""
    print_ldamodel_distribution(topic_word_distrib, row_labels=DEFAULT_TOPIC_NAME_FMT, val_labels=vocab,
                                top_n=n_top)


def print_ldamodel_doc_topics(doc_topic_distrib, doc_labels, n_top=3):
    """Print `n_top` values from a LDA model's document-topic distributions."""
    print_ldamodel_distribution(doc_topic_distrib, row_labels=doc_labels, val_labels=DEFAULT_TOPIC_NAME_FMT,
                                top_n=n_top)


def save_ldamodel_summary_to_excel(excel_file, topic_word_distrib, doc_topic_distrib, doc_labels, vocab,
                                   top_n_topics=10, top_n_words=10, dtm=None,
                                   rank_label_fmt=None, topic_label_fmt=None):
    rank_label_fmt = rank_label_fmt or DEFAULT_RANK_NAME_FMT
    topic_label_fmt = topic_label_fmt or DEFAULT_TOPIC_NAME_FMT
    excel_writer = pd.ExcelWriter(excel_file)
    sheets = OrderedDict()

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
        doc_lengths = get_doc_lengths(dtm)
        marg_topic_distr = get_marginal_topic_distrib(doc_topic_distrib, doc_lengths)
        row_names = [DEFAULT_TOPIC_NAME_FMT.format(i0=i, i1=i + 1) for i in range(len(marg_topic_distr))]
        sheets['marginal_topic_distrib'] = pd.DataFrame(marg_topic_distr, columns=['marginal_topic_distrib'],
                                                        index=row_names)

    for sh_name, sh_data in sheets.items():
        sh_data.to_excel(excel_writer, sh_name)

    excel_writer.save()

    return sheets


def save_ldamodel_to_pickle(picklefile, model, vocab, doc_labels, dtm=None, **kwargs):
    """Save a LDA model as pickle file."""
    pickle_data({'model': model, 'vocab': vocab, 'doc_labels': doc_labels, 'dtm': dtm}, picklefile)


def load_ldamodel_from_pickle(picklefile, **kwargs):
    """Load a LDA model from a pickle file."""
    return unpickle_file(picklefile, **kwargs)