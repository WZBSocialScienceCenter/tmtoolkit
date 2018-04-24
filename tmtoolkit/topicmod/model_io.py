# -*- coding: utf-8 -*-
"""
Functions for printing/exporting topic models.

Markus Konrad <markus.konrad@wzb.eu>
"""

from __future__ import unicode_literals
from collections import OrderedDict

import pandas as pd

from tmtoolkit.topicmod.model_stats import get_doc_lengths, get_marginal_topic_distrib, top_n_from_distribution, \
    _join_value_and_label_dfs
from tmtoolkit.utils import pickle_data, unpickle_file
from tmtoolkit.topicmod._common import DEFAULT_RANK_NAME_FMT, DEFAULT_TOPIC_NAME_FMT


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


def print_ldamodel_topic_words(topic_word_distrib, vocab, n_top=10, row_labels=DEFAULT_TOPIC_NAME_FMT):
    """Print `n_top` values from a LDA model's topic-word distributions."""
    print_ldamodel_distribution(topic_word_distrib, row_labels=row_labels, val_labels=vocab,
                                top_n=n_top)


def print_ldamodel_doc_topics(doc_topic_distrib, doc_labels, n_top=3, val_labels=DEFAULT_TOPIC_NAME_FMT):
    """Print `n_top` values from a LDA model's document-topic distributions."""
    print_ldamodel_distribution(doc_topic_distrib, row_labels=doc_labels, val_labels=val_labels,
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