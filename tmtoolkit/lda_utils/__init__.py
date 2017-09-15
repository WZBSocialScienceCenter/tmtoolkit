# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from . import evaluation
from ..utils import pickle_data, unpickle_file


def top_n_from_distribution(distrib, top_n=10, row_labels=None, val_labels=None):
    """
    Get `top_n` values from LDA model's distribution `distrib` as DataFrame. Can be used for topic-word distributions
    and document-topic distributions. Set `row_labels` to a prefix (pass a string) or a list. Set `val_labels` to
    return value labels instead of pure values (probabilities).
    """
    if row_labels is None:
        row_label_fixed = 'row'
    elif type(row_labels) is str:
        row_label_fixed = row_labels
    else:
        row_label_fixed = None

    df = pd.DataFrame(columns=range(top_n))

    for i, row_distrib in enumerate(distrib):
        if row_label_fixed:
            row_name = '%s %d' % (row_label_fixed, i+1)
        else:
            row_name = row_labels[i]

        # `sorter_arr` is an array of indices that would sort another array by `row_distrib` (from low to high!)
        sorter_arr = np.argsort(row_distrib)

        if val_labels is None:
            sorted_vals = row_distrib[sorter_arr][:-(top_n + 1):-1]
        else:
            if type(val_labels) is str:
                sorted_vals = ['%s %d' % (val_labels, num+1) for num in sorter_arr[::-1]][:top_n]
            else:
                # first brackets: sort vocab by `sorter_arr`
                # second brackets: slice operation that reverts ordering (:-1) and then selects only `n_top` number of
                # elements
                sorted_vals = val_labels[sorter_arr][:-(top_n + 1):-1]

        top_labels_series = pd.Series(sorted_vals, name=row_name)

        df = df.append(top_labels_series)

    return df


def _join_value_and_label_dfs(vals, labels, row_labels=None):
    df = pd.DataFrame()
    for i, (_, row) in enumerate(labels.iterrows()):
        joined = []
        for j, lbl in enumerate(row):
            val = vals.iloc[i, j]
            joined.append('%s (%f)' % (lbl, val))

        if row_labels is not None:
            if type(row_labels) is str:
                row_name = '%s %d' % (row_labels, i)
            else:
                row_name = row_labels[i]
        else:
            row_name = None

        df = df.append(pd.Series(joined, name=row_name))

    df.columns = pd.Series(range(1, df.shape[1]+1), name='rank')

    return df


def ldamodel_top_topic_words(topic_word_distrib, vocab, n_top=10):
    df_values = top_n_from_distribution(topic_word_distrib, top_n=n_top, row_labels='topic', val_labels=None)
    df_labels = top_n_from_distribution(topic_word_distrib, top_n=n_top, row_labels='topic', val_labels=vocab)
    return _join_value_and_label_dfs(df_values, df_labels, row_labels='topic')


def ldamodel_top_doc_topics(doc_topic_distrib, doc_labels, n_top=3):
    df_values = top_n_from_distribution(doc_topic_distrib, top_n=n_top, row_labels=doc_labels, val_labels=None)
    df_labels = top_n_from_distribution(doc_topic_distrib, top_n=n_top, row_labels=doc_labels, val_labels='topic')
    return _join_value_and_label_dfs(df_values, df_labels, row_labels=doc_labels)


def ldamodel_full_topic_words(topic_word_distrib, vocab, fmt_rownames='topic %d'):
    if fmt_rownames:
        rownames = [fmt_rownames % num for num in range(topic_word_distrib.shape[0])]
    else:
        rownames = None

    return pd.DataFrame(topic_word_distrib, columns=vocab, index=rownames)


def ldamodel_full_doc_topics(doc_topic_distrib, doc_labels, fmt_colnames='topic %d'):
    if fmt_colnames:
        colnames = [fmt_colnames % num for num in range(doc_topic_distrib.shape[0])]
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
    print_ldamodel_distribution(topic_word_distrib, row_labels='topic', val_labels=vocab, top_n=n_top)


def print_ldamodel_doc_topics(doc_topic_distrib, doc_labels, n_top=3):
    """Print `n_top` values from a LDA model's document-topic distributions."""
    print_ldamodel_distribution(doc_topic_distrib, row_labels=doc_labels, val_labels='topic', top_n=n_top)


def save_ldamodel_to_pickle(model, vocab, doc_labels, picklefile):
    """Save a LDA model as pickle file."""
    pickle_data({'model': model, 'vocab': vocab, 'doc_labels': doc_labels}, picklefile)


def load_ldamodel_from_pickle(picklefile):
    """Load a LDA model from a pickle file."""
    data = unpickle_file(picklefile)
    return data['model'], data['vocab'], data['doc_labels']
