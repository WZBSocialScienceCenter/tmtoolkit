"""
Common statistics from BoW matrices.

Markus Konrad <markus.konrad@wzb.eu>
"""

import itertools

import numpy as np
from scipy.sparse import issparse


def get_doc_lengths(dtm):
    if isinstance(dtm, np.matrix):
        dtm = dtm.A
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    res = np.sum(dtm, axis=1)
    if res.ndim != 1:
        return res.A.flatten()
    else:
        return res


def get_doc_frequencies(dtm, min_val=1, proportions=False):
    """
    For each word in the vocab of `dtm` (i.e. its columns), return how often it occurs at least `min_val` times.
    If `proportions` is True, return proportions scaled to the number of documents instead of absolute numbers.
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    doc_freq = np.sum(dtm >= min_val, axis=0)

    if doc_freq.ndim != 1:
        doc_freq = doc_freq.A.flatten()

    if proportions:
        return doc_freq / dtm.shape[0]
    else:
        return doc_freq


def get_codoc_frequencies(dtm, min_val=1, proportions=False):
    """
    For each unique pair of words `w1, w2` in the vocab of `dtm` (i.e. its columns), return how often both occur
    together at least `min_val` times. If `proportions` is True, return proportions scaled to the number of documents
    instead of absolute numbers.
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    n_docs, n_vocab = dtm.shape
    if n_vocab < 2:
        raise ValueError('`dtm` must have at least two columns (i.e. 2 unique words)')

    word_in_doc = dtm >= min_val

    codoc_freq = {}
    for w1, w2 in itertools.combinations(range(n_vocab), 2):
        if issparse(dtm):
            w1_in_docs = word_in_doc[:, w1].A.flatten()
            w2_in_docs = word_in_doc[:, w2].A.flatten()
        else:
            w1_in_docs = word_in_doc[:, w1]
            w2_in_docs = word_in_doc[:, w2]

        freq = np.sum(w1_in_docs & w2_in_docs)
        if proportions:
            freq /= n_docs
        codoc_freq[(w1, w2)] = freq

    return codoc_freq


def get_term_frequencies(dtm):
    if isinstance(dtm, np.matrix):
        dtm = dtm.A
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    res = np.sum(dtm, axis=0)
    if res.ndim != 1:
        return res.A.flatten()
    else:
        return res


def get_term_proportions(dtm):
    """
    Return the term proportions given the document-term matrix `dtm`
    """
    unnorm = get_term_frequencies(dtm)

    if unnorm.sum() == 0:
        raise ValueError('`dtm` does not contain any tokens (is all-zero)')
    else:
        return unnorm / unnorm.sum()
