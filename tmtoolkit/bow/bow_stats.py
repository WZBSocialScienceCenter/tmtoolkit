"""
Common statistics from BoW matrices.

Markus Konrad <markus.konrad@wzb.eu>
"""

import itertools

import numpy as np
from scipy.sparse import issparse
from deprecation import deprecated


@deprecated(deprecated_in='0.9.0', removed_in='0.10.0',
            details='This function was renamed to `doc_lengths`.')
def get_doc_lengths(dtm):
    return doc_lengths(dtm)


def doc_lengths(dtm):
    """
    Return the length, i.e. number of tokens for each document in document-term-matrix `dtm`.
    This corresponds to the row-wise sums in `dtm`.
    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw token counts.
    :return: NumPy array of size N (number of docs) with integers indicating the number of tokens per document.
    """
    if not isinstance(dtm, np.ndarray) and hasattr(dtm, 'A'):
        dtm = dtm.A
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    res = np.sum(dtm, axis=1)
    if res.ndim != 1:
        return res.A.flatten()
    else:
        return res


@deprecated(deprecated_in='0.9.0', removed_in='0.10.0',
            details='This function was renamed to `doc_frequencies`.')
def get_doc_frequencies(dtm, min_val=1, proportions=False):
    return doc_frequencies(dtm, min_val=min_val, proportions=proportions)


def doc_frequencies(dtm, min_val=1, proportions=False):
    """
    For each token in the vocab of `dtm` (i.e. its columns), return how often it occurs at least `min_val` times.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw token counts.
    :param min_val: threshold for counting occurrences
    :param proportions: If `proportions` is True, return proportions scaled to the number of documents instead of
                        absolute numbers.
    :return: NumPy array of size M (vocab size) indicating how often each token occurs at least `min_val` times.
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


@deprecated(deprecated_in='0.9.0', removed_in='0.10.0',
            details='This function was renamed to `codoc_frequencies`.')
def get_codoc_frequencies(dtm, min_val=1, proportions=False):
    return codoc_frequencies(dtm, min_val=min_val, proportions=proportions)


def codoc_frequencies(dtm, min_val=1, proportions=False):
    """
    For each unique pair of words `w1, w2` in the vocab of `dtm` (i.e. its columns), return how often both occur
    together at least `min_val` times. If `proportions` is True, return proportions scaled to the number of documents
    instead of absolute numbers.

    :param min_val: threshold for counting occurrences
    :param proportions: If `proportions` is True, return proportions scaled to the number of documents instead of
                        absolute numbers.
    :return: NumPy array of size M (vocab size) indicating how often each unique pair of words `w1, w2` occurs at
             least `min_val` times.
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


@deprecated(deprecated_in='0.9.0', removed_in='0.10.0',
            details='This function was renamed to `term_frequencies`.')
def get_term_frequencies(dtm):
    return term_frequencies(dtm)


def term_frequencies(dtm, proportions=False):
    """
    Return the number of occurrences of each token in the vocab across all documents in document-term-matrix `dtm`.
    This corresponds to the column-wise sums in `dtm`.
    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw token counts.
    :param proportions: If `proportions` is True, return proportions scaled to the number of tokens in the whole `dtm`.
    :return: NumPy array of size M (vocab size) with integers indicating the number of occurrences of each token in the
             vocab across all documents.
    """
    if not isinstance(dtm, np.ndarray) and hasattr(dtm, 'A'):
        dtm = dtm.A
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    unnorm = np.sum(dtm, axis=0)
    if unnorm.ndim != 1:
        unnorm = unnorm.A.flatten()

    if proportions:
        n = unnorm.sum()
        if n == 0:
            raise ValueError('`dtm` does not contain any tokens (is all-zero)')
        else:
            return unnorm / n
    else:
        return unnorm

@deprecated(deprecated_in='0.9.0', removed_in='0.10.0',
            details='Please use `term_frequencies()` with parameter `proportions=True` instead.')
def get_term_proportions(dtm):
    return term_frequencies(dtm, proportions=True)


def tf_binary(dtm):
    pass


def tf_frequency(dtm):
    pass


def tf_log(dtm, add=1, log_fn=np.log):
    pass


def tf_double_norm(dtm, K=0.5):
    pass


def tfidf(tf):
    pass
