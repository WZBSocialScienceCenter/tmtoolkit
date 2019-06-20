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
    Return the length, i.e. number of terms for each document in document-term-matrix `dtm`.
    This corresponds to the row-wise sums in `dtm`.
    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw terms counts.
    :return: NumPy array of size N (number of docs) with integers indicating the number of terms per document.
    """
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
    For each term in the vocab of `dtm` (i.e. its columns), return how often it occurs at least `min_val` times per
    document.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param min_val: threshold for counting occurrences
    :param proportions: If `proportions` is True, return proportions scaled to the number of documents instead of
                        absolute numbers.
    :return: NumPy array of size M (vocab size) indicating how often each term occurs at least `min_val` times.
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
    Return the number of occurrences of each term in the vocab across all documents in document-term-matrix `dtm`.
    This corresponds to the column-wise sums in `dtm`.
    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param proportions: If `proportions` is True, return proportions scaled to the number of terms in the whole `dtm`.
    :return: NumPy array of size M (vocab size) with integers indicating the number of occurrences of each term in the
             vocab across all documents.
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    unnorm = np.sum(dtm, axis=0)
    if unnorm.ndim != 1:
        unnorm = unnorm.A.flatten()

    if proportions:
        n = unnorm.sum()
        if n == 0:
            raise ValueError('`dtm` does not contain any terms (is all-zero)')
        else:
            return unnorm / n
    else:
        return unnorm

@deprecated(deprecated_in='0.9.0', removed_in='0.10.0',
            details='Please use `term_frequencies()` with parameter `proportions=True` instead.')
def get_term_proportions(dtm):
    return term_frequencies(dtm, proportions=True)


def tf_binary(dtm):
    """
    Transform raw count document-term-matrix `dtm` to binary term frequency matrix. This matrix contains 1 whereever
    a term occurred in a document, else 0.
    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :return: (sparse) binary term frequency matrix of type integer of size NxM
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    return (dtm > 0).astype(int)


def tf_proportions(dtm):
    """
    Transform raw count document-term-matrix `dtm` to term frequency matrix with proportions, i.e. term counts
    normalized by document length.
    Note that this may introduce NaN values due to division by zero when a document is of length 0.
    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :return: (sparse) term frequency matrix of size NxM with proportions, i.e. term counts normalized by document length
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    norm_factor = 1 / doc_lengths(dtm)[:, None]   # shape: Nx1

    if issparse(dtm):
        res = dtm.multiply(norm_factor)
    else:
        res = dtm * norm_factor

    if isinstance(res, np.matrix):
        return res.A
    else:
        return res


def tf_log(dtm, log_fn=np.log1p):
    """
    Transform raw count document-term-matrix `dtm` to log-normalized term frequency matrix `log_fn(1 + dtm)`.
    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param log_fn: log function to use. default is NumPy's `log1p`, which calculates `log(1 + x)`.
    :return: (sparse) log-normalized term frequency matrix of size NxM
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    if log_fn is np.log1p:
        if issparse(dtm):
            return dtm.log1p()
        else:
            return log_fn(dtm)
    else:
        if issparse(dtm):
            dtm = dtm.toarray()

        return log_fn(dtm)


def tf_double_norm(dtm, K=0.5):
    """
    Transform raw count document-term-matrix `dtm` to double-normalized term frequency matrix
    `K + (1-K) * dtm / max{t in doc}`, where `max{t in doc}` is vector of size N containing the maximum term count per
    document.
    Note that this may introduce NaN values due to division by zero when a document is of length 0.
    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param K: normalization factor
    :return: double-normalized term frequency matrix of size NxM
    """
    if dtm.ndim != 2 or 0 in dtm.shape:
        raise ValueError('`dtm` must be a non-empty 2D array/matrix')

    if issparse(dtm):
        dtm = dtm.toarray()

    max_per_doc = np.max(dtm, axis=1)

    return K + (1 - K) * dtm / max_per_doc[:, None]


def idf(dtm, smooth_log=1, smooth_df=1):
    """
    Calculate inverse document frequency (idf) vector from raw count document-term-matrix `dtm` with formula
    `log(smooth_log + N / (smooth_df + df))`, where `N` is the number of documents, `df` is the document frequency
    (see function `doc_frequencies()`) and `smooth_*` are smoothing constants. With default arguments, the formula
    is thus `log(1 + N/(1+df))`.

    Note that this may introduce NaN values due to division by zero when a document is of length 0.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param smooth_log: smoothing constant inside log()
    :param smooth_df: smoothing constant to add to document frequency
    :return: NumPy array of size M (vocab size) with inverse document frequency for each term in the vocab
    """
    if dtm.ndim != 2 or 0 in dtm.shape:
        raise ValueError('`dtm` must be a non-empty 2D array/matrix')

    n_docs = dtm.shape[0]
    df = doc_frequencies(dtm)
    x = n_docs / (smooth_df + df)

    if smooth_log == 1:      # log1p is faster than the equivalent log(1 + x)
        return np.log1p(x)
    else:
        return np.log(smooth_log + x)


def idf_probabilistic(dtm, smooth=1):
    """
    Calculate probabilistic inverse document frequency (idf) vector from raw count document-term-matrix `dtm` with
    formula `log(smooth + (N - df) / df)`, where `N` is the number of documents and `df` is the document frequency (see
    function `doc_frequencies()`).

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param smooth: smoothing constant (setting this to 0 can lead to -inf results)
    :return: NumPy array of size M (vocab size) with probabilistic inverse document frequency for each term in the vocab
    """
    if dtm.ndim != 2 or 0 in dtm.shape:
        raise ValueError('`dtm` must be a non-empty 2D array/matrix')

    n_docs = dtm.shape[0]
    df = doc_frequencies(dtm)
    x = (n_docs - df) / df

    if smooth == 1:      # log1p is faster than the equivalent log(1 + x)
        return np.log1p(x)
    else:
        return np.log(smooth + x)


def tfidf(dtm, tf_func=tf_proportions, idf_func=idf, **kwargs):
    """
    Calculate tfidf (term frequency inverse document frequency) matrix from raw count document-term-matrix `dtm` with
    matrix multiplication `tf * diag(idf)`, where `tf` is the term frequency matrix `tf_funct(dtm)` and `idf` is the
    document frequency vector `idf_func(dtm)`.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param tf_func: function to calculate term-frequency matrix. see `tf_*` functions in this module.
    :param idf_func: function to calculate inverse document frequency vector. see `tf_*` functions in this module.
    :param kwargs: additional parameters passed to `tf_func` or `idf_func` like `K` or `smooth` (depending on which
                   parameters these functions except)
    :return: (sparse) tfidf matrix of size NxM
    """
    if dtm.ndim != 2 or 0 in dtm.shape:
        raise ValueError('`dtm` must be a non-empty 2D array/matrix')

    if idf_func is idf:
        idf_opts = {}
        if 'smooth_log' in kwargs:
            idf_opts['smooth_log'] = kwargs.pop('smooth_log')
        if 'smooth_df' in kwargs:
            idf_opts['smooth_df'] = kwargs.pop('smooth_df')

        idf_vec = idf_func(dtm, **idf_opts)
    elif idf_func is idf_probabilistic and 'smooth' in kwargs:
        idf_vec = idf_func(dtm, smooth=kwargs.pop('smooth'))
    else:
        idf_vec = idf_func(dtm)

    tf_mat = tf_func(dtm, **kwargs)

    # formally, it would be a matrix multiplication: tf * diag(idf), i.e. np.matmul(tf_mat, np.diag(idf_vec)),
    # so that each column i in tf in multiplied by the respective idf value: tf[:, i] * idf[i]
    # but diag(df) would create a large intermediate matrix, so let's use NumPy broadcasting:
    if issparse(tf_mat):
        return tf_mat.multiply(idf_vec)
    else:
        return tf_mat * idf_vec
