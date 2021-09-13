"""
Common statistics from bag-of-words (BoW) matrices.
"""

import numpy as np
from scipy.sparse import issparse

import pandas as pd

def doc_lengths(dtm):
    """
    Return the length, i.e. number of terms for each document in document-term-matrix `dtm`.
    This corresponds to the row-wise sums in `dtm`.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw terms counts
    :return: NumPy array of size N (number of docs) with integers indicating the number of terms per document
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    res = np.sum(dtm, axis=1)
    if res.ndim != 1:
        return res.A.flatten()
    else:
        return res


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


def word_cooccurrence(dtm, min_val=1, proportions=False):
    """
    Calculate the co-document frequency (aka word co-occurrence) matrix. Alias for
    :func:`~tmtoolkit.bow.bow_stats.codoc_frequencies`.

    .. seealso:: :func:`~tmtoolkit.bow.bow_stats.codoc_frequencies`
    """
    return codoc_frequencies(dtm, min_val=min_val, proportions=proportions)


def codoc_frequencies(dtm, min_val=1, proportions=False):
    """
    Calculate the co-document frequency (aka word co-occurrence) matrix for a document-term matrix `dtm`, i.e. how often
    each pair of tokens occurs together at least `min_val` times in the same document. If `proportions` is True,
    return proportions scaled to the number of documents instead of absolute numbers.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param min_val: threshold for counting occurrences
    :param proportions: If `proportions` is True, return proportions scaled to the number of documents instead of
                        absolute numbers.
    :return: co-document frequency (aka word co-occurrence) matrix with shape (vocab size, vocab size)
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    if dtm.shape[1] < 2:
        raise ValueError('`dtm` must have at least two columns')

    if issparse(dtm) and dtm.format != 'csc':
        dtm = dtm.tocsc()

    bin_dtm = (dtm >= min_val).astype(int)

    cooc = bin_dtm.T @ bin_dtm

    if proportions:
        return cooc / dtm.shape[0]
    else:
        return cooc


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


def tf_binary(dtm):
    """
    Transform raw count document-term-matrix `dtm` to binary term frequency matrix. This matrix contains 1 whenever
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

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts
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
    Transform raw count document-term-matrix `dtm` to log-normalized term frequency matrix ``log_fn(dtm)``.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param log_fn: log function to use; default is NumPy's :func:`numpy.log1p`, which calculates ``log(1 + x)``
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
    ``K + (1-K) * dtm / max{t in doc}``, where ``max{t in doc}`` is vector of size N containing the maximum term count
    per document.

    Note that this may introduce NaN values due to division by zero when a document is of length 0.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts
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
    ``log(smooth_log + N / (smooth_df + df))``, where ``N`` is the number of documents, ``df`` is the document frequency
    (see function :func:`~tmtoolkit.bow.bow_stats.doc_frequencies`), `smooth_log` and `smooth_df` are smoothing
    constants. With default arguments, the formula is thus ``log(1 + N/(1+df))``.

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
    formula ``log(smooth + (N - df) / df)``, where ``N`` is the number of documents and ``df`` is the document
    frequency (see function :func:`~tmtoolkit.bow.bow_stats.doc_frequencies`).

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
    matrix multiplication ``tf * diag(idf)``, where `tf` is the term frequency matrix ``tf_func(dtm)`` and ``idf`` is
    the document frequency vector ``idf_func(dtm)``.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts
    :param tf_func: function to calculate term-frequency matrix; see ``tf_*`` functions in this module
    :param idf_func: function to calculate inverse document frequency vector; see ``tf_*`` functions in this module
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


def sorted_terms(mat, vocab, lo_thresh=0, hi_tresh=None, top_n=None, ascending=False, table_doc_labels=None):
    """
    For each row (i.e. document) in a (sparse) document-term-matrix `mat`, do the following:

    1. filter all values according to `lo_thresh` and `hi_thresh`
    2. sort values and the corresponding terms from `vocab` according to `ascending`
    3. optionally select the top `top_n` terms
    4. generate a list with pairs of terms and values

    Return the collected lists for each row or convert the result to a data frame if document labels are passed via
    `data_frame_doc_labels` (see shortcut function :func:`~tmtoolkit.bow.bow_stats.sorted_terms_table`).

    :param mat: (sparse) document-term-matrix `mat` (may be tf-idf transformed or any other transformation)
    :param vocab: list or array of vocabulary corresponding to columns in `mat`
    :param lo_thresh: if not None, filter for values greater than `lo_thresh`
    :param hi_tresh: if not None, filter for values lesser than or equal `hi_thresh`
    :param top_n: if not None, select only the top `top_n` terms
    :param ascending: sorting direction
    :param table_doc_labels: optional list/array of document labels corresponding to `mat` rows
    :return: list of list with tuples (term, value) or data table with columns "doc", "term", "value"
             if `data_frame_doc_labels` is given
    """
    n_vocab = len(vocab)

    if mat.shape[1] != n_vocab:
        raise ValueError('number of columns in `mat` does not match size of `vocab`')

    if lo_thresh is not None and hi_tresh is not None and lo_thresh > hi_tresh:
        raise ValueError('`lo_thresh` must be less than or equal `hi_thresh`')

    if top_n is not None and top_n < 1:
        raise ValueError('`top_n` must be at least 1')

    if table_doc_labels is not None and len(table_doc_labels) != mat.shape[0]:
        raise ValueError('length of `data_frame_doc_labels` must match number of rows in `mat`')

    if not isinstance(vocab, np.ndarray):
        vocab = np.array(vocab)

    if issparse(mat) and mat.format != 'csr':
        mat = mat.tocsr()

    if isinstance(mat, np.matrix):
        mat = mat.A

    res = []
    for i in range(mat.shape[0]):  # iterate through matrix rows
        row = mat[i, :]

        # create mask to filter all values in the row according to `lo_thresh` and `hi_thresh`
        row_mask = np.ones((n_vocab,), dtype=bool)
        if lo_thresh is not None:
            row_mask_tmp = row > lo_thresh
            row_mask &= row_mask_tmp.A[0] if issparse(mat) else row_mask_tmp
        if hi_tresh is not None:
            row_mask_tmp = row > hi_tresh   # using inverse of > here instead of <= because of sparse matrix
            row_mask &= ~(row_mask_tmp.A[0] if issparse(mat) else row_mask_tmp)

        # indices of values that we pick from this row
        mask_ind = np.where(row_mask)[0]

        # pick the values
        if issparse(row):
            mask_vals = row[:, mask_ind].A[0]
        else:
            mask_vals = row[mask_ind]

        # create indices that sort the selected values
        sorted_ind = np.argsort(mask_vals)
        if not ascending:
            sorted_ind = sorted_ind[::-1]

        # get the terms and values from the row in sorted order
        sorted_vocab_ind = mask_ind[sorted_ind]  # indices into vocab
        row_terms = vocab[sorted_vocab_ind]
        row_vals = mask_vals[sorted_ind]

        # optionally select the top `top_n` terms
        if top_n is not None:
            row_terms = row_terms[:top_n]
            row_vals = row_vals[:top_n]

        rowsize = len(row_terms)
        assert rowsize == len(row_vals)

        if table_doc_labels is not None:
            if rowsize > 0:
                res.append(pd.DataFrame({'doc': np.repeat(table_doc_labels[i], repeats=rowsize),
                                         'token': row_terms,
                                         'value': row_vals}))
        else:
            res.append(list(zip(row_terms, row_vals)))

    if table_doc_labels is not None:
        if res:
            return pd.concat(res, axis=0)
        else:
            return pd.DataFrame({'doc': [], 'token': [], 'value': []})
    else:
        return res


def sorted_terms_table(mat, vocab, doc_labels, lo_thresh=0, hi_tresh=None, top_n=None, ascending=False):
    """
    Shortcut function for :func:`~tmtoolkit.bow.bow_stats.sorted_terms` which generates a data table with `doc_labels`.

    :param mat: (sparse) document-term-matrix `mat` (may be tf-idf transformed or any other transformation)
    :param vocab: list or array of vocabulary corresponding to columns in `mat`
    :param doc_labels: list/array of document labels corresponding to `mat` rows
    :param lo_thresh: if not None, filter for values greater than `lo_thresh`
    :param hi_tresh: if not None, filter for values lesser than or equal `hi_thresh`
    :param top_n: if not None, select only the top `top_n` terms
    :param ascending: sorting direction
    :return: data table with columns "doc", "term", "value"
    """
    return sorted_terms(mat, vocab, lo_thresh=lo_thresh, hi_tresh=hi_tresh, top_n=top_n,
                        ascending=ascending, table_doc_labels=doc_labels)
