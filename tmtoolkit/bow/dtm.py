"""
Functions for creating a document-term matrix (DTM) and some compatibility functions for Gensim.
"""

import numpy as np
from scipy.sparse import coo_matrix, issparse

import pandas as pd


#%% DTM creation

def create_sparse_dtm(vocab, docs, n_unique_tokens, vocab_is_sorted=False, dtype=np.intc):
    """
    Create a sparse document-term-matrix (DTM) as matrix in
    `COO sparse format <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix>`_
    from vocabulary array `vocab`, a list of tokenized documents `docs` and the number of unique tokens across all
    documents `n_unique_tokens`.

    The DTM's rows are document names, its columns are indices in `vocab`, hence a value ``DTM[j, k]`` is the
    term frequency of term ``vocab[k]`` in document ``j``.

    A note on performance: Creating the three arrays for a COO matrix seems to be the fastest way to generate a DTM.
    An alternative implementation using LIL format was ~2x slower.

    Memory requirement: about ``3 * <n_unique_tokens> * 4`` bytes with default dtype (32-bit integer).

    .. seealso:: This is the "low level" function. For the straight-forward to use function see
                 :func:`tmtoolkit.preprocess.sparse_dtm`, which also calculates `n_unique_tokens`.

    :param vocab: list or array of vocabulary used as column names; size must equal number of columns in `dtm`
    :param docs: a list of tokenized documents
    :param n_unique_tokens: number of unique tokens across all documents
    :param vocab_is_sorted: if True, assume that `vocab` is sorted when creating the token IDs
    :param dtype: data type of the resulting matrix
    :return: a sparse document-term-matrix in COO sparse format
    """

    if vocab_is_sorted:
        vocab_sorter = None
    else:
        vocab_sorter = np.argsort(vocab)  # indices that sort <vocab>

    nvocab = len(vocab)
    ndocs = len(docs)

    # create arrays for sparse matrix
    data = np.empty(n_unique_tokens, dtype=dtype)  # all non-zero term frequencies at data[k]
    cols = np.empty(n_unique_tokens, dtype=dtype)  # column index for kth data item (kth term freq.)
    rows = np.empty(n_unique_tokens, dtype=dtype)  # row index for kth data item (kth term freq.)

    ind = 0  # current index in the sparse matrix data
    # go through all documents with their terms
    for doc_idx, terms in enumerate(docs):
        if len(terms) == 0: continue   # skip empty documents

        # find indices into `vocab` such that, if the corresponding elements in `terms` were
        # inserted before the indices, the order of `vocab` would be preserved
        # -> array of indices of `terms` in `vocab`
        if vocab_is_sorted:
            term_indices = np.searchsorted(vocab, terms)
        else:
            term_indices = vocab_sorter[np.searchsorted(vocab, terms, sorter=vocab_sorter)]

        # count the unique terms of the document and get their vocabulary indices
        uniq_indices, counts = np.unique(term_indices, return_counts=True)
        n_vals = len(uniq_indices)
        ind_end = ind + n_vals

        data[ind:ind_end] = counts  # save the counts (term frequencies)
        cols[ind:ind_end] = uniq_indices  # save the column index: index in <vocab>
        rows[ind:ind_end] = np.repeat(doc_idx, n_vals)  # save it as repeated value

        ind = ind_end

    assert ind == len(data)

    return coo_matrix((data, (rows, cols)), shape=(ndocs, nvocab), dtype=dtype)


def dtm_to_dataframe(dtm, doc_labels, vocab):
    """
    Convert a (sparse) DTM to a pandas DataFrame using document labels `doc_labels` as row index and `vocab` as column
    names.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw terms counts
    :param doc_labels: document labels used as row index (row names); size must equal number of rows in `dtm`
    :param vocab: list or array of vocabulary used as column names; size must equal number of columns in `dtm`
    :return: pandas DataFrame
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    if dtm.shape[0] != len(doc_labels):
        raise ValueError('number of rows must be equal to `len(doc_labels)')

    if dtm.shape[1] != len(vocab):
        raise ValueError('number of rows must be equal to `len(vocab)')

    if not isinstance(dtm, np.ndarray):
        dtm = dtm.toarray()

    return pd.DataFrame(dtm, index=doc_labels, columns=vocab)


#%% Gensim compatibility functions


def dtm_to_gensim_corpus(dtm):
    """
    Convert a (sparse) DTM to a Gensim Corpus object.

    .. seealso:: :func:`~tmtoolkit.bow.dtm.gensim_corpus_to_dtm` for the reverse function or
                 :func:`~tmtoolkit.bow.dtm.dtm_and_vocab_to_gensim_corpus_and_dict` which additionally creates a Gensim
                 :class:`~gensim.corpora.dictionary.Dictionary`.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw terms counts
    :return: a Gensim :class:`gensim.matutils.Sparse2Corpus` object
    """
    import gensim

    # DTM with documents to words sparse matrix in COO format has to be converted to transposed sparse matrix in CSC
    # format
    dtm_t = dtm.transpose()

    if issparse(dtm_t):
        if dtm_t.format != 'csc':
            dtm_sparse = dtm_t.tocsc()
        else:
            dtm_sparse = dtm_t
    else:
        from scipy.sparse.csc import csc_matrix
        dtm_sparse = csc_matrix(dtm_t)

    return gensim.matutils.Sparse2Corpus(dtm_sparse)


def gensim_corpus_to_dtm(corpus):
    """
    Convert a Gensim corpus object to a sparse DTM in COO format.

    .. seealso:: :func:`~tmtoolkit.bow.dtm.dtm_to_gensim_corpus` for the reverse function.

    :param corpus: Gensim corpus object
    :return: sparse DTM in COO format
    """
    import gensim
    from scipy.sparse import coo_matrix

    dtm_t = gensim.matutils.corpus2csc(corpus)
    return coo_matrix(dtm_t.transpose())


def dtm_and_vocab_to_gensim_corpus_and_dict(dtm, vocab, as_gensim_dictionary=True):
    """
    Convert a (sparse) DTM *and* a vocabulary list to a Gensim Corpus object and
    Gensim :class:`~gensim.corpora.dictionary.Dictionary` object or a Python :func:`dict`.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw terms counts
    :param vocab: list or array of vocabulary
    :param as_gensim_dictionary: if True create Gensim :class:`~gensim.corpora.dictionary.Dictionary` from `vocab`,
                                 else create Python :func:`dict`
    :return: a 2-tuple with (Corpus object, Gensim :class:`~gensim.corpora.dictionary.Dictionary` or
             Python :func:`dict`)
    """
    corpus = dtm_to_gensim_corpus(dtm)

    # vocabulary array has to be converted to dict with index -> word mapping
    id2word = dict(zip(range(len(vocab)), vocab))

    if as_gensim_dictionary:
        import gensim
        return corpus, gensim.corpora.dictionary.Dictionary().from_corpus(corpus, id2word)
    else:
        return corpus, id2word
