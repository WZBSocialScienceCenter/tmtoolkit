"""
Functions for creating a document-term-matrix (DTM) and some compatibility functions for Gensim.
"""

import numpy as np
import datatable as dt
from scipy.sparse import coo_matrix, issparse


#%% DTM creation


def create_sparse_dtm(vocab, doc_labels, docs_terms, sum_uniques_per_doc, vocab_is_sorted=False, dtype=None):
    """
    Create a sparse document-term-matrix (DTM) as matrix in COO sparse format from vocabulary array `vocab`, document
    IDs/labels array `doc_labels`, dict of doc_label -> document terms `docs_terms` and the sum of unique terms
    per document `sum_uniques_per_doc`.
    The DTM's rows are document names, its columns are indices in `vocab`, hence a value `DTM[j, k]` is the
    term frequency of term `vocab[k]` in `docnames[j]`.

    A note on performance: Creating the three arrays for a COO matrix seems to be the fastest way to generate a DTM.
    An alternative implementation using LIL format was 2x slower.

    Memory requirement: about 3 * <sum_uniques_per_doc> * 4 bytes.
    """
    if dtype is None:
        dtype = np.intc

    if not isinstance(doc_labels, np.ndarray):
        doc_labels = np.array(doc_labels)

    if vocab_is_sorted:
        vocab_sorter = None
    else:
        vocab_sorter = np.argsort(vocab)  # indices that sort <vocab>

    nvocab = len(vocab)
    ndocs = len(doc_labels)

    # create arrays for sparse matrix
    data = np.empty(sum_uniques_per_doc, dtype=dtype)  # all non-zero term frequencies at data[k]
    cols = np.empty(sum_uniques_per_doc, dtype=dtype)  # column index for kth data item (kth term freq.)
    rows = np.empty(sum_uniques_per_doc, dtype=dtype)  # row index for kth data item (kth term freq.)

    ind = 0  # current index in the sparse matrix data
    # go through all documents with their terms
    for i, (doc_label, terms) in enumerate(docs_terms.items()):
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
        doc_idx = np.where(doc_labels == doc_label)  # get the document index for the document name
        assert len(doc_idx) == 1
        rows[ind:ind_end] = np.repeat(doc_idx, n_vals)  # save it as repeated value

        ind = ind_end

    assert ind == len(data)

    return coo_matrix((data, (rows, cols)), shape=(ndocs, nvocab), dtype=dtype)


def dtm_to_dataframe(dtm, doc_labels, vocab):
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError('package `pandas` must be installed to use this function')

    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    if dtm.shape[0] != len(doc_labels):
        raise ValueError('number of rows must be equal to `len(doc_labels)')

    if dtm.shape[1] != len(vocab):
        raise ValueError('number of rows must be equal to `len(vocab)')

    if not isinstance(dtm, np.ndarray):
        dtm = dtm.toarray()

    return pd.DataFrame(dtm, index=doc_labels, columns=vocab)


def dtm_to_datatable(dtm, doc_labels, vocab, colname_rowindex='_doc'):
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    if dtm.shape[0] != len(doc_labels):
        raise ValueError('number of rows must be equal to `len(doc_labels)')

    if dtm.shape[1] != len(vocab):
        raise ValueError('number of rows must be equal to `len(vocab)')

    if not isinstance(dtm, np.ndarray):
        dtm = dtm.toarray()

    return dt.cbind(dt.Frame({colname_rowindex: doc_labels}),
                    dt.Frame(dtm, names=vocab))


#%% Gensim compatibility functions


def dtm_to_gensim_corpus(dtm):
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
    import gensim
    from scipy.sparse import coo_matrix

    dtm_t = gensim.matutils.corpus2csc(corpus)
    return coo_matrix(dtm_t.transpose())


def dtm_and_vocab_to_gensim_corpus_and_dict(dtm, vocab, as_gensim_dictionary=True):
    corpus = dtm_to_gensim_corpus(dtm)

    # vocabulary array has to be converted to dict with index -> word mapping
    id2word = dict(zip(range(len(vocab)), vocab))

    if as_gensim_dictionary:
        import gensim
        return corpus, gensim.corpora.dictionary.Dictionary().from_corpus(corpus, id2word)
    else:
        return corpus, id2word
