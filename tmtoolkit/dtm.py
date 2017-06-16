# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import coo_matrix

from .utils import pickle_data, unpickle_file


def get_vocab_and_terms(docs):
    """
    From a dict `docs` with document ID -> terms/tokens list mapping, generate an array of vocabulary (i.e.
    unique terms of the whole corpus `docs`), an array of document labels (i.e. document IDs), a dict
    with document ID -> document terms array mapping and a sum of the number of unique terms per document.

    The returned variable `sum_uniques_per_doc` tells us how many elements will be non-zero in a DTM which
    will be created later. Hence this is the allocation size for the sparse DTM.

    This function provides the input for create_sparse_dtm().

    Return a tuple with:
    - np.array of vocabulary
    - np.array of document names
    - dict with mapping: document name -> np.array of document terms
    - overall sum of unique terms per document (allocation size for the sparse DTM)
    """
    vocab = set()
    docs_terms = {}
    sum_uniques_per_doc = 0

    # go through the documents
    for i, (doc_label, terms) in enumerate(docs.items()):
        terms_arr = np.array(terms)
        docs_terms[doc_label] = terms_arr

        # update the vocab set
        vocab |= set(terms)

        # update the sum of unique values per document
        sum_uniques_per_doc += len(np.unique(terms_arr))

    return np.array(list(vocab)), np.array(list(docs_terms.keys())), docs_terms, sum_uniques_per_doc


def create_sparse_dtm(vocab, doc_labels, docs_terms, sum_uniques_per_doc):
    """
    Create a sparse document-term-matrix (DTM) as scipy "coo_matrix" from vocabulary array `vocab`, document
    IDs/labels array `doc_labels`, dict of doc_label -> document terms `docs_terms` and the sum of unique terms
    per document `sum_uniques_per_doc`.
    The DTM's rows are document names, its columns are indices in `vocab`, hence a value `DTM[j, k]` is the
    term frequency of term `vocab[k]` in `docnames[j]`.

    Memory requirement: about 3 * <sum_uniques_per_doc>.
    """
    vocab_sorter = np.argsort(vocab)  # indices that sort <vocab>

    nvocab = len(vocab)
    ndocs = len(doc_labels)

    # create arrays for sparse matrix
    data = np.empty(sum_uniques_per_doc, dtype=np.intc)  # all non-zero term frequencies at data[k]
    cols = np.empty(sum_uniques_per_doc, dtype=np.intc)  # column index for kth data item (kth term freq.)
    rows = np.empty(sum_uniques_per_doc, dtype=np.intc)  # row index for kth data item (kth term freq.)

    ind = 0  # current index in the sparse matrix data
    # go through all documents with their terms
    for i, (doc_label, terms) in enumerate(docs_terms.items()):
        if len(terms) == 0: continue   # skip empty documents

        # find indices into `vocab` such that, if the corresponding elements in `terms` were
        # inserted before the indices, the order of `vocab` would be preserved
        # -> array of indices of `terms` in `vocab`
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

    return coo_matrix((data, (rows, cols)), shape=(ndocs, nvocab), dtype=np.intc)


def save_dtm_to_pickle(dtm, vocab, docnames, picklefile):
    """Save a DTM as pickle file."""
    pickle_data({'dtm': dtm, 'vocab': vocab, 'docnames': docnames}, picklefile)


def load_dtm_from_pickle(picklefile):
    """Load a DTM from a pickle file."""
    data = unpickle_file(picklefile)
    assert data['dtm'].shape[0] == len(data['docnames'])
    assert data['dtm'].shape[1] == len(data['vocab'])

    return data['dtm'], data['vocab'], data['docnames']
