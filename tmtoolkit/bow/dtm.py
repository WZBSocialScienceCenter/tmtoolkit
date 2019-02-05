"""
Functions for creating a document-term-matrix (DTM) and some compatibility functions for Gensim.
"""

import numpy as np
from scipy.sparse import coo_matrix, issparse


#%% DTM creation


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
        terms_unique = set(terms)
        vocab |= terms_unique

        # update the sum of unique values per document
        sum_uniques_per_doc += len(terms_unique)

    doc_labels = docs_terms.keys()

    return np.fromiter(vocab, dtype='<U%d' % max(map(len, vocab)), count=len(vocab)), \
           np.fromiter(doc_labels, dtype='<U%d' % max(map(len, doc_labels)), count=len(doc_labels)),\
           docs_terms, sum_uniques_per_doc


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