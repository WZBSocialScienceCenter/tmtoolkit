"""
Common utility functions for LDA model evaluation.
"""

import numpy as np
from scipy.sparse import issparse


def split_dtm_for_cross_validation(dtm, n_folds, shuffle_docs=True):
    """
    Split a (sparse) document-term matrix `dtm` for n-fold cross validation with `n_folds` folds.

    :param dtm: (sparse) document-term matrix
    :param n_folds: number of folds during cross validation
    :param shuffle_docs: shuffle documents (matrix rows) before splitting
    :return: a generator for `n_folds` folds, each yielding a 3-tuple with (fold index starting at zero, training DTM,
             test DTM)
    """
    if issparse(dtm) and dtm.format != 'csr':
        dtm = dtm.tocsr()

    n_docs = dtm.shape[0]

    if n_folds < 2:
        raise ValueError('`n_folds` must be at least 2')

    if n_docs < n_folds:
        raise ValueError('not enough documents in `dtm` (must be >= `n_folds`)')

    rand_doc_ind = np.arange(n_docs)

    if shuffle_docs:
        np.random.shuffle(rand_doc_ind)

    n_per_fold = n_docs // n_folds
    assert n_per_fold > 0
    start_idx = 0
    for fold in range(n_folds):
        end_idx = start_idx + n_per_fold
        fold_doc_ind = rand_doc_ind[slice(start_idx, end_idx)]
        test_dtm = dtm[fold_doc_ind, :]

        if issparse(dtm):
            inv_fold_doc_ind = np.ones(n_docs, np.bool)
            inv_fold_doc_ind[fold_doc_ind] = 0
            train_dtm = dtm[inv_fold_doc_ind, :]
        else:
            train_dtm = np.delete(dtm, fold_doc_ind, axis=0)   # can't be used with sparse matrices

        assert test_dtm.shape[0] + train_dtm.shape[0] == dtm.shape[0]

        yield fold, train_dtm, test_dtm

        start_idx = end_idx


class FakedGensimDict:
    """
    A class that resembles a Gensim :class:`~gensim.corpora.dictionary.Dictionary`.
    """
    def __init__(self, data):
        if not isinstance(data, dict):
            raise ValueError('`data` must be an instance of `dict`')

        self.id2token = data
        self.token2id = {v: k for k, v in data.items()}

    @staticmethod
    def from_vocab(vocab):
        return FakedGensimDict(dict(zip(range(len(vocab)), vocab)))

    def __iter__(self):
        """Iterate over all tokens."""
        return iter(self.keys())

    # restore Py2-style dict API
    iterkeys = __iter__

    def iteritems(self):
        return self.items()

    def itervalues(self):
        return self.values()

    def keys(self):
        """Get all stored ids.

        Returns
        -------
        list of int
            List of all token ids.

        """
        return list(self.token2id.values())
