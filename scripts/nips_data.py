"""
Convert NIPS data from http://archive.ics.uci.edu/ml/datasets/Bag+of+Words to sparse DTM format stored as pickle file.

Markus Konrad <markus.konrad@wzb.eu>
"""

import numpy as np
from scipy.sparse import coo_matrix
from tmtoolkit.utils import pickle_data


#%%

with open('fulldata/vocab.nips.txt') as f:
    vocab = np.array([l.strip() for l in f.readlines() if l.strip()])

#%%

n_docs = None
n_vocab = None
n_nonzero = None
entries = []
row_ind = []
col_ind = []

with open('fulldata/docword.nips.txt') as f:
    for i, l in enumerate(f):
        l = l.strip()

        if i < 3:
            n = int(l)
            if i == 0:
                n_docs = n
            elif i == 1:
                n_vocab = n
            elif i == 2:
                n_nonzero = n
        else:
            j, k, n = list(map(int, l.split()))
            entries.append(n)
            row_ind.append(j-1)   # convert to zero-based index
            col_ind.append(k-1)   # convert to zero-based index


assert len(vocab) == n_vocab
assert len(entries) == len(row_ind) == len(col_ind) == n_nonzero

dtm = coo_matrix((entries, (row_ind, col_ind)), shape=(n_docs, n_vocab), dtype='int64')

doc_labels = np.fromiter((f'doc{str(i+1).zfill(4)}' for i in range(n_docs)), dtype='<U7', count=n_docs)

#%%

pickle_data((doc_labels, vocab, dtm), '../examples/data/nips.pickle')
