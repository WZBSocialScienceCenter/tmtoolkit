"""
Common functions and constants.
"""

from collections import Counter

import nltk

from .. import defaults
from ..bow.dtm import create_sparse_dtm
from ..utils import flatten_list, require_listlike


PATTERN_SUBMODULES = {
    'english': 'en',
    'german': 'de',
    'spanish': 'es',
    'french': 'fr',
    'italian': 'it',
    'dutch': 'nl',
}


def tokenize(docs, language=defaults.language):
    require_listlike(docs)

    return [nltk.tokenize.word_tokenize(text, language) for text in docs]


def doc_lengths(docs):
    return list(map(len, docs))


def vocabulary(docs, sort=False):
    v = set(flatten_list(docs))

    if sort:
        return sorted(v)
    else:
        return v


def vocabulary_counts(docs):
    return Counter(flatten_list(docs))


def doc_frequencies(docs, proportions=False):
    doc_freqs = Counter()

    for dtok in docs:
        for t in set(dtok):
            doc_freqs[t] += 1

    if proportions:
        n_docs = len(docs)
        return {w: n/n_docs for w, n in doc_freqs.items()}
    else:
        return doc_freqs


def ngrams(docs, n, join=True, join_str=' '):
    return [_ngrams_from_tokens(dtok, n=n, join=join, join_str=join_str) for dtok in docs]


def sparse_dtm(docs, vocab=None):
    """
    Create a sparse document-term-matrix (DTM) from a list of tokenized documents `docs`. If `vocab` is None, determine
    the vocabulary (unique terms) from `docs`, otherwise take `vocab` which must be a *sorted* list or NumPy array.
    If `vocab` is None, the generated sorted vocabulary list is returned as second value, else only a single value is
    returned -- the DTM.

    :param docs: list of tokenized documents
    :param vocab: optional *sorted* list / NumPy array of vocabulary (unique terms) in `docs`
    :return: either a single value (sparse document-term-matrix) or a tuple with sparse DTM and sorted vocabulary if
             none was passed
    """

    if vocab is None:
        vocab = vocabulary(docs, sort=True)
        return_vocab = True
    else:
        return_vocab = False

    alloc_size = sum(len(set(dtok)) for dtok in docs)  # sum of *unique* tokens in each document

    dtm = create_sparse_dtm(vocab, docs, alloc_size, vocab_is_sorted=True)

    if return_vocab:
        return dtm, vocab
    else:
        return dtm


def _ngrams_from_tokens(tokens, n, join=True, join_str=' '):
    if n < 2:
        raise ValueError('`n` must be at least 2')

    if len(tokens) == 0:
        return []

    if len(tokens) < n:
        # raise ValueError('`len(tokens)` should not be smaller than `n`')
        ngrams = [tokens]
    else:
        ngrams = [[tokens[i+j] for j in range(n)]
                  for i in range(len(tokens)-n+1)]
    if join:
        return list(map(lambda x: join_str.join(x), ngrams))
    else:
        return ngrams