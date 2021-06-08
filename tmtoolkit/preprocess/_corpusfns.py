"""
Functions that operate on lists of spaCy documents.
"""

import operator
import string
from collections import Counter, OrderedDict
from functools import partial

import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab

from ._common import DEFAULT_LANGUAGE_MODELS, load_stopwords, simplified_pos
from ._tokenfuncs import (
    require_tokendocs, token_match, token_match_subsequent, token_glue_subsequent, make_index_window_around_matches,
    expand_compound_token
)
from ..utils import require_listlike, require_types, flatten_list, empty_chararray, widen_chararray
from ..bow.dtm import create_sparse_dtm
from .._pd_dt_compat import pd_dt_frame, pd_dt_concat, pd_dt_sort


#%% initialization and tokenization

def init_corpus_language(corpus, language=None, language_model=None, **spacy_opts):
    if language is None and language_model is None:
        raise ValueError('either `language` or `language_model` must be given')

    if language_model is None:
        if not isinstance(language, str) or len(language) != 2:
            raise ValueError('`language` must be a two-letter ISO 639-1 language code')

        if language not in DEFAULT_LANGUAGE_MODELS:
            raise ValueError('language "%s" is not supported' % language)
        language_model = DEFAULT_LANGUAGE_MODELS[language] + '_sm'

    spacy_kwargs = dict(disable=['parser', 'ner'])
    spacy_kwargs.update(spacy_opts)

    corpus.nlp = spacy.load(language_model, **spacy_kwargs)
    corpus.language = language


def tokenize(corpus):
    # TODO: check corpus nlp, check already tokenized, inplace arg?

    tokenizerpipe = corpus.nlp.pipe(corpus.values(), n_process=corpus.n_max_workers)
    corpus.docs = {dl: _init_doc(dt, doc_label=dl) for dl, dt in zip(corpus.keys(), tokenizerpipe)}
    corpus.tokenized = True

    return corpus


def tokendocs2spacydocs(docs, vocab=None, doc_labels=None, return_vocab=False):
    """
    Create new spaCy documents from token lists in `docs`.

    .. note:: spaCy doesn't handle empty tokens (`""`), hence these tokens will not appear in the resulting spaCy
              documents if they exist in the input documents.

    :param docs: list of document tokens
    :param vocab: provide vocabulary to be used when generating spaCy documents; if no vocabulary is given, it will be
                  generated from `docs`
    :param doc_labels: optional list of document labels; if given, must be of same length as `docs`
    :param return_vocab: if True, additionally return generated vocabulary as spaCy `Vocab` object
    :return: list of spaCy documents or tuple with additional generated vocabulary if `return_vocab` is True
    """
    require_tokendocs(docs)

    if doc_labels is not None and len(doc_labels) != len(docs):
        raise ValueError('`doc_labels` must have the same length as `docs`')

    if vocab is None:
        vocab = Vocab(strings=list(vocabulary(docs) - {''}))
    else:
        vocab = Vocab(strings=vocab.tolist() if isinstance(vocab, np.ndarray) else list(vocab))

    spacydocs = []
    for i, tokdoc in enumerate(docs):
        spacydocs.append(spacydoc_from_tokens(tokdoc, vocab=vocab, label=None if doc_labels is None else doc_labels[i]))

    if return_vocab:
        return spacydocs, vocab
    else:
        return spacydocs


#%% functions that operate on Corpus objects

def doc_tokens(docs, to_lists=False):
    """
    If `docs` is a list of spaCy documents, return the (potentially filtered) tokens from these documents as list of
    string tokens, otherwise return the input list as-is.

    :param docs: list of string tokens or spaCy documents
    :param to_lists: if `docs` is list of spaCy documents or list of NumPy arrays, convert result to lists
    :return: list of string tokens as NumPy arrays (default) or lists (if `to_lists` is True)
    """
    # TODO: check corpus nlp, check already tokenized

    if to_lists:
        fn = partial(_filtered_doc_tokens, as_list=True)
    else:
        fn = _filtered_doc_tokens

    return list(map(fn, docs.values()))


def vocabulary(docs, sort=False):
    """
    Return vocabulary, i.e. set of all tokens that occur at least once in at least one of the documents in `docs`.

    :param docs: list of string tokens or spaCy documents
    :param sort: return as sorted list
    :return: either set of token strings or sorted list if `sort` is True
    """
    # TODO: check corpus nlp, check already tokenized

    from ..corpus import Corpus

    v = set(flatten_list(doc_tokens(docs, to_lists=True) if isinstance(docs, Corpus) else docs))

    if sort:
        return sorted(v)
    else:
        return v


def doc_lengths(docs):
    """
    Return document length (number of tokens in doc.) for each document.

    :param docs: list of string tokens or spaCy documents
    :return: list of document lengths
    """
    # TODO: check corpus nlp, check already tokenized, arg as dict?

    return list(map(len, doc_tokens(docs)))


def vocabulary_counts(docs):
    """
    Return :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
    all documents.

    :param docs: list of string tokens or spaCy documents
    :return: :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
             all documents
    """
    # TODO: check corpus nlp, check already tokenized

    return Counter(flatten_list(doc_tokens(docs)))


def doc_frequencies(docs, proportions=False):
    """
    Document frequency per vocabulary token as dict with token to document frequency mapping.
    Document frequency is the measure of how often a token occurs *at least once* in a document.
    Example with absolute document frequencies:

    .. code-block:: text

        doc tokens
        --- ------
        A   z, z, w, x
        B   y, z, y
        C   z, z, y, z

        document frequency df(z) = 3  (occurs in all 3 documents)
        df(x) = df(w) = 1 (occurs only in A)
        df(y) = 2 (occurs in B and C)
        ...

    :param docs: list of string tokens or spaCy documents
    :param proportions: if True, normalize by number of documents to obtain proportions
    :return: dict mapping token to document frequency
    """
    # TODO: check corpus nlp, check already tokenized

    doc_freqs = Counter()

    for dtok in docs.values():
        for t in set(_filtered_doc_tokens(dtok, as_list=True)):
            doc_freqs[t] += 1

    if proportions:
        n_docs = len(docs)
        return {w: n/n_docs for w, n in doc_freqs.items()}
    else:
        return doc_freqs


#%% functions that operate on a single spacy document

def spacydoc_from_tokens(tokens, vocab=None, spaces=None, lemmata=None, label=None):
    """
    Create a new spaCy ``Doc`` document with tokens `tokens`.

    :param tokens: list, tuple or NumPy array of string tokens
    :param vocab: list, tuple, set, NumPy array or spaCy ``Vocab`` object with vocabulary; if None, vocabulary will be
                  generated from `tokens`
    :param spaces: list, tuple or NumPy array of whitespace for each token
    :param lemmata: list, tuple or NumPy array of string lemmata for each token
    :param label: document label
    :return: spaCy ``Doc`` document
    """
    require_types(tokens, (tuple, list, np.ndarray), error_msg='the argument must be a list, tuple or NumPy array')

    tokens = [t for t in (tokens.tolist() if isinstance(tokens, np.ndarray) else tokens)
              if t]    # spaCy doesn't accept empty tokens and also no NumPy "np.str_" type tokens

    if vocab is None:
        vocab = Vocab(strings=list(set(tokens)))
    elif not isinstance(vocab, Vocab):
        vocab = Vocab(strings=vocab.tolist() if isinstance(vocab, np.ndarray) else list(vocab))

    if lemmata is not None and len(lemmata) != len(tokens):
        raise ValueError('`lemmata` must have the same length as `tokens`')

    new_doc = Doc(vocab, words=tokens, spaces=spaces)
    assert len(new_doc) == len(tokens)

    _init_doc(new_doc, doc_label=label)

    if lemmata is not None:
        lemmata = lemmata.tolist() if isinstance(lemmata, np.ndarray) else lemmata
        for t, lem in zip(new_doc, lemmata):
            t.lemma_ = lem

    return new_doc


#%% helper functions


def _filtered_doc_tokens(doc, as_list=False):
    if len(doc) == 0 or doc.user_data['mask'].sum() == 0:
        return [] if as_list else empty_chararray()

    res = np.asarray([t.text for t in doc])[doc.user_data['mask']]
    return res.tolist() if as_list else res


def _init_doc(doc, mask=None, doc_label=None):   # tokens=None
    assert isinstance(doc, Doc)

    if doc_label is not None:
        doc._.label = doc_label

    if mask is not None:
        assert isinstance(mask, np.ndarray)
        assert mask.dtype.kind == 'b'
        assert len(doc) == len(mask)

    doc.user_data['mask'] = mask if isinstance(mask, np.ndarray) else np.repeat(True, len(doc))

    return doc

