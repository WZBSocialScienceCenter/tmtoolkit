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


#%% global spaCy nlp instance

#: Global spaCy nlp instance which must be initiated via :func:`tmtoolkit.preprocess.init_for_language` when using
#: the functional preprocess API
nlp = None


#%% initialization and tokenization

def init_for_language(language=None, language_model=None, **spacy_opts):
    """
    Initialize the functional API for a given language code `language` or a spaCy language model `language_model`.
    The spaCy nlp instance will be returned and will also be used by default in all subsequent preprocess API calls.

    :param language: two-letter ISO 639-1 language code (lowercase)
    :param language_model: spaCy language model `language_model`
    :param spacy_opts: additional keyword arguments passed to ``spacy.load()``
    :return: spaCy nlp instance
    """
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

    global nlp
    nlp = spacy.load(language_model, **spacy_kwargs)

    return nlp


def tokenize(docs, as_spacy_docs=True, doc_labels=None, doc_labels_fmt='doc-{i1}', enable_vectors=False,
             nlp_instance=None):
    """
    Tokenize a list or dict of documents `docs`, where each element contains the raw text of the document as string.

    Requires that :func:`~tmtoolkit.preprocess.init_for_language` is called before or `nlp_instance` is passed.

    :param docs: list or dict of documents with raw text strings; if dict, use dict keys as document labels
    :param as_spacy_docs: if True, return list of spaCy ``Doc`` objects, otherwise return list of string tokens
    :param doc_labels: if not None and `docs` is a list, use strings in this list as document labels
    :param doc_labels_fmt: if `docs` is a list and `doc_labels` is None, generate document labels according to this
                           format, where ``{i0}`` or ``{i1}`` are replaced by the respective zero- or one-indexed
                           document numbers
    :param enable_vectors: if True, generate word vectors (aka word embeddings) during tokenization;
                           this will be more computationally expensive
    :param nlp_instance: spaCy nlp instance
    :return: list of spaCy ``Doc`` documents if `as_spacy_docs` is True (default) or list of string token documents
    """

    dictlike = hasattr(docs, 'keys') and hasattr(docs, 'values')

    if not isinstance(docs, (list, tuple)) and not dictlike:
        raise ValueError('`docs` must be a list, tuple or dict-like object')

    if not isinstance(doc_labels_fmt, str):
        raise ValueError('`doc_labels_fmt` must be a string')

    _nlp = _current_nlp(nlp_instance)

    if doc_labels is None:
        if dictlike:
            doc_labels = docs.keys()
            docs = docs.values()
        elif as_spacy_docs:
            doc_labels = [doc_labels_fmt.format(i0=i, i1=i+1) for i in range(len(docs))]
    elif len(doc_labels) != len(docs):
        raise ValueError('`doc_labels` must have same length as `docs`')

    if enable_vectors:
        tokenized_docs = [_nlp(d) for d in docs]
    else:
        tokenized_docs = [_nlp.make_doc(d) for d in docs]

    del docs

    if as_spacy_docs:
        for dl, doc in zip(doc_labels, tokenized_docs):
            doc._.label = dl
            _init_doc(doc)

        return tokenized_docs
    else:
        return [[t.text for t in doc] for doc in tokenized_docs]


#%% functions that operate on lists of string tokens *or* spacy documents

def doc_tokens(docs, to_lists=False):
    """
    If `docs` is a list of spaCy documents, return the (potentially filtered) tokens from these documents as list of
    string tokens, otherwise return the input list as-is.

    :param docs: list of string tokens or spaCy documents
    :param to_lists: if `docs` is list of spaCy documents or list of NumPy arrays, convert result to lists
    :return: list of string tokens as NumPy arrays (default) or lists (if `to_lists` is True)
    """
    require_spacydocs_or_tokens(docs)

    if to_lists:
        fn = partial(_filtered_doc_tokens, as_list=True)
    else:
        fn = _filtered_doc_tokens

    return list(map(fn, docs))


def tokendocs2spacydocs(docs, vocab=None, doc_labels=None, return_vocab=False):
    """
    TODO: docu. + tests

    :param docs:
    :param vocab:
    :param doc_labels:
    :return:
    """
    require_tokendocs(docs)

    if doc_labels is not None and len(doc_labels) != len(docs):
        raise ValueError('`doc_labels` must have the same length as `docs`')

    if vocab is None:
        vocab = Vocab(strings=list(vocabulary(docs) - {''}))

    spacydocs = []
    for i, tokdoc in enumerate(docs):
        spacydocs.append(spacydoc_from_tokens(tokdoc, vocab=vocab, label='' if doc_labels is None else doc_labels[i]))

    if return_vocab:
        return spacydocs, vocab
    else:
        return spacydocs


def doc_lengths(docs):
    """
    Return document length (number of tokens in doc.) for each document.

    :param docs: list of string tokens or spaCy documents
    :return: list of document lengths
    """
    require_spacydocs_or_tokens(docs)

    return list(map(len, doc_tokens(docs)))


def vocabulary(docs, sort=False):
    """
    Return vocabulary, i.e. set of all tokens that occur at least once in at least one of the documents in `docs`.

    :param docs: list of string tokens or spaCy documents
    :param sort: return as sorted list
    :return: either set of token strings or sorted list if `sort` is True
    """
    require_spacydocs_or_tokens(docs)

    v = set(flatten_list(doc_tokens(docs, to_lists=True)))

    if sort:
        return sorted(v)
    else:
        return v


def vocabulary_counts(docs):
    """
    Return :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
    all documents.

    :param docs: list of string tokens or spaCy documents
    :return: :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
             all documents
    """
    require_spacydocs_or_tokens(docs)

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
        df(x) = df(w) = 1 (occur only in A)
        df(y) = 2 (occurs in B and C)
        ...

    :param docs: list of string tokens or spaCy documents
    :param proportions: if True, normalize by number of documents to obtain proportions
    :return: dict mapping token to document frequency
    """
    require_spacydocs_or_tokens(docs)

    doc_freqs = Counter()

    for dtok in docs:
        for t in set(_filtered_doc_tokens(dtok, as_list=True)):
            doc_freqs[t] += 1

    if proportions:
        n_docs = len(docs)
        return {w: n/n_docs for w, n in doc_freqs.items()}
    else:
        return doc_freqs


def ngrams(docs, n, join=True, join_str=' '):
    """
    Generate and return n-grams of length `n`.

    :param docs: list of string tokens or spaCy documents
    :param n: length of n-grams, must be >= 2
    :param join: if True, join generated n-grams by string `join_str`
    :param join_str: string used for joining
    :return: list of n-grams; if `join` is True, the list contains strings of joined n-grams, otherwise the list
             contains lists of size `n` in turn containing the strings that make up the n-gram
    """
    require_spacydocs_or_tokens(docs)

    return [_ngrams_from_tokens(_filtered_doc_tokens(dtok, as_list=True), n=n, join=join, join_str=join_str)
            for dtok in docs]


def sparse_dtm(docs, vocab=None):
    """
    Create a sparse document-term-matrix (DTM) from a list of tokenized documents `docs`. If `vocab` is None, determine
    the vocabulary (unique terms) from `docs`, otherwise take `vocab` which must be a *sorted* list or NumPy array.
    If `vocab` is None, the generated sorted vocabulary list is returned as second value, else only a single value is
    returned -- the DTM.

    :param docs: list of string tokens or spaCy documents
    :param vocab: optional *sorted* list / NumPy array of vocabulary (unique terms) in `docs`
    :return: either a single value (sparse document-term-matrix) or a tuple with sparse DTM and sorted vocabulary if
             none was passed
    """
    require_spacydocs_or_tokens(docs)

    if vocab is None:
        vocab = vocabulary(docs, sort=True)
        return_vocab = True
    else:
        return_vocab = False

    tokens = doc_tokens(docs)
    alloc_size = sum(len(set(dtok)) for dtok in tokens)  # sum of *unique* tokens in each document

    dtm = create_sparse_dtm(vocab, tokens, alloc_size, vocab_is_sorted=True)

    if return_vocab:
        return dtm, vocab
    else:
        return dtm


def kwic(docs, search_tokens, context_size=2, match_type='exact', ignore_case=False,
         glob_method='match', inverse=False, with_metadata=False, as_dict=False, as_datatable=False, non_empty=False,
         glue=None, highlight_keyword=None):
    """
    Perform keyword-in-context (kwic) search for search pattern(s) `search_tokens`. Returns result as list of KWIC
    windows or datatable / dataframe. If you want to filter with KWIC, use
    :func:`~tmtoolkit.preprocess.filter_tokens_with_kwic`, which returns results as list of tokenized documents (same
    structure as `docs`).

    Uses similar search parameters as :func:`~tmtoolkit.preprocess.filter_tokens`.

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param with_metadata: also return metadata (like POS) along with each token
    :param as_dict: if True, return result as dict with document labels mapping to KWIC results
    :param as_datatable: return result as data frame with indices "doc" (document label) and "context" (context
                          ID per document) and optionally "position" (original token position in the document) if
                          tokens are not glued via `glue` parameter
    :param non_empty: if True, only return non-empty result documents
    :param glue: if not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: if not None, this must be a string which is used to indicate the start and end of the
                              matched keyword
    :return: return either as: (1) list with KWIC results per document, (2) as dict with document labels mapping to
             KWIC results when `as_dict` is True or (3) dataframe / datatable when `as_datatable` is True
    """
    if as_dict or as_datatable:
        require_spacydocs(docs)   # because we need the document labels later
    else:
        require_spacydocs_or_tokens(docs)

    if isinstance(context_size, int):
        context_size = (context_size, context_size)
    else:
        require_listlike(context_size)

    if highlight_keyword is not None and not isinstance(highlight_keyword, str):
        raise ValueError('if `highlight_keyword` is given, it must be of type str')

    if glue:
        if with_metadata or as_datatable:
            raise ValueError('when `glue` is set to True, `with_metadata` and `as_datatable` must be False')
        if not isinstance(glue, str):
            raise ValueError('if `glue` is given, it must be of type str')

    kwic_raw = _build_kwic(docs, search_tokens,
                           highlight_keyword=highlight_keyword,
                           with_metadata=with_metadata,
                           with_window_indices=as_datatable,
                           context_size=context_size,
                           match_type=match_type,
                           ignore_case=ignore_case,
                           glob_method=glob_method,
                           inverse=inverse)

    if as_dict or as_datatable:
        kwic_raw = dict(zip(doc_labels(docs), kwic_raw))

    return _finalize_kwic_results(kwic_raw,
                                  non_empty=non_empty,
                                  glue=glue,
                                  as_datatable=as_datatable,
                                  with_metadata=with_metadata)


def kwic_table(docs, search_tokens, context_size=2, match_type='exact', ignore_case=False,
               glob_method='match', inverse=False, glue=' ', highlight_keyword='*'):
    """
    Shortcut for :func:`~tmtoolkit.preprocess.kwic()` to directly return a data frame table with highlighted keywords
    in context.

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`).
    :param inverse: Invert the matching results.
    :param glue: If not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                              matched keyword.
    :return: datatable or pandas DataFrame with columns "doc" (document label), "context" (context ID per document) and
             "kwic" containing strings with highlighted keywords in context.
    """

    kwic_raw = kwic(docs, search_tokens,
                    context_size=context_size,
                    match_type=match_type,
                    ignore_case=ignore_case,
                    glob_method=glob_method,
                    inverse=inverse,
                    with_metadata=False,
                    as_dict=True,
                    as_datatable=False,
                    non_empty=True,
                    glue=glue,
                    highlight_keyword=highlight_keyword)

    return _datatable_from_kwic_results(kwic_raw)


def glue_tokens(docs, patterns, glue='_', match_type='exact', ignore_case=False, glob_method='match', inverse=False,
                return_glued_tokens=False):
    """
    Match N *subsequent* tokens to the N patterns in `patterns` using match options like in
    :func:`~tmtoolkit.preprocess.filter_tokens`. Join the matched tokens by glue string `glue`. Replace these tokens
    in the documents.

    If there is metadata, the respective entries for the joint tokens are set to None.

    .. note:: If `docs` is a list of spaCy documents, this modifies the documents in `docs` in place.

    :param docs: list of string tokens or spaCy documents
    :param patterns: a sequence of search patterns as excepted by :func:`~tmtoolkit.preprocess.filter_tokens`
    :param glue: string for joining the subsequent matches
    :param match_type: one of: 'exact', 'regex', 'glob'; if 'regex', `search_token` must be RE pattern; if `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre)
    :param ignore_case: if True, ignore case for matching
    :param glob_method: if `match_type` is 'glob', use this glob method; must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: invert the matching results
    :param return_glued_tokens: if True, additionally return a set of tokens that were glued
    :return: updated documents `docs` if `docs` is a list of spaCy documents or otherwise a list of string token
             documents; if `return_glued_tokens` is True, return 2-tuple with additional set of tokens that were glued
    """
    is_spacydocs = require_spacydocs_or_tokens(docs)

    glued_tokens = set()

    if is_spacydocs is not None:
        match_opts = {'match_type': match_type, 'ignore_case': ignore_case, 'glob_method': glob_method}

        # all documents must be compact before applying "token_glue_subsequent"
        if is_spacydocs:
            docs = compact_documents(docs)

        res = []
        for doc in docs:
            # no need to use _filtered_doc_tokens() here because tokens are compact already
            matches = token_match_subsequent(patterns, _filtered_doc_tokens(doc), **match_opts)

            if inverse:
                matches = [~m for m in matches]

            if is_spacydocs:
                new_doc, glued = doc_glue_subsequent(doc, matches, glue=glue, return_glued=True)
            else:
                new_doc, glued = token_glue_subsequent(doc, matches, glue=glue, return_glued=True)

            res.append(new_doc)
            glued_tokens.update(glued)
    else:
        res = []

    if return_glued_tokens:
        return res, glued_tokens
    else:
        return res


def expand_compounds(docs, split_chars=('-',), split_on_len=2, split_on_casechange=False):
    """
    Expand all compound tokens in documents `docs`, e.g. splitting token "US-Student" into two tokens "US" and
    "Student".

    :param docs: list of string tokens or spaCy documents
    :param split_chars: characters to split on
    :param split_on_len: minimum length of a result token when considering splitting (e.g. when ``split_on_len=2``
                         "e-mail" would not be split into "e" and "mail")
    :param split_on_casechange: use case change to split tokens, e.g. "CamelCase" would become "Camel", "Case"
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    is_spacydocs = require_spacydocs_or_tokens(docs)

    if is_spacydocs is None:
        return []

    exp_comp = partial(expand_compound_token, split_chars=split_chars, split_on_len=split_on_len,
                       split_on_casechange=split_on_casechange)

    list_creator = list if is_spacydocs else flatten_list
    exptoks = [list_creator(map(exp_comp, _filtered_doc_tokens(doc))) for doc in docs]

    assert len(exptoks) == len(docs)

    if not is_spacydocs:
        if isinstance(next(iter(docs)), np.ndarray):
            return [np.array(d) if d else empty_chararray() for d in exptoks]
        else:
            return exptoks

    new_docs = []
    for doc_exptok, doc in zip(exptoks, docs):
        words = []
        tokens = []
        spaces = []
        lemmata = []
        for exptok, t, oldtok in zip(doc_exptok, doc, _filtered_doc_tokens(doc)):
            n_exptok = len(exptok)
            spaces.extend([''] * (n_exptok-1) + [t.whitespace_])

            if n_exptok > 1:
                lemmata.extend(exptok)
                words.extend(exptok)
                tokens.extend(exptok)
            else:
                lemmata.append(t.lemma_)
                words.append(t.text)
                tokens.append(oldtok)

        new_doc = Doc(doc.vocab, words=words, spaces=spaces)
        new_doc._.label = doc._.label

        assert len(new_doc) == len(lemmata)
        _init_doc(new_doc, tokens)
        for t, lem in zip(new_doc, lemmata):
            t.lemma_ = lem

        new_docs.append(new_doc)

    return new_docs


def transform(docs, func, **kwargs):
    """
    Apply `func` to each token in each document of `docs` and return the result.

    :param docs: list of string tokens or spaCy documents
    :param func: function to apply to each token; should accept a string as first arg. and optional `kwargs`
    :param kwargs: keyword arguments passed to `func`
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    if not callable(func):
        raise ValueError('`func` must be callable')

    is_spacydocs = require_spacydocs_or_tokens(docs)

    if is_spacydocs is None:
        return []

    is_arrays = not is_spacydocs and isinstance(next(iter(docs)), np.ndarray)

    if kwargs:
        func_wrapper = lambda t: func(t, **kwargs)
    else:
        func_wrapper = func

    if is_spacydocs:
        labels = doc_labels(docs)
        docs = doc_tokens(docs)

    res = [list(map(func_wrapper, dtok)) for dtok in docs]

    if is_spacydocs:
        return tokendocs2spacydocs(res, doc_labels=labels)
    elif is_arrays:
        return [np.array(d) if d else empty_chararray() for d in res]
    else:
        return res


def to_lowercase(docs):
    """
    Apply lowercase transformation to each document.

    :param docs: list of string tokens or spaCy documents
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    return transform(docs, str.lower)


def remove_chars(docs, chars):
    """
    Remove all characters listed in `chars` from all tokens.

    :param docs: list of string tokens or spaCy documents
    :param chars: list of characters to remove
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    if not chars:
        raise ValueError('`chars` must be a non-empty sequence')

    is_spacydocs = require_spacydocs_or_tokens(docs)

    if is_spacydocs is None:
        return []

    is_arrays = not is_spacydocs and isinstance(next(iter(docs)), np.ndarray)

    if is_spacydocs:
        labels = doc_labels(docs)
        docs = doc_tokens(docs)

    del_chars = str.maketrans('', '', ''.join(chars))
    res = [[t.translate(del_chars) for t in dtok] for dtok in docs]

    if is_spacydocs:
        return tokendocs2spacydocs(res, doc_labels=labels)
    elif is_arrays:
        return [np.array(d) if d else empty_chararray() for d in res]
    else:
        return res


def clean_tokens(docs, remove_punct=True, remove_stopwords=True, remove_empty=True,
                 remove_shorter_than=None, remove_longer_than=None, remove_numbers=False,
                 nlp_instance=None, language=None):
    """
    Apply several token cleaning steps to documents `docs` and optionally documents metadata `docs_meta`, depending on
    the given parameters.

    :param docs: list of string tokens or spaCy documents
    :param remove_punct: if True, remove all tokens marked as ``is_punct`` by spaCy if `docs` are spaCy documents,
                         otherwise remove tokens that match the characters listed in ``string.punctuation``;
                         if arg is a list, tuple or set, remove all tokens listed in this arg from the
                         documents; if False do not apply punctuation token removal
    :param remove_stopwords: if True, remove stop words for the given `language` as loaded via
                             `~tmtoolkit.preprocess.load_stopwords` ; if arg is a list, tuple or set, remove all tokens
                             listed in this arg from the documents; if False do not apply stop word token removal
    :param remove_empty: if True, remove empty strings ``""`` from documents
    :param remove_shorter_than: if given a positive number, remove tokens that are shorter than this number
    :param remove_longer_than: if given a positive number, remove tokens that are longer than this number
    :param remove_numbers: if True, remove all tokens that are deemed numeric by :func:`np.char.isnumeric`
    :param nlp_instance: spaCy nlp instance
    :param language: language for stop word removal
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    if remove_shorter_than is not None and remove_shorter_than < 0:
        raise ValueError('`remove_shorter_than` must be >= 0')
    if remove_longer_than is not None and remove_longer_than < 0:
        raise ValueError('`remove_longer_than` must be >= 0')

    is_spacydocs = require_spacydocs_or_tokens(docs)

    if is_spacydocs is None:
        return []

    is_arrays = not is_spacydocs and isinstance(next(iter(docs)), np.ndarray)

    # add empty token to list of tokens to remove
    tokens_to_remove = [''] if remove_empty else []

    # add punctuation characters to list of tokens to remove
    if isinstance(remove_punct, (tuple, list, set)):
        tokens_to_remove.extend(remove_punct)

    # add stopwords to list of tokens to remove
    if remove_stopwords is True:
        # default stopword list
        tokens_to_remove.extend(load_stopwords(language or _current_nlp(nlp_instance).lang))
    elif isinstance(remove_stopwords, (tuple, list, set)):
        tokens_to_remove.extend(remove_stopwords)

    # the "remove masks" list holds a binary array for each document where `True` signals a token to be removed
    docs_as_tokens = doc_tokens(docs)
    remove_masks = [np.repeat(False, len(doc)) for doc in docs_as_tokens]

    # update remove mask for punctuation
    if remove_punct is True:
        if is_spacydocs:
            remove_masks = [mask | doc.to_array('is_punct')[doc.user_data['mask']].astype(np.bool_)
                            for mask, doc in zip(remove_masks, docs)]
        else:
            tokens_to_remove.extend(list(string.punctuation))

    # update remove mask for tokens shorter/longer than a certain number of characters
    if remove_shorter_than is not None or remove_longer_than is not None:
        token_lengths = [np.fromiter(map(len, doc), np.int, len(doc)) for doc in docs_as_tokens]

        if remove_shorter_than is not None:
            remove_masks = [mask | (n < remove_shorter_than) for mask, n in zip(remove_masks, token_lengths)]

        if remove_longer_than is not None:
            remove_masks = [mask | (n > remove_longer_than) for mask, n in zip(remove_masks, token_lengths)]

    # update remove mask for numeric tokens
    if remove_numbers:
        if is_spacydocs:
            remove_masks = [mask | doc.to_array('like_num')[doc.user_data['mask']].astype(np.bool_)
                            for mask, doc in zip(remove_masks, docs)]
        elif is_arrays:
            remove_masks = [mask | np.char.isnumeric(doc)
                            for mask, doc in zip(remove_masks, docs_as_tokens)]
        else:
            remove_masks = [mask | np.array([t.isnumeric() for t in doc], dtype=np.bool_)
                            for mask, doc in zip(remove_masks, docs_as_tokens)]

    # update remove mask for general list of tokens to be removed
    if tokens_to_remove:
        tokens_to_remove = set(tokens_to_remove)
        # this is actually much faster than using np.isin:
        remove_masks = [mask | np.array([t in tokens_to_remove for t in doc], dtype=bool)
                        for mask, doc in zip(remove_masks, docs_as_tokens)]

    # apply the mask
    return _apply_matches_array(docs, remove_masks, invert=True)


def filter_tokens_by_mask(docs, mask, inverse=False):
    """
    Filter tokens in `docs` according to a binary mask specified by `mask`.

    :param docs: list of string tokens or spaCy documents
    :param mask: a list containing a mask list for each document in `docs`; each mask list contains boolean values for
                 each token in that document, where `True` means keeping that token and `False` means removing it;
    :param inverse: inverse the mask for filtering, i.e. keep all tokens with a mask set to `False` and remove all those
                    with `True`
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    require_spacydocs_or_tokens(docs)

    if len(mask) > 0 and not isinstance(mask[0], np.ndarray):
        mask = list(map(lambda x: np.array(x, dtype=np.bool), mask))

    return _apply_matches_array(docs, mask, invert=inverse)


def remove_tokens_by_mask(docs, mask):
    """
    Same as :func:`~tmtoolkit.preprocess.filter_tokens_by_mask` but with ``inverse=True``.

    .. seealso:: :func:`~tmtoolkit.preprocess.filter_tokens_by_mask`
    """
    return filter_tokens_by_mask(docs, mask, inverse=True)


def filter_tokens(docs, search_tokens, by_meta=None, match_type='exact', ignore_case=False,
                  glob_method='match', inverse=False):
    """
    Filter tokens in `docs` according to search pattern(s) `search_tokens` and several matching options. Only those
    tokens are retained that match the search criteria unless you set ``inverse=True``, which will *remove* all tokens
    that match the search criteria (which is the same as calling :func:`~tmtoolkit.preprocess.remove_tokens`).

    .. seealso:: :func:`~tmtoolkit.preprocess.remove_tokens`  and :func:`~tmtoolkit.preprocess.token_match`

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: typically a single string or non-empty list of strings that specify the search pattern(s);
                          when matching against meta data via `by_meta`, may also be of any other type
    :param by_meta: if not None, this should be a string of a token meta data attribute; this meta data will then be
                    used for matching instead of the tokens in `docs`
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    require_spacydocs_or_tokens(docs)

    matches = _token_pattern_matches(_match_against(docs, by_meta), search_tokens, match_type=match_type,
                                     ignore_case=ignore_case, glob_method=glob_method)

    return filter_tokens_by_mask(docs, matches, inverse=inverse)


def remove_tokens(docs, search_tokens, by_meta=None, match_type='exact', ignore_case=False, glob_method='match'):
    """
    Same as :func:`~tmtoolkit.preprocess.filter_tokens` but with ``inverse=True``.

    .. seealso:: :func:`~tmtoolkit.preprocess.filter_tokens` and :func:`~tmtoolkit.preprocess.token_match`.
    """
    return filter_tokens(docs, search_tokens=search_tokens, by_meta=by_meta, match_type=match_type,
                         ignore_case=ignore_case, glob_method=glob_method, inverse=True)


def filter_tokens_with_kwic(docs, search_tokens, context_size=2, match_type='exact', ignore_case=False,
                            glob_method='match', inverse=False):
    """
    Filter tokens in `docs` according to Keywords-in-Context (KWIC) context window of size `context_size` around
    `search_tokens`. Works similar to :func:`~tmtoolkit.preprocess.kwic`, but returns result as list of tokenized
    documents, i.e. in the same structure as `docs` whereas :func:`~tmtoolkit.preprocess.kwic` returns result as
    list of *KWIC windows* into `docs`.

    .. seealso:: :func:`~tmtoolkit.preprocess.kwic`

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: typically a single string or non-empty list of strings that specify the search pattern(s);
                          when matching against meta data via `by_meta`, may also be of any other type
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    require_spacydocs_or_tokens(docs)

    if isinstance(context_size, int):
        context_size = (context_size, context_size)
    else:
        require_listlike(context_size)

    matches = _build_kwic(docs, search_tokens,
                          context_size=context_size,
                          match_type=match_type,
                          ignore_case=ignore_case,
                          glob_method=glob_method,
                          inverse=inverse,
                          only_token_masks=True)

    return filter_tokens_by_mask(docs, matches)


def remove_tokens_by_doc_frequency(docs, which, df_threshold, absolute=False, return_blacklist=False,
                                   return_mask=False):
    """
    Remove tokens according to their document frequency.

    :param docs: list of string tokens or spaCy documents
    :param which: which threshold comparison to use: either ``'common'``, ``'>'``, ``'>='`` which means that tokens
                  with higher document freq. than (or equal to) `df_threshold` will be removed;
                  or ``'uncommon'``, ``'<'``, ``'<='`` which means that tokens with lower document freq. than
                  (or equal to) `df_threshold` will be removed
    :param df_threshold: document frequency threshold value
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``; POS tags must exist for all documents in `docs_meta`
                      (``"meta_pos"`` key)
    :param return_blacklist: if True return a list of tokens that should be removed instead of the filtered tokens
    :param return_mask: if True return a list of token masks where each occurrence of True signals a token to
                        be removed
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    require_spacydocs_or_tokens(docs)

    which_opts = {'common', '>', '>=', 'uncommon', '<', '<='}

    if which not in which_opts:
        raise ValueError('`which` must be one of: %s' % ', '.join(which_opts))

    n_docs = len(docs)

    if absolute:
        if not 0 <= df_threshold <= n_docs:
            raise ValueError('`df_threshold` must be in range [0, %d]' % n_docs)
    else:
        if not 0 <= df_threshold <= 1:
            raise ValueError('`df_threshold` must be in range [0, 1]')

    if which in ('common', '>='):
        comp = operator.ge
    elif which == '>':
        comp = operator.gt
    elif which == '<':
        comp = operator.lt
    else:
        comp = operator.le

    toks = doc_tokens(docs, to_lists=True)
    doc_freqs = doc_frequencies(toks, proportions=not absolute)
    mask = [[comp(doc_freqs[t], df_threshold) for t in dtok] for dtok in toks]

    if return_blacklist:
        blacklist = set(t for t, f in doc_freqs.items() if comp(f, df_threshold))
        if return_mask:
            return blacklist, mask

    if return_mask:
        return mask

    return remove_tokens_by_mask(docs, mask)


def remove_common_tokens(docs, df_threshold=0.95, absolute=False):
    """
    Shortcut for :func:`~tmtoolkit.preprocess.remove_tokens_by_doc_frequency` for removing tokens *above* a certain
    document frequency.

    :param docs: list of string tokens or spaCy documents
    :param df_threshold: document frequency threshold value
    :param absolute: if True, use absolute document frequency (i.e. number of times token X occurs at least once
                 in a document), otherwise use relative document frequency (normalized by number of documents)
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    return remove_tokens_by_doc_frequency(docs, 'common', df_threshold=df_threshold, absolute=absolute)


def remove_uncommon_tokens(docs, df_threshold=0.05, absolute=False):
    """
    Shortcut for :func:`~tmtoolkit.preprocess.remove_tokens_by_doc_frequency` for removing tokens *below* a certain
    document frequency.

    :param docs: list of string tokens or spaCy documents
    :param df_threshold: document frequency threshold value
    :param absolute: if True, use absolute document frequency (i.e. number of times token X occurs at least once
                 in a document), otherwise use relative document frequency (normalized by number of documents)
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    return remove_tokens_by_doc_frequency(docs, 'uncommon', df_threshold=df_threshold, absolute=absolute)


def filter_documents(docs, search_tokens, by_meta=None, matches_threshold=1,
                     match_type='exact', ignore_case=False, glob_method='match', inverse_result=False,
                     inverse_matches=False):
    """
    This function is similar to :func:`~tmtoolkit.preprocess.filter_tokens` but applies at document level. For each
    document, the number of matches is counted. If it is at least `matches_threshold` the document is retained,
    otherwise removed. If `inverse_result` is True, then documents that meet the threshold are *removed*.

    .. seealso:: :func:`~tmtoolkit.preprocess.remove_documents`

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: typically a single string or non-empty list of strings that specify the search pattern(s);
                          when matching against meta data via `by_meta`, may also be of any other type
    :param by_meta: if not None, this should be a string of a token meta data attribute; this meta data will then be
                    used for matching instead of the tokens in `docs`
    :param matches_threshold: the minimum number of matches required per document
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse_result: inverse the threshold comparison result
    :param inverse_matches: inverse the match results for filtering
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    require_spacydocs_or_tokens(docs)

    matches = _token_pattern_matches(_match_against(docs, by_meta), search_tokens, match_type=match_type,
                                     ignore_case=ignore_case, glob_method=glob_method)

    if inverse_matches:
        matches = [~m for m in matches]

    new_docs = []
    for i, (doc, n_matches) in enumerate(zip(docs, map(np.sum, matches))):
        thresh_met = n_matches >= matches_threshold
        if inverse_result:
            thresh_met = not thresh_met
        if thresh_met:
            new_docs.append(doc)

    return new_docs


def remove_documents(docs, search_tokens, by_meta=None, matches_threshold=1,
                     match_type='exact', ignore_case=False, glob_method='match', inverse_matches=False):
    """
    Same as :func:`~tmtoolkit.preprocess.filter_documents` but with ``inverse=True``.

    .. seealso:: :func:`~tmtoolkit.preprocess.filter_documents`
    """
    return filter_documents(docs, search_tokens, by_meta=by_meta,
                            matches_threshold=matches_threshold, match_type=match_type, ignore_case=ignore_case,
                            glob_method=glob_method, inverse_matches=inverse_matches, inverse_result=True)


def filter_documents_by_name(docs, name_patterns, labels=None, match_type='exact', ignore_case=False,
                             glob_method='match', inverse=False):
    """
    Filter documents by their name (i.e. document label). Keep all documents whose name matches `name_pattern`
    according to additional matching options. If `inverse` is True, drop all those documents whose name matches,
    which is the same as calling :func:`~tmtoolkit.preprocess.remove_documents_by_name`.

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: typically a single string or non-empty list of strings that specify the search pattern(s);
                          when matching against meta data via `by_meta`, may also be of any other type
    :param labels: if `docs` is not a list of spaCy documents, you must pass the document labels as list of strings
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: invert the matching results
    :return: list of string tokens or spaCy documents, depending on `docs`
    """
    is_spacydocs = require_spacydocs_or_tokens(docs)

    if is_spacydocs is None:
        return []

    if isinstance(name_patterns, str):
        name_patterns = [name_patterns]
    else:
        require_listlike(name_patterns)

        if not name_patterns:
            raise ValueError('`name_patterns` must not be empty')

    if is_spacydocs and labels is None:
        labels = doc_labels(docs)
    elif not is_spacydocs and labels is None:
        raise ValueError('if not passing a list of spaCy documents as `docs`, you must pass document labels via '
                         '`labels`')

    if len(labels) != len(docs):
        raise ValueError('number of document labels must match number of documents')

    matches = None

    for pat in name_patterns:
        pat_match = token_match(pat, labels, match_type=match_type, ignore_case=ignore_case,
                                glob_method=glob_method)

        if matches is None:
            matches = pat_match
        else:
            matches |= pat_match

    assert matches is not None
    assert len(labels) == len(matches)

    if inverse:
        matches = ~matches

    return [doc for doc, m in zip(docs, matches) if m]


def remove_documents_by_name(docs, name_patterns, labels=None, match_type='exact', ignore_case=False,
                             glob_method='match'):
    """
    Same as :func:`~tmtoolkit.preprocess.filter_documents_by_name` but with ``inverse=True``.

    .. seealso:: :func:`~tmtoolkit.preprocess.filter_documents_by_name`
    """

    return filter_documents_by_name(docs, name_patterns, labels=labels, match_type=match_type,
                                    ignore_case=ignore_case, glob_method=glob_method, inverse=True)


#%% functions that operate *only* on lists of spacy documents

def doc_labels(docs):
    """
    Return list of document labels that are assigned to spaCy documents `docs`.

    :param docs: list of spaCy documents
    :return: list of document labels
    """
    require_spacydocs(docs)

    return [d._.label for d in docs]


def compact_documents(docs):
    """
    Compact documents `docs` by recreating new documents using the previously applied filters.

    :param docs: list of spaCy documents
    :return: list with compact spaCy documents
    """
    require_spacydocs(docs)

    return _apply_matches_array(docs, compact=True)


def pos_tag(docs, tagger=None, nlp_instance=None):
    """
    Apply Part-of-Speech (POS) tagging to all documents.

    The meanings of the POS tags are described in the
    `spaCy documentation <https://spacy.io/api/annotation#pos-tagging>`_.

    .. note:: This function only applies POS tagging to the documents but doesn't retrieve the tags. If you want to
              retrieve the tags, you may use :func:`~tmtoolkit.preprocess.pos_tags`.

    .. note:: This function modifies the documents in `docs` in place and adds/modifies a `pos_` attribute in each
              token.


    :param docs: list of spaCy documents
    :param tagger: POS tagger instance to use; by default, use the tagger for the currently loaded spaCy nlp instance
    :param nlp_instance: spaCy nlp instance
    :return: input spaCy documents `docs` with in-place modified documents
    """
    require_spacydocs(docs)

    tagger = tagger or _current_nlp(nlp_instance, pipeline_component='tagger')

    for doc in docs:
        # this will be done for all tokens in the document, also for masked tokens
        # unless "compact" is called before
        tagger(doc)

    return docs


def pos_tags(docs, tag_attrib='pos_', tagger=None, nlp_instance=None):
    """
    Return Part-of-Speech (POS) tags of `docs`. If POS tagging was not applied to `docs` yet, this function
    runs :func:`~tmtoolkit.preprocess.pos_tag` first.

    :param docs: list of spaCy documents
    :param tag_attrib: spaCy document tag attribute to fetch the POS tag; ``"pos_"`` and ``"pos"`` give coarse POS
                       tags as string or integer tags respectively, ``"tag_"`` and ``"tag"`` give fine grained POS
                       tags as string or integer tags
    :param tagger: POS tagger instance to use; by default, use the tagger for the currently loaded spaCy nlp instance
    :param nlp_instance: spaCy nlp instance
    :return: POS tags of `docs` as list of strings or integers depending on `tag_attrib`
    """
    require_spacydocs(docs)

    if tag_attrib not in {'pos', 'pos_', 'tag', 'tag_'}:
        raise ValueError("`tag_attrib` must be 'pos', 'pos_', 'tag' or 'tag_'")

    if not docs:
        return []

    first_doc = next(iter(docs))
    if not getattr(first_doc, tag_attrib, False):
        pos_tag(docs, tagger=tagger, nlp_instance=nlp_instance)

    return _get_docs_tokenattrs(docs, tag_attrib, custom_attr=False)


def filter_for_pos(docs, required_pos, simplify_pos=True, pos_attrib='pos_', tagset='ud', inverse=False):
    """
    Filter tokens for a specific POS tag (if `required_pos` is a string) or several POS tags (if `required_pos`
    is a list/tuple/set of strings).  The POS tag depends on the tagset used during tagging. See
    https://spacy.io/api/annotation#pos-tagging for a general overview on POS tags in SpaCy and refer to the
    documentation of your language model for specific tags.

    If `simplify_pos` is True, then the tags are matched to the following simplified forms:

    * ``'N'`` for nouns
    * ``'V'`` for verbs
    * ``'ADJ'`` for adjectives
    * ``'ADV'`` for adverbs
    * ``None`` for all other

    :param docs: list of spaCy documents
    :param required_pos: single string or list of strings with POS tag(s) used for filtering
    :param simplify_pos: before matching simplify POS tags in documents to forms shown above
    :param pos_attrib: token attribute name for POS tags
    :param tagset: POS tagset used while tagging; necessary for simplifying POS tags when `simplify_pos` is True
    :param inverse: inverse the matching results, i.e. *remove* tokens that match the POS tag
    :return: filtered list of spaCy documents
    """
    require_spacydocs(docs)

    docs_pos = _get_docs_tokenattrs(docs, pos_attrib, custom_attr=False)

    if required_pos is None or not isinstance(required_pos, (tuple, list)):
        required_pos = [required_pos]

    if simplify_pos:
        simplify_fn = np.vectorize(lambda x: simplified_pos(x, tagset=tagset))
    else:
        simplify_fn = np.vectorize(lambda x: x)  # identity function

    matches = [np.isin(simplify_fn(dpos), required_pos)
               if len(dpos) > 0
               else np.array([], dtype=bool)
               for dpos in docs_pos]

    return _apply_matches_array(docs, matches, invert=inverse)


def lemmatize(docs, lemma_attrib='lemma_'):
    """
    Lemmatize documents according to `language` or use a custom lemmatizer function `lemmatizer_fn`.

    :param docs: list of spaCy documents
    :param tag_attrib: spaCy document tag attribute to fetch the lemmata; ``"lemma_"`` gives lemmata as strings,
                       ``"lemma"`` gives lemmata as integer token IDs
    :return: list of string lists with lemmata for each document
    """
    require_spacydocs(docs)

    docs_lemmata = _get_docs_tokenattrs(docs, lemma_attrib, custom_attr=False)

    # SpaCy lemmata sometimes contain special markers like -PRON- instead of the lemma;
    # fix this here by resorting to the original token
    toks = doc_tokens(docs, to_lists=True)
    new_docs_lemmata = []
    assert len(docs_lemmata) == len(toks)
    for doc_tok, doc_lem in zip(toks, docs_lemmata):
        assert len(doc_tok) == len(doc_lem)
        new_docs_lemmata.append([t if l.startswith('-') and l.endswith('-') else l
                                 for t, l in zip(doc_tok, doc_lem)])

    return new_docs_lemmata


def tokens2ids(docs):
    """
    Convert a list of spaCy documents `docs` to a list of numeric token ID arrays. The IDs correspond to the current
    spaCy vocabulary.

    .. seealso:: :func:`~tmtoolkit.preprocess.ids2tokens` which reverses this operation.

    :param docs: list of spaCy documents
    :return: list of token ID arrays
    """
    require_spacydocs(docs)

    return [d.to_array('ORTH') for d in docs]


def ids2tokens(vocab, tokids):
    """
    Convert list of numeric token ID arrays `tokids` to a character token array with the help of the spaCy vocabulary
    `vocab`. Returns result as list of spaCy documents.

    .. seealso:: :func:`~tmtoolkit.preprocess.tokens2ids` which reverses this operation.

    :param vocab: spaCy vocabulary
    :param tokids: list of numeric token ID arrays as from :func:`~tmtoolkit.preprocess.tokens2ids`
    :return: list of spaCy documents
    """
    return [Doc(vocab, words=[vocab[t].orth_ for t in ids]) for ids in tokids]


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
        vocab = Vocab(strings=list(vocabulary([tokens])))
    elif not isinstance(vocab, Vocab):
        vocab = Vocab(strings=vocab.tolist() if isinstance(vocab, np.ndarray) else list(vocab))

    if lemmata is not None and len(lemmata) != len(tokens):
        raise ValueError('`lemmata` must have the same length as `tokens`')

    new_doc = Doc(vocab, words=tokens, spaces=spaces)
    assert len(new_doc) == len(tokens)

    if label is not None:
        new_doc._.label = label

    _init_doc(new_doc, tokens)

    if lemmata is not None:
        lemmata = lemmata.tolist() if isinstance(lemmata, np.ndarray) else lemmata
        for t, lem in zip(new_doc, lemmata):
            t.lemma_ = lem

    return new_doc


def doc_glue_subsequent(doc, matches, glue='_', return_glued=False):
    """
    Select subsequent tokens in `doc` as defined by list of indices `matches` (e.g. output of
    :func:`~tmtoolkit.preprocess.token_match_subsequent`) and join those by string `glue`. Return `doc` again.

    .. note:: This function modifies `doc` in place. All token attributes will be reset to default values besides
              ``"lemma_"`` which are set to the joint token string.

    .. warning:: Only works correctly when matches contains indices of *subsequent* tokens.

    Example::

        # doc is a SpaCy document with tokens ['a', 'b', 'c', 'd', 'd', 'a', 'b', 'c']
        token_glue_subsequent(doc, [np.array([1, 2]), np.array([6, 7])])
        # doc now contains tokens ['a', 'b_c', 'd', 'd', 'a', 'b_c']

    .. seealso:: :func:`~tmtoolkit.preprocess.token_match_subsequent`

    :param doc: a SpaCy document which must be compact (i.e. no filter mask set)
    :param matches: list of NumPy arrays with *subsequent* indices into `tokens` (e.g. output of
                    :func:`~tmtoolkit.preprocess.token_match_subsequent`)
    :param glue: string for joining the subsequent matches or None if no joint tokens but an empty string should be set
                 as token value and lemma
    :param return_glued: if yes, return also a list of joint tokens
    :return: either two-tuple or input `doc`; if `return_glued` is True, return a two-tuple with 1) `doc` and 2) a list
             of joint tokens; if `return_glued` is True only return 1)
    """
    require_listlike(matches)

    if return_glued and glue is None:
        raise ValueError('if `glue` is None, `return_glued` must be False')

    if not doc.user_data['mask'].all():
        raise ValueError('`doc` must be compact, i.e. no filter mask should be set (all elements in '
                         '`doc.user_data["mask"]` must be True)')

    n_tok = len(doc)

    if n_tok == 0:
        if return_glued:
            return doc, []
        else:
            return doc

    # map span start index to end index
    glued = []

    # within this context, `doc` doesn't change (i.e. we can use the same indices into `doc` throughout the for loop
    # even when we merge tokens
    del_tokens_indices = []
    with doc.retokenize() as retok:
        # we will need to update doc.user_data['tokens'], which is a NumPy character array;
        # a NumPy char array has a maximum element size and we will need to update that to the
        # maximum string length in `chararray_updates` by using `widen_chararray()` below
        chararray_updates = {}
        for m in matches:
            assert len(m) >= 2
            begin, end = m[0], m[-1]
            span = doc[begin:end+1]
            merged = '' if glue is None else glue.join(doc.user_data['tokens'][begin:end+1])
            assert begin not in chararray_updates.keys()
            chararray_updates[begin] = merged
            del_tokens_indices.extend(list(range(begin+1, end+1)))
            attrs = {
                'LEMMA': merged,
                'WHITESPACE': doc[end].whitespace_
            }
            retok.merge(span, attrs=attrs)

            if return_glued:
                glued.append(merged)

        if chararray_updates:
            new_maxsize = max(map(len, chararray_updates.values()))
            doc.user_data['tokens'] = widen_chararray(doc.user_data['tokens'], new_maxsize)
            for begin, merged in chararray_updates.items():
                doc.user_data['tokens'][begin] = merged

    doc.user_data['tokens'] = np.delete(doc.user_data['tokens'], del_tokens_indices)
    doc.user_data['mask'] = np.delete(doc.user_data['mask'], del_tokens_indices)

    if return_glued:
        return doc, glued
    else:
        return doc



#%% helper functions


def _current_nlp(nlp_instance, pipeline_component=None):
    _nlp = nlp_instance or nlp
    if not _nlp:
        raise ValueError('neither global nlp instance is set, nor `nlp_instance` argument is given; did you call '
                         '`init_for_language()` before?')

    if pipeline_component:
        return dict(nlp.pipeline)[pipeline_component]
    else:
        return _nlp


def _init_doc(doc, tokens=None, mask=None):
    assert isinstance(doc, Doc)

    if tokens is None:
        tokens = [t.text for t in doc]
    else:
        assert isinstance(tokens, (list, tuple, np.ndarray))
        assert len(doc) == len(tokens)

    if mask is not None:
        assert isinstance(mask, np.ndarray)
        assert np.issubdtype(mask.dtype, np.bool_)
        assert len(doc) == len(mask)

    if len(tokens) == 0:
        doc.user_data['tokens'] = empty_chararray()
    else:
        doc.user_data['tokens'] = np.array(tokens) if not isinstance(tokens, np.ndarray) else tokens
    doc.user_data['mask'] = mask if isinstance(mask, np.ndarray) else np.repeat(True, len(doc))


def _get_docs_tokenattrs_keys(docs, default_attrs=None):
    if default_attrs:
        attrs = default_attrs
    else:
        attrs = ['lemma_', 'whitespace_']

    if len(docs) > 0:
        for doc in docs:
            if doc.is_tagged and 'pos_' not in attrs:
                attrs.append('pos_')

            if len(doc) > 0:
                firsttok = next(iter(doc))
                attrs.extend([attr for attr in dir(firsttok._)
                             if attr not in {'has', 'get', 'set'} and not attr.startswith('_')])
                break

    return attrs


def _filtered_doc_arr(lst, doc):
    return np.array(lst)[doc.user_data['mask']]


def _filtered_doc_tokens(doc, as_list=False):
    if isinstance(doc, Doc):
        res = doc.user_data['tokens'][doc.user_data['mask']]
        return res.tolist() if as_list else res
    else:
        assert isinstance(doc, (list, np.ndarray))
        if isinstance(doc, np.ndarray) and as_list:
            return doc.tolist()
        else:
            return doc


def _ngrams_from_tokens(tokens, n, join=True, join_str=' '):
    """
    Helper function to produce ngrams of length `n` from a list of string tokens `tokens`.

    :param tokens: list of string tokens
    :param n: size of ngrams
    :param join: if True, join each ngram by `join_str`, i.e. return list of ngram strings; otherwise return list of
                 ngram lists
    :param join_str: if `join` is True, use this string to join the parts of the ngrams
    :return: return list of ngram strings if `join` is True, otherwise list of ngram lists
    """
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


def _match_against(docs, by_meta=None):
    """Return the list of values to match against in filtering functions."""
    if by_meta:
        require_spacydocs(docs)
        return [_filtered_doc_arr([_get_spacytoken_attr(t, by_meta) for t in doc], doc) for doc in docs]
    else:
        return [_filtered_doc_tokens(doc) for doc in docs]


def _token_pattern_matches(tokens, search_tokens, match_type, ignore_case, glob_method):
    """
    Helper function to apply `token_match` with multiple patterns in `search_tokens` to `docs`.
    The matching results for each pattern in `search_tokens` are combined via logical OR.
    Returns a list of length `docs` containing boolean arrays that signal the pattern matches for each token in each
    document.
    """

    # search tokens may be of any type (e.g. bool when matching against token meta data)
    if not isinstance(search_tokens, (list, tuple, set)):
        search_tokens = [search_tokens]
    elif isinstance(search_tokens, (list, tuple, set)) and not search_tokens:
        raise ValueError('`search_tokens` must not be empty')

    matches = [np.repeat(False, repeats=len(dtok)) for dtok in tokens]

    for dtok, dmatches in zip(tokens, matches):
        for pat in search_tokens:
            pat_match = token_match(pat, dtok, match_type=match_type, ignore_case=ignore_case, glob_method=glob_method)

            dmatches |= pat_match

    return matches


def _build_kwic(docs, search_tokens, context_size, match_type, ignore_case, glob_method, inverse,
                highlight_keyword=None, with_metadata=False, with_window_indices=False,
                only_token_masks=False):
    """
    Helper function to build keywords-in-context (KWIC) results from documents `docs`.

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: search pattern(s)
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`).
    :param inverse: Invert the matching results.
    :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                              matched keyword.
    :param with_metadata: add document metadata to KWIC results
    :param with_window_indices: add window indices to KWIC results
    :param only_token_masks: return only flattened token masks for filtering
    :return: list with KWIC results per document
    """

    # find matches for search criteria -> list of NumPy boolean mask arrays
    matches = _token_pattern_matches(_match_against(docs), search_tokens, match_type=match_type,
                                     ignore_case=ignore_case, glob_method=glob_method)

    if not only_token_masks and inverse:
        matches = [~m for m in matches]

    left, right = context_size

    kwic_list = []
    for mask, doc in zip(matches, docs):
        ind = np.where(mask)[0]
        ind_windows = make_index_window_around_matches(mask, left, right,
                                                       flatten=only_token_masks, remove_overlaps=True)

        if only_token_masks:
            assert ind_windows.ndim == 1
            assert len(ind) <= len(ind_windows)

            # from indices back to binary mask; this only works with remove_overlaps=True
            win_mask = np.repeat(False, len(doc))
            win_mask[ind_windows] = True

            if inverse:
                win_mask = ~win_mask

            kwic_list.append(win_mask)
        else:
            doc_arr = _filtered_doc_tokens(doc)
            if not isinstance(doc_arr, np.ndarray):
                assert isinstance(doc_arr, list)
                doc_arr = np.array(doc_arr) if doc_arr else empty_chararray()

            assert len(ind) == len(ind_windows)

            windows_in_doc = []
            for match_ind, win in zip(ind, ind_windows):  # win is an array of indices into dtok_arr
                tok_win = doc_arr[win].tolist()

                if highlight_keyword is not None:
                    highlight_mask = win == match_ind
                    assert np.sum(highlight_mask) == 1
                    highlight_ind = np.where(highlight_mask)[0][0]
                    tok_win[highlight_ind] = highlight_keyword + tok_win[highlight_ind] + highlight_keyword

                win_res = {'token': tok_win}

                if with_window_indices:
                    win_res['index'] = win

                if with_metadata:
                    attr_keys = _get_docs_tokenattrs_keys(docs)
                    attr_vals = {}
                    attr_keys_base = []
                    attr_keys_ext = []

                    for attr in attr_keys:
                        if attr.endswith('_'):
                            attrlabel = attr[:-1]   # remove trailing underscore
                            attr_keys_base.append(attrlabel)
                        else:
                            attrlabel = attr
                            attr_keys_ext.append(attrlabel)

                        if attr == 'whitespace_':
                            attrdata = [bool(t.whitespace_) for t in doc]
                        else:
                            attrdata = [_get_spacytoken_attr(t, attr) for t in doc]

                        attr_vals[attrlabel] = np.array(attrdata)[win].tolist()

                    for attrlabel in list(sorted(attr_keys_base)) + list(sorted(attr_keys_ext)):
                        win_res[attrlabel] = attr_vals[attrlabel]

                windows_in_doc.append(win_res)

            kwic_list.append(windows_in_doc)

    assert len(kwic_list) == len(docs)

    return kwic_list


def _finalize_kwic_results(kwic_results, non_empty, glue, as_datatable, with_metadata):
    """
    Helper function to finalize raw KWIC results coming from `_build_kwic()`: Filter results, "glue" (join) tokens,
    transform to datatable, return or dismiss metadata.
    """
    kwic_results_ind = None

    if non_empty:
        if isinstance(kwic_results, dict):
            kwic_results = {dl: windows for dl, windows in kwic_results.items() if len(windows) > 0}
        else:
            assert isinstance(kwic_results, (list, tuple))
            kwic_results_w_indices = [(i, windows) for i, windows in enumerate(kwic_results) if len(windows) > 0]
            if kwic_results_w_indices:
                kwic_results_ind, kwic_results = zip(*kwic_results_w_indices)
            else:
                kwic_results_ind = []
                kwic_results = []

    if glue is not None:
        if isinstance(kwic_results, dict):
            return {dl: [glue.join(win['token']) for win in windows] for dl, windows in kwic_results.items()}
        else:
            assert isinstance(kwic_results, (list, tuple))
            return [[glue.join(win['token']) for win in windows] for windows in kwic_results]
    elif as_datatable:
        dfs = []
        if not kwic_results_ind:
            kwic_results_ind = range(len(kwic_results))

        for i_doc, dl_or_win in zip(kwic_results_ind, kwic_results):
            if isinstance(kwic_results, dict):
                dl = dl_or_win
                windows = kwic_results[dl]
            else:
                dl = i_doc
                windows = dl_or_win

            for i_win, win in enumerate(windows):
                if isinstance(win, list):
                    win = {'token': win}

                n_tok = len(win['token'])
                df_windata = [np.repeat(dl, n_tok),
                              np.repeat(i_win, n_tok),
                              win['index'],
                              win['token']]

                if with_metadata:
                    meta_cols = [col for col in win.keys() if col not in {'token', 'index'}]
                    df_windata.extend([win[col] for col in meta_cols])
                else:
                    meta_cols = []

                df_cols = ['doc', 'context', 'position', 'token'] + meta_cols
                dfs.append(pd_dt_frame(OrderedDict(zip(df_cols, df_windata))))

        if dfs:
            kwic_df = pd_dt_concat(dfs)
            return pd_dt_sort(kwic_df, ('doc', 'context', 'position'))
        else:
            return pd_dt_frame(OrderedDict(zip(['doc', 'context', 'position', 'token'], [[] for _ in range(4)])))
    elif not with_metadata:
        if isinstance(kwic_results, dict):
            return {dl: [win['token'] for win in windows]
                    for dl, windows in kwic_results.items()}
        else:
            return [[win['token'] for win in windows] for windows in kwic_results]
    else:
        return kwic_results


def _datatable_from_kwic_results(kwic_results):
    """
    Helper function to transform raw KWIC results coming from `_build_kwic()` to a datatable for `kwic_table()`.
    """
    dfs = []

    for i_doc, dl_or_win in enumerate(kwic_results):
        if isinstance(kwic_results, dict):
            dl = dl_or_win
            windows = kwic_results[dl]
        else:
            dl = i_doc
            windows = dl_or_win

        dfs.append(pd_dt_frame(OrderedDict(zip(['doc', 'context', 'kwic'],
                                               [np.repeat(dl, len(windows)), np.arange(len(windows)), windows]))))
    if dfs:
        kwic_df = pd_dt_concat(dfs)
        return pd_dt_sort(kwic_df, ('doc', 'context'))
    else:
        return pd_dt_frame(OrderedDict(zip(['doc', 'context', 'kwic'], [[] for _ in range(3)])))


def _apply_matches_array(docs, matches=None, invert=False, compact=False):
    """
    Helper function to apply a list of boolean arrays `matches` that signal token to pattern matches to a list of
    tokenized documents `docs`. If `compact` is False, simply set the new filter mask to previously unfiltered elements,
    which changes document masks in-place. If `compact` is True, create new Doc objects from filtered data *if there
    are any filtered tokens*, otherwise return the same unchanged Doc object.
    """
    is_spacydocs = require_spacydocs_or_tokens(docs)

    if is_spacydocs is None:
        return []

    if matches is None and is_spacydocs:
        matches = [doc.user_data['mask'] for doc in docs]

    if invert:
        matches = [~m for m in matches]

    assert len(matches) == len(docs)

    if is_spacydocs:
        if compact:   # create new Doc objects from filtered data
            more_attrs = _get_docs_tokenattrs_keys(docs, default_attrs=['lemma_'])
            new_docs = []

            for mask, doc in zip(matches, docs):
                assert len(mask) == len(doc) == len(doc.user_data['tokens'])

                if mask.all():
                    new_doc = doc
                else:
                    filtered = [(t, tt) for m, t, tt in zip(mask, doc, doc.user_data['tokens']) if m]

                    if filtered:
                        new_doc_text, new_doc_spaces = list(zip(*[(t.text, t.whitespace_) for t, _ in filtered]))
                    else:  # empty doc.
                        new_doc_text = []
                        new_doc_spaces = []

                    new_doc = Doc(doc.vocab, words=new_doc_text, spaces=new_doc_spaces)
                    new_doc._.label = doc._.label

                    _init_doc(new_doc, list(zip(*filtered))[1] if filtered else empty_chararray())

                    for attr in more_attrs:
                        if attr.endswith('_'):
                            attrname = attr[:-1]
                            vals = doc.to_array(attrname)   # without trailing underscore
                            new_doc.from_array(attrname, vals[mask])
                        else:
                            for v, nt in zip((getattr(t._, attr) for t, _ in filtered), new_doc):
                                setattr(nt._, attr, v)

                new_docs.append(new_doc)

            assert len(docs) == len(new_docs)

            return new_docs
        else:   # simply set the new filter mask to previously unfiltered elements; changes document masks in-place
            for mask, doc in zip(matches, docs):
                assert len(mask) == sum(doc.user_data['mask'])
                doc.user_data['mask'][doc.user_data['mask']] = mask

            return docs
    else:
        return [dtok[mask] if isinstance(dtok, np.ndarray) else np.array(dtok)[mask].tolist()
                for mask, dtok in zip(matches, docs)]


def _replace_doc_tokens(doc, new_tok):
    if isinstance(doc, (list, np.ndarray)):
        return new_tok
    else:
        assert isinstance(doc, Doc)
        # replace all non-filtered tokens
        assert sum(doc.user_data['mask']) == len(new_tok)
        doc.user_data['tokens'][doc.user_data['mask']] = new_tok
        return doc


def _get_docs_tokenattrs(docs, attr_name, custom_attr=True):
    return [[getattr(t._, attr_name) if custom_attr else getattr(t, attr_name)
             for t, m in zip(doc, doc.user_data['mask']) if m]   # apply filter
            for doc in docs]


def _get_spacytoken_attr(t, attr):
    return getattr(t._, attr, getattr(t, attr, None))


def require_spacydocs(docs, types=(Doc, ), error_msg='the argument must be a list of spaCy documents'):
    require_listlike(docs)

    if docs:
        first_doc = next(iter(docs))
        if not isinstance(first_doc, types):
            raise ValueError(error_msg)

        return first_doc

    return None


def require_spacydocs_or_tokens(docs):
    first_doc = require_spacydocs(docs, (Doc, list, np.ndarray), error_msg='the argument must be a list of string '
                                                                           'token documents or spaCy documents')
    return None if first_doc is None else isinstance(first_doc, Doc)
