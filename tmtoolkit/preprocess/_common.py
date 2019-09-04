"""
Common functions for text processing.

Most functions of this internal module are exported in ``__init__.py`` and make up the functional text processing API of
tmtoolkit.

Markus Konrad <markus.konrad@wzb.eu>
"""

import re
import os
import string
import operator
from collections import Counter, OrderedDict
from importlib import import_module
from functools import partial

import globre
import numpy as np
import datatable as dt
import nltk

from .. import defaults
from ..bow.dtm import create_sparse_dtm
from ..utils import flatten_list, require_listlike, empty_chararray, require_listlike_or_set, unpickle_file


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.normpath(os.path.join(MODULE_PATH, '..', 'data'))
POS_TAGGER_PICKLE = 'pos_tagger.pickle'

PATTERN_SUBMODULES = {
    'english': 'en',
    'german': 'de',
    'spanish': 'es',
    'french': 'fr',
    'italian': 'it',
    'dutch': 'nl',
}


#%% functions that operate on lists of documents


def tokenize(docs, language=defaults.language):
    """
    Tokenize a list of documents `docs` containing, each containing the raw text as string.

    Uses NLTK's ``word_tokenize()`` function for tokenization optimized for `language`.

    :param docs: list of documents: raw text strings
    :param language: language in which `docs` is given
    :return: list of tokenized documents (list of lists)
    """
    require_listlike(docs)

    return [nltk.tokenize.word_tokenize(text, language) for text in docs]


def doc_lengths(docs):
    """
    Return document length (number of tokens in doc.) for each document.

    :param docs: list of tokenized documents
    :return: list of document lengths
    """
    require_listlike(docs)

    return list(map(len, docs))


def vocabulary(docs, sort=False):
    """
    Return vocabulary, i.e. set of all tokens that occur across all documents.

    :param docs: list of tokenized documents
    :param sort: return as sorted list
    :return: either set of token strings or sorted list if `sort` is True
    """
    require_listlike(docs)

    v = set(flatten_list(docs))

    if sort:
        return sorted(v)
    else:
        return v


def vocabulary_counts(docs):
    """
    Return :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
    all documents.

    :param docs: list of tokenized documents
    :return: :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
             all documents
    """
    require_listlike(docs)

    return Counter(flatten_list(docs))


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

    :param docs: list of tokenized documents
    :param proportions: if True, normalize by number of documents to obtain proportions
    :return: dict mapping token to document frequency
    """
    require_listlike(docs)

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
    """
    Generate and return n-grams of length `n`.

    :param docs: list of tokenized documents
    :param n: length of n-grams, must be >= 2
    :param join: if True, join generated n-grams by string `join_str`
    :param join_str: string used for joining
    :return: list of n-grams; if `join` is True, the list contains strings of joined n-grams, otherwise the list
             contains lists of size `n` in turn containing the strings that make up the n-gram
    """
    require_listlike(docs)

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
    require_listlike(docs)

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


def kwic(docs, search_token, context_size=2, match_type='exact', ignore_case=False, glob_method='match',
         inverse=False, with_metadata=False, as_data_table=False, non_empty=False, glue=None,
         highlight_keyword=None):
    """
    Perform keyword-in-context (kwic) search for `search_token`. Uses similar search parameters as
    :func:`~tmtoolkit.preprocess.filter_tokens`.

    :param docs: list of tokenized documents, optionally as 2-tuple where each element in `docs` is a tuple
                 of (tokens list, tokens metadata dict)
    :param search_token: search pattern
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`).
    :param inverse: Invert the matching results.
    :param with_metadata: Also return metadata (like POS) along with each token.
    :param as_data_table: Return result as data frame with indices "doc" (document label) and "context" (context
                          ID per document) and optionally "position" (original token position in the document) if
                          tokens are not glued via `glue` parameter.
    :param non_empty: If True, only return non-empty result documents.
    :param glue: If not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                              matched keyword.
    :return: Return list with KWIC results per document or a data frame, depending
             on `as_data_table`.
    """
    require_listlike(docs)

    if isinstance(context_size, int):
        context_size = (context_size, context_size)
    else:
        require_listlike(context_size)

    if highlight_keyword is not None and not isinstance(highlight_keyword, str):
        raise ValueError('if `highlight_keyword` is given, it must be of type str')

    if glue:
        if with_metadata or as_data_table:
            raise ValueError('when `glue` is set to True, `with_metadata` and `as_data_table` must be False')
        if not isinstance(glue, str):
            raise ValueError('if `glue` is given, it must be of type str')

    kwic_raw = _build_kwic(docs, search_token,
                           highlight_keyword=highlight_keyword,
                           with_metadata=with_metadata,
                           with_window_indices=as_data_table,
                           context_size=context_size,
                           match_type=match_type,
                           ignore_case=ignore_case,
                           glob_method=glob_method,
                           inverse=inverse)

    return _finalize_kwic_results(kwic_raw,
                                  non_empty=non_empty,
                                  glue=glue,
                                  as_data_table=as_data_table,
                                  with_metadata=with_metadata)


def kwic_table(docs, search_token, context_size=2, match_type='exact', ignore_case=False, glob_method='match',
               inverse=False, glue=' ', highlight_keyword='*'):
    """
    Shortcut for :func:`~tmtoolkit.preprocess.kwic()` to directly return a data frame table with highlighted keywords
    in context.

    :param docs: list of tokenized documents, optionally as 2-tuple where each element in `docs` is a tuple
                 of (tokens list, tokens metadata dict)
    :param search_token: search pattern
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
    :return: Datatable with columns "doc" (document label), "context" (context ID per document) and
             "kwic" containing strings with highlighted keywords in context.
    """

    kwic_raw = kwic(docs, search_token,
                    context_size=context_size,
                    match_type=match_type,
                    ignore_case=ignore_case,
                    glob_method=glob_method,
                    inverse=inverse,
                    with_metadata=False,
                    as_data_table=False,
                    non_empty=True,
                    glue=glue,
                    highlight_keyword=highlight_keyword)

    return _datatable_from_kwic_results(kwic_raw)


def glue_tokens(docs, patterns, glue='_', match_type='exact', ignore_case=False, glob_method='match', inverse=False,
                return_glued_tokens=False):
    """
    Match N *subsequent* tokens to the N patterns in `patterns` using match options like in
    :func:`~tmtoolkit.preprocess.filter_tokens`.
    Join the matched tokens by glue string `glue`. Replace these tokens in the documents.

    If there is metadata, the respective entries for the joint tokens are set to None.

    Return a set of all joint tokens.

    :param docs: list of tokenized documents, optionally as 2-tuple where each element in `docs` is a tuple
                 of (tokens list, tokens metadata dict)
    :param patterns: A sequence of search patterns as excepted by :func:`~tmtoolkit.preprocess.filter_tokens`.
    :param glue: String for joining the subsequent matches.
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`).
    :param inverse: Invert the matching results.
    :param return_glued_tokens: If True, additionally return a set of tokens that were glued
    :return: List of transformed documents or, if `return_glued_tokens` is True, a 2-tuple with
             the list of transformed documents and a set of tokens that were glued
    """
    require_listlike(docs)

    new_tokens = []
    new_tokens_meta = []
    glued_tokens = set()
    match_opts = {'match_type': match_type, 'ignore_case': ignore_case, 'glob_method': glob_method}

    for dtok in docs:
        if isinstance(dtok, tuple):
            dtok, dmeta = dtok
        else:
            dmeta = None

        matches = token_match_subsequent(patterns, dtok, **match_opts)

        if inverse:
            matches = [~m for m in matches]

        dtok, glued = token_glue_subsequent(dtok, matches, glue=glue, return_glued=True)
        glued_tokens.update(glued)
        new_tokens.append(dtok)

        if dmeta is not None:
            new_tokens_meta.append({k: token_glue_subsequent(v, matches, glue=None) for k, v in dmeta.items()})

    assert len(new_tokens) == len(docs)

    if new_tokens_meta:
        assert len(new_tokens_meta) == len(docs)
        new_tokens = list(zip(new_tokens, new_tokens_meta))

    if return_glued_tokens:
        return new_tokens, glued_tokens
    else:
        return new_tokens


def remove_chars(docs, chars):
    """
    Remove all characters listed in `chars` from all tokens.

    :param docs: list of tokenized documents
    :param chars: list of characters to remove
    :return: list of processed documents
    """
    require_listlike(docs)

    if not chars:
        raise ValueError('`chars` must be a non-empty sequence')

    del_chars = str.maketrans('', '', ''.join(chars))

    return [[t.translate(del_chars) for t in dtok] for dtok in docs]


def transform(docs, func, **kwargs):
    """
    Apply `func` to each token in each document of `docs` and return the result.

    :param docs: list of tokenized documents
    :param func: function to apply to each token; should accept a string as first arg. and optional `kwargs`
    :param kwargs: keyword arguments passed to `func`
    :return: list of processed documents
    """
    require_listlike(docs)

    if not callable(func):
        raise ValueError('`func` must be callable')

    if kwargs:
        func_wrapper = lambda t: func(t, **kwargs)
    else:
        func_wrapper = func

    return [list(map(func_wrapper, dtok)) for dtok in docs]


def to_lowercase(docs):
    """
    Apply lowercase transformation to each document.

    :param docs: list of tokenized documents
    :return: list of processed documents
    """
    return transform(docs, str.lower)


def stem(docs, language=defaults.language, stemmer_instance=None):
    """
    Apply stemming to all tokens using a stemmer `stemmer_instance`.

    :param docs: list of tokenized documents
    :param language: language in which `docs` is given
    :param stemmer_instance: a stemmer instance; it must implement a method `stem` that accepts a single string;
                             default is :class:`nltk.stem.SnowballStemmer`
    :return: list of processed documents
    """
    require_listlike(docs)

    if stemmer_instance is None:
        stemmer_instance = nltk.stem.SnowballStemmer(language)
    return transform(docs, stemmer_instance.stem)


def load_pos_tagger_for_language(language):
    """
    Load a Part-of-Speech (POS) tagger for language `language`. Currently only "english" and "german" are supported.

    :param language: the language for the POS tagger
    :return: a 2-tuple with POS tagger instance that has a method ``tag()`` and a string determining the POS tagset like
             "penn" or "stts"
    """

    pos_tagset = None
    picklefile = os.path.join(DATAPATH, language, POS_TAGGER_PICKLE)

    try:
        tagger = unpickle_file(picklefile)

        if language == 'german':
            pos_tagset = 'stts'
    except IOError:
        if language == 'english':
            tagger = _GenericPOSTaggerNLTK
            pos_tagset = tagger.tag_set
        else:
            raise ValueError('no POS tagger available for language "%s"' % language)

    if not hasattr(tagger, 'tag') or not callable(tagger.tag):
        raise ValueError("pos_tagger must have a callable attribute `tag`")

    return tagger, pos_tagset


def pos_tag(docs, language=defaults.language, tagger_instance=None, doc_meta_key='meta_pos'):
    """
    Apply Part-of-Speech (POS) tagging to list of documents `docs`. Either load a tagger based on supplied `language`
    or use the tagger instance `tagger` which must have a method ``tag()``. A tagger can be loaded via
    :func:`~tmtoolkit.preprocess.load_pos_tagger_for_language`.

    POS tagging so far only works for English and German. The English tagger uses the Penn Treebank tagset
    (https://ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html), the
    German tagger uses STTS (http://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/TagSets/stts-table.html).

    :param docs: list of tokenized documents
    :param language: the language for the POS tagger (currently only "english" and "german" are supported) if no
                    `tagger` is given
    :param tagger: a tagger instance to use for tagging if no `language` is given
    :return: a list of the same length as `docs`, containing lists of POS tags according to the tagger's tagset;
             tags correspond to the respective tokens in each document
    """
    require_listlike(docs)

    if tagger_instance is None:
        tagger_instance, _ = load_pos_tagger_for_language(language)

    docs_meta = []
    for dtok in docs:
        if len(dtok) > 0:
            tokens_and_tags = tagger_instance.tag(dtok)
            tags = list(list(zip(*tokens_and_tags))[1])
        else:
            tags = []

        if doc_meta_key:
            docs_meta.append({doc_meta_key: tags})
        else:
            docs_meta.append(tags)

    return docs_meta


def _lemmatize_wrapper_english_wn(row, lemmatizer):
    """Wrapper function to lemmatize English texts using NLTK's WordNet lemmatizer."""
    tok, pos = row
    wn_pos = pos_tag_convert_penn_to_wn(pos)
    if wn_pos:
        return lemmatizer.lemmatize(tok, wn_pos)
    else:
        return tok


def _lemmatize_wrapper_german_germalemma(row, lemmatizer):
    """Wrapper function to lemmatize German texts using ``germalemma`` package."""
    tok, pos = row
    try:
        return lemmatizer.find_lemma(tok, pos)
    except ValueError:
        return tok


def _lemmatize_wrapper_general_patternlib(row, lemmatizer):
    """Wrapper function to lemmatize texts using ``pattern`` package."""
    tok, pos = row
    if pos.startswith('NP'):  # singularize noun
        return lemmatizer.singularize(tok)
    elif pos.startswith('V'):  # get infinitive of verb
        return lemmatizer.conjugate(tok, lemmatizer.INFINITIVE)
    elif pos.startswith('ADJ') or pos.startswith('ADV'):  # get baseform of adjective or adverb
        return lemmatizer.predicative(tok)
    return tok


def load_lemmatizer_for_language(language):
    """
    Load a lemmatize function for a given language.
    For ``language='english'`` returns a lemmatizer based on NLTK WordNet, for ``language='german'`` returns a
    lemmatizer based on ``germalemma``, otherwise returns a lemmatizer based on ``pattern`` package.

    :param language: the language for which the lemmatizer should be loaded
    :return: lemmatize function that accepts a tuple consisting of (token, POS tag)
    """
    if language == 'english':
        lemmatize_wrapper = partial(_lemmatize_wrapper_english_wn, lemmatizer=nltk.stem.WordNetLemmatizer())
    elif language == 'german':
        from germalemma import GermaLemma
        lemmatize_wrapper = partial(_lemmatize_wrapper_german_germalemma, lemmatizer=GermaLemma())
    else:
        if language not in PATTERN_SUBMODULES:
            raise ValueError("no CLiPS pattern module for this language:", language)

        modname = 'pattern.%s' % PATTERN_SUBMODULES[language]
        lemmatize_wrapper = partial(_lemmatize_wrapper_general_patternlib, lemmatizer=import_module(modname))

    return lemmatize_wrapper


def lemmatize(docs, docs_meta, language=defaults.language, lemmatizer_fn=None):
    """
    Lemmatize documents according to `language` or use a custom lemmatizer function `lemmatizer_fn`.

    :param docs: list of tokenized documents
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``; each dict must contain an element ``meta_pos`` with a list
                      containing a POS tag for each token in the respective document
    :param language: the language for which the lemmatizer should be loaded
    :param lemmatizer_fn: alternatively, use this lemmatizer function; this function should accept a tuple consisting
                          of (token, POS tag)
    :return: list of processed documents
    """
    require_listlike(docs)

    if len(docs) != len(docs_meta):
        raise ValueError('`docs` and `docs_meta` must have the same length')

    if lemmatizer_fn is None:
        lemmatizer_fn = load_lemmatizer_for_language(language)

    new_tokens = []
    for i, (dtok, dmeta) in enumerate(zip(docs, docs_meta)):
        if 'meta_pos' not in dmeta:
            raise ValueError('no POS meta data for document #%d' % i)
        new_tokens.append(list(map(lemmatizer_fn, zip(dtok, dmeta['meta_pos']))))

    return new_tokens


def expand_compounds(docs, split_chars=('-',), split_on_len=2, split_on_casechange=False):
    """
    Expand all compound tokens in documents `docs`, e.g. splitting token "US-Student" into two tokens "US" and
    "Student".

    :param docs: list of tokenized documents
    :param split_chars: characters to split on
    :param split_on_len: minimum length of a result token when considering splitting (e.g. when ``split_on_len=2``
                         "e-mail" would not be split into "e" and "mail")
    :param split_on_casechange: use case change to split tokens, e.g. "CamelCase" would become "Camel", "Case"
    :return: list of processed documents
    """
    require_listlike(docs)

    exp_comp = partial(expand_compound_token, split_chars=split_chars, split_on_len=split_on_len,
                       split_on_casechange=split_on_casechange)

    return [flatten_list(map(exp_comp, dtok)) for dtok in docs]


def clean_tokens(docs, docs_meta=None, remove_punct=True, remove_stopwords=True, remove_empty=True,
                 remove_shorter_than=None, remove_longer_than=None, remove_numbers=False, language=defaults.language):
    """
    Apply several token cleaning steps to documents `docs` and optionally documents metadata `docs_meta`, depending on
    the given parameters.

    :param docs: list of tokenized documents
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``
    :param remove_punct: if True, remove all tokens that match the characters listed in ``string.punctuation`` from the
                         documents; if arg is a list, tuple or set, remove all tokens listed in this arg from the
                         documents; if False do not apply punctuation token removal
    :param remove_stopwords: if True, remove stop words as listed in :data:`nltk.corpus.stopwords.words` for the given
                             `languge`; if arg is a list, tuple or set, remove all tokens listed in this arg from the
                             documents; if False do not apply stop word token removal
    :param remove_empty: if True, remove empty strings ``""`` from documents
    :param remove_shorter_than: if given a positive number, remove tokens that are shorter than this number
    :param remove_longer_than: if given a positive number, remove tokens that are longer than this number
    :param remove_numbers: if True, remove all tokens that are deemed numeric by :func:`np.char.isnumeric`
    :param language: language for stop word removal
    :return: either list of processed documents or optional tuple with (processed documents, document meta data)
    """
    require_listlike(docs)

    # add empty token to list of tokens to remove
    tokens_to_remove = [''] if remove_empty else []

    # add punctuation characters to list of tokens to remove
    if remove_punct is True:
        tokens_to_remove.extend(list(string.punctuation))  # default punct. list
    elif isinstance(remove_punct, (tuple, list, set)):
        tokens_to_remove.extend(remove_punct)

    # add stopwords to list of tokens to remove
    if remove_stopwords is True:
        tokens_to_remove.extend(nltk.corpus.stopwords.words(language))   # default stopword list from NLTK
    elif isinstance(remove_stopwords, (tuple, list, set)):
        tokens_to_remove.extend(remove_stopwords)

    # the "remove masks" list holds a binary array for each document where `True` signals a token to be removed
    remove_masks = [np.repeat(False, len(dtok)) for dtok in docs]

    # update remove mask for tokens shorter/longer than a certain number of characters
    if remove_shorter_than is not None or remove_longer_than is not None:
        token_lengths = [np.fromiter(map(len, dtok), np.int, len(dtok)) for dtok in docs]

        if remove_shorter_than is not None:
            if remove_shorter_than < 0:
                raise ValueError('`remove_shorter_than` must be >= 0')
            remove_masks = [mask | (n < remove_shorter_than) for mask, n in zip(remove_masks, token_lengths)]

        if remove_longer_than is not None:
            if remove_longer_than < 0:
                raise ValueError('`remove_longer_than` must be >= 0')
            remove_masks = [mask | (n > remove_longer_than) for mask, n in zip(remove_masks, token_lengths)]

    # update remove mask for numeric tokens
    if remove_numbers:
        remove_masks = [mask | np.char.isnumeric(np.array(dtok, dtype=str))
                        for mask, dtok in zip(remove_masks, docs)]

    # update remove mask for general list of tokens to be removed
    if tokens_to_remove:
        tokens_to_remove = set(tokens_to_remove)
        # this is actually much faster than using np.isin:
        remove_masks = [mask | np.array([t in tokens_to_remove for t in dtok], dtype=bool)
                        for mask, dtok in zip(remove_masks, docs)]

    # apply the mask
    docs, docs_meta = _apply_matches_array(docs, docs_meta, remove_masks, invert=True)

    if docs_meta is None:
        return docs
    else:
        return docs, docs_meta


def filter_tokens(docs, search_tokens, docs_meta=None, match_type='exact', ignore_case=False, glob_method='match',
                  inverse=False):
    """
    Filter tokens in `docs` according to search pattern(s) `search_tokens` and several matching options. Only those
    tokens are retained that match the search criteria unless you set ``inverse=True``, which will *remove* all tokens
    that match the search criteria (which is the same as calling :func:`~tmtoolkit.preprocess.remove_tokens`).

    .. seealso:: :func:`~tmtoolkit.preprocess.remove_tokens`  and :func:`~tmtoolkit.preprocess.token_match`

    :param docs: list of tokenized documents
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``
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
    :return: either list of processed documents or optional tuple with (processed documents, document meta data)
    """
    require_listlike(docs)

    matches = _token_pattern_matches(docs, search_tokens, match_type=match_type, ignore_case=ignore_case,
                                     glob_method=glob_method)

    # apply the mask
    docs, docs_meta = _apply_matches_array(docs, docs_meta, matches, invert=inverse)

    if docs_meta is None:
        return docs
    else:
        return docs, docs_meta


def remove_tokens(docs, search_tokens, docs_meta=None, match_type='exact', ignore_case=False, glob_method='match'):
    """
    Same as :func:`~tmtoolkit.preprocess.filter_tokens` but with ``inverse=True``.

    .. seealso:: :func:`~tmtoolkit.preprocess.filter_tokens`  and :func:`~tmtoolkit.preprocess.token_match`
    """
    return filter_tokens(docs, search_tokens, docs_meta, match_type=match_type, ignore_case=ignore_case,
                         glob_method=glob_method, inverse=True)


def filter_documents(docs, search_tokens, docs_meta=None, doc_labels=None, matches_threshold=1, match_type='exact',
                     ignore_case=False, glob_method='match', inverse_result=False, inverse_matches=False):
    """
    This function is similar to :func:`~tmtoolkit.preprocess.filter_tokens` but applies at document level. For each
    document, the number of matches is counted. If it is at least `matches_threshold` the document is retained,
    otherwise removed. If `inverse_result` is True, then documents that meet the threshold are *removed*.

    .. seealso:: :func:`~tmtoolkit.preprocess.remove_documents`

    :param docs: list of tokenized documents
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``
    :param doc_labels: list of document labels for `docs`
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
    :return: either list of processed documents or optional tuple with (processed documents, document meta data)
    """
    require_listlike(docs)

    matches = _token_pattern_matches(docs, search_tokens, match_type=match_type, ignore_case=ignore_case,
                                     glob_method=glob_method)

    if inverse_matches:
        matches = [~m for m in matches]

    if docs_meta is not None:
        assert len(docs) == len(docs_meta)

    if doc_labels is not None:
        assert len(docs) == len(doc_labels)

    new_doc_labels = []
    new_tokens = []
    new_meta = []
    for i, (dtok, n_matches) in enumerate(zip(docs, map(np.sum, matches))):
        thresh_met = n_matches >= matches_threshold
        if inverse_result:
            thresh_met = not thresh_met
        if thresh_met:
            new_tokens.append(dtok)

            if doc_labels is not None:
                new_doc_labels.append(doc_labels[i])

            if docs_meta:
                new_meta.append(docs_meta[i])

    res = [new_tokens]

    if docs_meta is not None:
        res.append(new_meta)

    if doc_labels is not None:
        res.append(new_doc_labels)

    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)


def remove_documents(docs, search_tokens, docs_meta=None, doc_labels=None, matches_threshold=1, match_type='exact',
                     ignore_case=False, glob_method='match', inverse_matches=False):
    """
    Same as :func:`~tmtoolkit.preprocess.filter_documents` but with ``inverse=True``.

    .. seealso:: :func:`~tmtoolkit.preprocess.filter_documents`
    """
    return filter_documents(docs, search_tokens, docs_meta=docs_meta, doc_labels=doc_labels,
                            matches_threshold=matches_threshold, match_type=match_type, ignore_case=ignore_case,
                            glob_method=glob_method, inverse_matches=inverse_matches, inverse_result=True)


def filter_documents_by_name(docs, doc_labels, name_patterns, docs_meta=None, match_type='exact', ignore_case=False,
                             glob_method='match', inverse=False):
    """
    Filter documents by their name (i.e. document label). Keep all documents whose name matches `name_pattern`
    according to additional matching options. If `inverse` is True, drop all those documents whose name matches,
    which is the same as calling :func:`~tmtoolkit.preprocess.remove_documents_by_name`.

    :param docs: list of tokenized documents
    :param doc_labels: list of document labels for `docs`
    :param name_patterns: either single search string or sequence of search strings
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: invert the matching results
    :return: tuple with list of processed documents, processed document labels, optional document meta data
    """
    require_listlike(docs)
    require_listlike(doc_labels)

    if len(docs) != len(doc_labels):
        raise ValueError('`docs` and `doc_labels` must have the same length')

    if docs_meta is not None and len(docs) != len(docs_meta):
        raise ValueError('`docs` and `docs_meta` must have the same length')

    if isinstance(name_patterns, str):
        name_patterns = [name_patterns]

    matches = np.repeat(True, repeats=len(doc_labels))

    for pat in name_patterns:
        pat_match = token_match(pat, doc_labels, match_type=match_type, ignore_case=ignore_case,
                                glob_method=glob_method)

        if inverse:
            pat_match = ~pat_match

        matches &= pat_match

    assert len(doc_labels) == len(matches)

    new_doc_labels = []
    new_docs = []
    new_docs_meta = []
    for i, (dl, dtok, m) in enumerate(zip(doc_labels, docs, matches)):
        if m:
            new_doc_labels.append(dl)
            new_docs.append(dtok)

            if docs_meta is not None:
                new_docs_meta.append(docs_meta[i])

    if docs_meta is not None:
        return new_docs, new_doc_labels, new_docs_meta
    else:
        return new_docs, new_doc_labels


def remove_documents_by_name(docs, doc_labels, name_patterns, docs_meta=None, match_type='exact', ignore_case=False,
                             glob_method='match'):
    """
    Same as :func:`~tmtoolkit.preprocess.filter_documents_by_name` but with ``inverse=True``.

    .. seealso:: :func:`~tmtoolkit.preprocess.filter_documents_by_name`
    """

    return filter_documents_by_name(docs, doc_labels, name_patterns, docs_meta=docs_meta, match_type=match_type,
                                    ignore_case=ignore_case, glob_method=glob_method)


def filter_for_pos(docs, docs_meta, required_pos, simplify_pos=True, tagset=None, language=defaults.language,
                   inverse=False):
    """
    Filter tokens for a specific POS tag (if `required_pos` is a string) or several POS tags (if `required_pos`
    is a list/tuple/set of strings). The POS tag depends on the tagset used during tagging. If `simplify_pos` is
    True, then the tags are matched to the following simplified forms:

    * ``'N'`` for nouns
    * ``'V'`` for verbs
    * ``'ADJ'`` for adjectives
    * ``'ADV'`` for adverbs
    * ``None`` for all other

    :param docs: list of tokenized documents
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``; POS tags must exist for all documents in `docs_meta`
                      (``"meta_pos"`` key)
    :param required_pos: single string or list of strings with POS tag(s) used for filtering
    :param simplify_pos: before matching simplify POS tags in documents to forms shown above
    :param tagset: POS tagset used while tagging; necessary for simplifying POS tags when `simplify_pos` is True
    :param language: if `tagset` is None, infer from the default POS tagger for the given language
    :param inverse: inverse the matching results, i.e. *remove* tokens that match the POS tag
    :return: tuple with list of (processed documents, document meta data)
    """
    require_listlike(docs)
    require_listlike(docs_meta)

    if len(docs) != len(docs_meta):
        raise ValueError('`docs` and `docs_meta` must have the same length')

    if any('meta_pos' not in d.keys() for d in docs_meta):
        raise ValueError('POS tags must exist for all documents in `docs_meta` ("meta_pos" key)')

    if not isinstance(required_pos, (tuple, list, set, str)) \
            and required_pos is not None:
        raise ValueError('`required_pos` must be a string, tuple, list, set or None')

    if required_pos is None or isinstance(required_pos, str):
        required_pos = [required_pos]

    if tagset is None and language is not None:
        if language == 'german':
            tagset = 'stts'
        if language == 'english':
            tagset = _GenericPOSTaggerNLTK.tag_set

    if simplify_pos:
        simplify_fn = np.vectorize(lambda x: simplified_pos(x, tagset=tagset))
    else:
        simplify_fn = np.vectorize(lambda x: x)  # identity function

    matches = [np.isin(simplify_fn(dmeta['meta_pos']), required_pos) if len(dmeta['meta_pos']) > 0
               else np.array([], dtype=bool)
               for dtok, dmeta in zip(docs, docs_meta)]

    return _apply_matches_array(docs, docs_meta, matches, invert=inverse)


def remove_tokens_by_doc_frequency(docs, which, df_threshold, docs_meta=None, absolute=False, return_blacklist=False):
    """
    Remove tokens according to their document frequency.

    :param docs: list of tokenized documents
    :param which: which threshold comparison to use: either ``'common'``, ``'>'``, ``'>='`` which means that tokens
                  with higher document freq. than (or equal to) `df_threshold` will be removed;
                  or ``'uncommon'``, ``'<'``, ``'<='`` which means that tokens with lower document freq. than
                  (or equal to) `df_threshold` will be removed
    :param df_threshold: document frequency threshold value
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``; POS tags must exist for all documents in `docs_meta`
                      (``"meta_pos"`` key)
    :param absolute: if True, use absolute document frequency (i.e. number of times token X occurs at least once
                     in a document), otherwise use relative document frequency (normalized by number of documents)
    :param return_blacklist: if True return a list of tokens that should be removed instead of the filtered tokens
    :return: when `return_blacklist` is True, return a list of tokens that should be removed; otherwise either return
             list of processed documents or optional tuple with (processed documents, document meta data)
    """
    require_listlike(docs)

    if docs_meta is not None:
        require_listlike(docs_meta)
        if len(docs) != len(docs_meta):
            raise ValueError('`docs` and `docs_meta` must have the same length')

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

    doc_freqs = doc_frequencies(docs, proportions=not absolute)
    blacklist = set(t for t, f in doc_freqs.items() if comp(f, df_threshold))

    if return_blacklist:
        return blacklist

    if blacklist:
        return remove_tokens(docs, docs_meta=docs_meta, search_tokens=blacklist)
    else:
        if docs_meta is None:
            return docs
        else:
            return docs, docs_meta


def remove_common_tokens(docs, docs_meta=None, df_threshold=0.95, absolute=False):
    """
    Shortcut for :func:`~tmtoolkit.preprocess.remove_tokens_by_doc_frequency` for removing tokens *above* a certain
    document frequency.

    :param docs: list of tokenized documents
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``; POS tags must exist for all documents in `docs_meta`
                      (``"meta_pos"`` key)
    :param df_threshold: document frequency threshold value
    :param absolute: if True, use absolute document frequency (i.e. number of times token X occurs at least once
                 in a document), otherwise use relative document frequency (normalized by number of documents)
    :return: either list of processed documents or optional tuple with (processed documents, document meta data)
    """
    return remove_tokens_by_doc_frequency(docs, 'common', df_threshold=df_threshold, docs_meta=docs_meta,
                                          absolute=absolute)


def remove_uncommon_tokens(docs, docs_meta=None, df_threshold=0.05, absolute=False):
    """
    Shortcut for :func:`~tmtoolkit.preprocess.remove_tokens_by_doc_frequency` for removing tokens *below* a certain
    document frequency.

    :param docs: list of tokenized documents
    :param docs_meta: list of meta data for each document in `docs`; each element at index ``i`` is a dict containing
                      the meta data for document ``i``; POS tags must exist for all documents in `docs_meta`
                      (``"meta_pos"`` key)
    :param df_threshold: document frequency threshold value
    :param absolute: if True, use absolute document frequency (i.e. number of times token X occurs at least once
                 in a document), otherwise use relative document frequency (normalized by number of documents)
    :return: either list of processed documents or optional tuple with (processed documents, document meta data)
    """
    return remove_tokens_by_doc_frequency(docs, 'uncommon', df_threshold=df_threshold, docs_meta=docs_meta,
                                          absolute=absolute)


def tokens2ids(docs, return_counts=False):
    """
    Convert a character token array `tok` to a numeric token ID array.
    Return the vocabulary array (char array where indices correspond to token IDs) and token ID array.
    Optionally return the counts of each token in the token ID array when `return_counts` is True.

    .. seealso:: :func:`~tmtoolkit.preprocess.ids2tokens` which reverses this operation.

    :param docs: list of tokenized documents
    :param return_counts: if True, also return array with counts of each unique token in `tok`
    :return: tuple with (vocabulary array, documents as arrays with token IDs) and optional counts
    """
    if not docs:
        if return_counts:
            return empty_chararray(), [], np.array([], dtype=int)
        else:
            return empty_chararray(), []

    if not isinstance(docs[0], np.ndarray):
        docs = list(map(np.array, docs))

    res = np.unique(np.concatenate(docs), return_inverse=True, return_counts=return_counts)

    if return_counts:
        vocab, all_tokids, vocab_counts = res
    else:
        vocab, all_tokids = res

    vocab = vocab.astype(np.str)
    doc_tokids = np.split(all_tokids, np.cumsum(list(map(len, docs))))[:-1]

    if return_counts:
        return vocab, doc_tokids, vocab_counts
    else:
        return vocab, doc_tokids


def ids2tokens(vocab, tokids):
    """
    Convert list of numeric token ID arrays `tokids` to a character token array with the help of the vocabulary
    array `vocab`.
    Returns result as list of string token arrays.

    .. seealso:: :func:`~tmtoolkit.preprocess.tokens2ids` which reverses this operation.

    :param vocab: vocabulary array as from :func:`~tmtoolkit.preprocess.tokens2ids`
    :param tokids: list of numeric token ID arrays as from :func:`~tmtoolkit.preprocess.tokens2ids`
    :return: list of string token arrays
    """
    return [vocab[ids] for ids in tokids]


#%% functions that operate on single token documents (lists or arrays of string tokens)


def token_match(pattern, tokens, match_type='exact', ignore_case=False, glob_method='match'):
    """
    Return a boolean NumPy array signaling matches between `pattern` and `tokens`. `pattern` is a string that will be
    compared with each element in sequence `tokens` either as exact string equality (`match_type` is ``'exact'``) or
    regular expression (`match_type` is ``'regex'``) or glob pattern (`match_type` is ``'glob'``).

    :param pattern: either a string or a compiled RE pattern used for matching against `tokens`
    :param tokens: list or NumPy array of string tokens
    :param match_type: one of: 'exact', 'regex', 'glob'; if 'regex', `search_token` must be RE pattern; if `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre)
    :param ignore_case: if True, ignore case for matching
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's `re.match` or `re.search`)
    :return: 1D boolean NumPy array of length ``len(tokens)`` where elements signal matches between `pattern` and the
             respective token from `tokens`
    """
    if match_type not in {'exact', 'regex', 'glob'}:
        raise ValueError("`match_type` must be one of `'exact', 'regex', 'glob'`")

    if len(tokens) == 0:
        return np.array([], dtype=bool)

    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens)

    if match_type == 'exact':
        return np.char.lower(tokens) == pattern.lower() if ignore_case else tokens == pattern
    elif match_type == 'regex':
        if isinstance(pattern, str):
            pattern = re.compile(pattern, flags=re.IGNORECASE)
        vecmatch = np.vectorize(lambda x: bool(pattern.search(x)))
        return vecmatch(tokens)
    else:
        if glob_method not in {'search', 'match'}:
            raise ValueError("`glob_method` must be one of `'search', 'match'`")

        if isinstance(pattern, str):
            pattern = globre.compile(pattern, flags=re.IGNORECASE)

        if glob_method == 'search':
            vecmatch = np.vectorize(lambda x: bool(pattern.search(x)))
        else:
            vecmatch = np.vectorize(lambda x: bool(pattern.match(x)))

        return vecmatch(tokens) if len(tokens) > 0 else np.array([], dtype=bool)


def token_match_subsequent(patterns, tokens, **kwargs):
    """
    Using N patterns in `patterns`, return each tuple of N matching subsequent tokens from `tokens`. Excepts the same
    token matching options via `kwargs` as :func:`~tmtoolkit.preprocess.token_match`. The results are returned as list
    of NumPy arrays with indices into `tokens`.

    Example::

        # indices:   0        1        2         3        4       5       6
        tokens = ['hello', 'world', 'means', 'saying', 'hello', 'world', '.']

        token_match_subsequent(['hello', 'world'], tokens)
        # [array([0, 1]), array([4, 5])]

        token_match_subsequent(['world', 'hello'], tokens)
        # []

        token_match_subsequent(['world', '*'], tokens, match_type='glob')
        # [array([1, 2]), array([5, 6])]

    .. seealso:: :func:`~tmtoolkit.preprocess.token_match`

    :param patterns: A sequence of search patterns as excepted by :func:`~tmtoolkit.preprocess.token_match`
    :param tokens: A sequence of tokens to be used for matching.
    :param kwargs: Token matching options as passed to :func:`~tmtoolkit.preprocess.token_match`
    :return: List of NumPy arrays with subsequent indices into `tokens`
    """
    require_listlike(patterns)

    n_pat = len(patterns)

    if n_pat < 2:
        raise ValueError('`patterns` must contain at least two strings')

    n_tok = len(tokens)

    if n_tok == 0:
        return []

    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens)

    # iterate through the patterns
    for i_pat, pat in enumerate(patterns):
        if i_pat == 0:   # initial matching on full token array
            next_indices = np.arange(n_tok)
        else:  # subsequent matching uses previous match indices + 1 to match on tokens right after the previous matches
            next_indices = match_indices + 1
            next_indices = next_indices[next_indices < n_tok]   # restrict maximum index

        # do the matching with the current subset of "tokens"
        pat_match = token_match(pat, tokens[next_indices], **kwargs)

        # pat_match is boolean array. use it to select the token indices where we had a match
        # this is used in the next iteration again to select the tokens right after these matches
        match_indices = next_indices[pat_match]

        if len(match_indices) == 0:   # anytime when no successful match appeared, we can return the empty result
            return []                 # because *all* subsequent patterns must match corresponding subsequent tokens

    # at this point, match_indices contains indices i that point to the *last* matched token of the `n_pat` subsequently
    # matched tokens

    assert np.min(match_indices) - n_pat + 1 >= 0
    assert np.max(match_indices) < n_tok

    # so we can use this to reconstruct the whole "trace" subsequently matched indices as final result
    return list(map(lambda i: np.arange(i - n_pat + 1, i + 1), match_indices))


def token_glue_subsequent(tokens, matches, glue='_', return_glued=False):
    """
    Select subsequent tokens as defined by list of indices `matches` (e.g. output of
    :func:`~tmtoolkit.preprocess.token_match_subsequent`) and join those by string `glue`. Return a list of tokens
    where the subsequent matches are replaced by the joint tokens.

    .. warning:: Only works correctly when matches contains indices of *subsequent* tokens.

    Example::

        token_glue_subsequent(['a', 'b', 'c', 'd', 'd', 'a', 'b', 'c'], [np.array([1, 2]), np.array([6, 7])])
        # ['a', 'b_c', 'd', 'd', 'a', 'b_c']

    .. seealso:: :func:`~tmtoolkit.preprocess.token_match_subsequent`

    :param tokens: a sequence of tokens
    :param matches: list of NumPy arrays with *subsequent* indices into `tokens` (e.g. output of
                    :func:`~tmtoolkit.preprocess.token_match_subsequent`)
    :param glue: string for joining the subsequent matches or None if no joint tokens but a None object should be placed
                 in the result list
    :param return_glued: if yes, return also a list of joint tokens
    :return: either two-tuple or list; if `return_glued` is True, return a two-tuple with 1) list of tokens where the
             subsequent matches are replaced by the joint tokens and 2) a list of joint tokens; if `return_glued` is
             True only return 1)
    """
    require_listlike(matches)

    if return_glued and glue is None:
        raise ValueError('if `glue` is None, `return_glued` must be False')

    n_tok = len(tokens)

    if n_tok == 0:
        if return_glued:
            return [], []
        else:
            return []

    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens)

    start_ind = dict(zip(map(lambda x: x[0], matches), matches))
    res = []
    glued = []

    i_t = 0
    while i_t < n_tok:
        if i_t in start_ind:
            seq = tokens[start_ind[i_t]]
            t = None if glue is None else glue.join(seq)
            if return_glued:
                glued.append(t)
            res.append(t)
            i_t += len(seq)
        else:
            res.append(tokens[i_t])
            i_t += 1

    if return_glued:
        return res, glued
    else:
        return res


def make_index_window_around_matches(matches, left, right, flatten=False, remove_overlaps=True):
    """
    Take a boolean 1D vector `matches` of length N and generate an array of indices, where each occurrence of a True
    value in the boolean vector at index i generates a sequence of the form:

    .. code-block:: text

        [i-left, i-left+1, ..., i, ..., i+right-1, i+right, i+right+1]

    If `flatten` is True, then a flattened NumPy 1D array is returned. Otherwise, a list of NumPy arrays is returned,
    where each array contains the window indices.

    `remove_overlaps` is only applied when `flatten` is True.

    Example with ``left=1 and right=1, flatten=False``:

    .. code-block:: text

        input:
        #   0      1      2      3     4      5      6      7     8
        [True, True, False, False, True, False, False, False, True]
        output (matches *highlighted*):
        [[0, *1*], [0, *1*, 2], [3, *4*, 5], [7, *8*]]

    Example with ``left=1 and right=1, flatten=True, remove_overlaps=True``:

    .. code-block:: text

        input:
        #   0      1      2      3     4      5      6      7     8
        [True, True, False, False, True, False, False, False, True]
        output (matches *highlighted*, other values belong to the respective "windows"):
        [*0*, *1*, 2, 3, *4*, 5, 7, *8*]
    """
    if not isinstance(matches, np.ndarray) or matches.dtype != bool:
        raise ValueError('`matches` must be a boolean NumPy array')
    if not isinstance(left, int) or left < 0:
        raise ValueError('`left` must be an integer >= 0')
    if not isinstance(right, int) or right < 0:
        raise ValueError('`right` must be an integer >= 0')

    ind = np.where(matches)[0]
    nested_ind = list(map(lambda x: np.arange(x - left, x + right + 1), ind))

    if flatten:
        if not nested_ind:
            return np.array([], dtype=np.int)

        window_ind = np.concatenate(nested_ind)
        window_ind = window_ind[(window_ind >= 0) & (window_ind < len(matches))]

        if remove_overlaps:
            return np.sort(np.unique(window_ind))
        else:
            return window_ind
    else:
        return [w[(w >= 0) & (w < len(matches))] for w in nested_ind]


#%% functions that operate on single tokens / strings


def expand_compound_token(t, split_chars=('-',), split_on_len=2, split_on_casechange=False):
    """
    Expand a token `t` if it is a compound word, e.g. splitting token "US-Student" into two tokens "US" and
    "Student".

    .. seealso:: :func:`~tmtoolkit.preprocess.expand_compounds` which operates on token documents

    :param t: string token
    :param split_chars: characters to split on
    :param split_on_len: minimum length of a result token when considering splitting (e.g. when ``split_on_len=2``
                         "e-mail" would not be split into "e" and "mail")
    :param split_on_casechange: use case change to split tokens, e.g. "CamelCase" would become "Camel", "Case"
    :return: list with split sub-tokens or single original token, i.e. ``[t]``
    """
    if not isinstance(t, str):
        raise ValueError('`t` must be a string')

    if not split_on_len and not split_on_casechange:
        raise ValueError('At least one of the arguments `split_on_len` and `split_on_casechange` must evaluate to True')

    if isinstance(split_chars, str):
        split_chars = (split_chars,)

    require_listlike_or_set(split_chars)

    split_chars = set(split_chars)
    t_parts = str_multisplit(t, split_chars)

    n_parts = len(t_parts)
    assert n_parts > 0

    if n_parts == 1:
        return t_parts
    else:
        parts = []
        add = False  # signals if current part should be appended to previous part

        for p in t_parts:
            if not p: continue  # skip empty part
            if add and parts:   # append current part p to previous part
                parts[-1] += p
            else:               # add p as separate token
                parts.append(p)

            if split_on_len:
                # if p consists of less than `split_on_len` characters -> append the next p to it
                add = len(p) < split_on_len

            if split_on_casechange:
                # alt. strategy: if p is all uppercase ("US", "E", etc.) -> append the next p to it
                add = add and p.isupper() if split_on_len else p.isupper()

        if add and len(parts) >= 2:
            parts = parts[:-2] + [parts[-2] + parts[-1]]

        return parts or [t]


def str_multisplit(s, split_chars):
    """
    Split string `s` by all characters in `split_chars`.

    :param s: a string to split
    :param split_chars: sequence or set of characters to use for splitting
    :return: list of split string parts
    """
    if not isinstance(s, (str, bytes)):
        raise ValueError('`s` must be of type `str` or `bytes`')

    require_listlike_or_set(split_chars)

    parts = [s]
    for c in split_chars:
        parts_ = []
        for p in parts:
            if c in p:
                parts_.extend(p.split(c))
            else:
                parts_.append(p)
        parts = parts_

    return parts


#%% Part-of-Speech tag handling


def pos_tag_convert_penn_to_wn(tag):
    """
    Convert POS tag from Penn tagset to WordNet tagset.

    :param tag: a tag from Penn tagset
    :return: a tag from WordNet tagset or None if no corresponding tag could be found
    """
    from nltk.corpus import wordnet as wn

    if tag in ['JJ', 'JJR', 'JJS']:
        return wn.ADJ
    elif tag in ['RB', 'RBR', 'RBS']:
        return wn.ADV
    elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return wn.NOUN
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return wn.VERB
    return None


def simplified_pos(pos, tagset=None, default=''):
    """
    Return a simplified POS tag for a full POS tag `pos` belonging to a tagset `tagset`. By default the WordNet
    tagset is assumed.
    Does the following conversion by default:

    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all ADJ... (adjective) tags to 'ADJ'
    - all ADV... (adverb) tags to 'ADV'
    - all other to `default`

    Does the following conversion by with ``tagset=='penn'``:

    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all JJ... (adjective) tags to 'ADJ'
    - all RB... (adverb) tags to 'ADV'
    - all other to `default`

    :param pos: a POS tag
    :param tagset: the tagset used for `pos`
    :param default: default return value when tag could not be simplified
    :return: simplified tag
    """
    if tagset == 'penn':
        if pos.startswith('N') or pos.startswith('V'):
            return pos[0]
        elif pos.startswith('JJ'):
            return 'ADJ'
        elif pos.startswith('RB'):
            return 'ADV'
        else:
            return default

    else:   # default: WordNet, STTS or unknown
        if pos.startswith('N') or pos.startswith('V'):
            return pos[0]
        elif pos.startswith('ADJ') or pos.startswith('ADV'):
            return pos[:3]
        else:
            return default


#%% helper functions and classes


def _build_kwic(docs, search_token, highlight_keyword, with_metadata, with_window_indices, context_size,
                match_type, ignore_case, glob_method, inverse):
    """
    Helper function to build keywords-in-context (KWIC) results from documents `docs`.

    :param docs: list of tokenized documents, optionally as 2-tuple where each element in `docs` is a tuple
                 of (tokens list, tokens metadata dict)
    :param search_token: search pattern
    :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                              matched keyword.
    :param with_metadata: add document metadata to KWIC results
    :param with_window_indices: add window indices to KWIC results
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`).
    :param inverse: Invert the matching results.
    :return: list with KWIC results per document
    """
    # find matches for search criteria -> list of NumPy boolean mask arrays
    matches = [token_match(search_token, dtok[0] if isinstance(dtok, tuple) else dtok,
                           match_type=match_type,
                           ignore_case=ignore_case,
                           glob_method=glob_method) for dtok in docs]

    if inverse:
        matches = [~m for m in matches]

    left, right = context_size

    kwic_list = []
    for mask, dtok in zip(matches, docs):
        if isinstance(dtok, tuple):
            dtok, dmeta = dtok
        else:
            dmeta = None

        dtok_arr = np.array(dtok, dtype=str)

        ind = np.where(mask)[0]
        ind_windows = make_index_window_around_matches(mask, left, right, flatten=False)

        assert len(ind) == len(ind_windows)
        windows_in_doc = []
        for match_ind, win in zip(ind, ind_windows):  # win is an array of indices into dtok_arr
            tok_win = dtok_arr[win].tolist()

            if highlight_keyword is not None:
                highlight_mask = win == match_ind
                assert np.sum(highlight_mask) == 1
                highlight_ind = np.where(highlight_mask)[0][0]
                tok_win[highlight_ind] = highlight_keyword + tok_win[highlight_ind] + highlight_keyword

            win_res = {'token': tok_win}

            if with_window_indices:
                win_res['index'] = win

            if with_metadata and dmeta is not None:
                for meta_key, meta_vals in dmeta.items():
                    win_res[meta_key] = np.array(meta_vals)[win].tolist()

            windows_in_doc.append(win_res)

        kwic_list.append(windows_in_doc)

    return kwic_list


def _finalize_kwic_results(kwic_results, non_empty, glue, as_data_table, with_metadata):
    """
    Helper function to finalize raw KWIC results coming from `_build_kwic()`: Filter results, "glue" (join) tokens,
    transform to datatable, return or dismiss metadata.
    """
    if non_empty:
        if isinstance(kwic_results, dict):
            kwic_results = {dl: windows for dl, windows in kwic_results.items() if len(windows) > 0}
        else:
            assert isinstance(kwic_results, list)
            kwic_results = [windows for windows in kwic_results if len(windows) > 0]

    if glue is not None:
        if isinstance(kwic_results, dict):
            return {dl: [glue.join(win['token']) for win in windows] for dl, windows in kwic_results.items()}
        else:
            assert isinstance(kwic_results, list)
            return [[glue.join(win['token']) for win in windows] for windows in kwic_results]
    elif as_data_table:
        dfs = []
        for i_doc, dl_or_win in enumerate(kwic_results):
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
                dfs.append(dt.Frame(OrderedDict(zip(df_cols, df_windata))))

        if dfs:
            kwic_df = dt.rbind(*dfs)
            return kwic_df[:, :, dt.sort('doc', 'context', 'position')]
        else:
            return dt.Frame(OrderedDict(zip(['doc', 'context', 'position', 'token'], [[] for _ in range(4)])))
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

        dfs.append(dt.Frame(OrderedDict(zip(['doc', 'context', 'kwic'],
                                            [np.repeat(dl, len(windows)), np.arange(len(windows)), windows]))))
    if dfs:
        kwic_df = dt.rbind(*dfs)
        return kwic_df[:, :, dt.sort('doc', 'context')]
    else:
        return dt.Frame(OrderedDict(zip(['doc', 'context', 'kwic'], [[] for _ in range(3)])))


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


def _token_pattern_matches(docs, search_tokens, match_type, ignore_case, glob_method):
    """
    Helper function to apply `token_match` with multiple patterns in `search_tokens` to `docs`.
    The matching results for each pattern in `search_tokens` are combined via logical OR.
    Returns a list of length `docs` containing boolean arrays that signal the pattern matches for each token in each
    document.
    """
    if isinstance(search_tokens, str):
        search_tokens = [search_tokens]

    matches = [np.repeat(False, repeats=len(dtok)) for dtok in docs]

    for dtok, dmatches in zip(docs, matches):
        for pat in search_tokens:
            pat_match = token_match(pat, dtok, match_type=match_type, ignore_case=ignore_case, glob_method=glob_method)

            dmatches |= pat_match

    return matches


def _apply_matches_array(docs, docs_meta, matches, invert=False):
    """
    Helper function to apply a list of boolean arrays `matches` that signal token to pattern matches to a list of
    tokenized documents `docs` and optional document meta data `docs_meta` (if it is not None).
    Returns a tuple with (list of filtered documents, filtered document meta data).
    """
    if invert:
        matches = [~m for m in matches]

    docs = [np.array(dtok)[mask].tolist() for mask, dtok in zip(matches, docs)]

    if docs_meta is not None:
        new_meta = []
        assert len(matches) == len(docs_meta)
        for mask, dmeta in zip(matches, docs_meta):
            new_dmeta = {}
            for meta_key, meta_vals in dmeta.items():
                new_dmeta[meta_key] = np.array(meta_vals)[mask].tolist()
            new_meta.append(new_dmeta)

        docs_meta = new_meta

    return docs, docs_meta


class _GenericPOSTaggerNLTK:
    """
    Default POS tagger for English.
    Uses ``nltk.pos_tag`` and produces tags from Penn tagset.
    """
    tag_set = 'penn'

    @staticmethod
    def tag(tokens):
        return nltk.pos_tag(tokens)

