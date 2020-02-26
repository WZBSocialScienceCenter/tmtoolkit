"""
Common functions for text processing.

Most functions of this internal module are exported in ``__init__.py`` and make up the functional text processing API of
tmtoolkit.

Markus Konrad <markus.konrad@wzb.eu>
"""

import re
import os
import operator
import pickle
from collections import Counter, OrderedDict
from functools import partial

import globre
import numpy as np
from spacy.tokens import Doc

from .._pd_dt_compat import pd_dt_frame, pd_dt_concat, pd_dt_sort
from ..bow.dtm import create_sparse_dtm
from ..utils import flatten_list, require_listlike, empty_chararray, widen_chararray, require_listlike_or_set


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.normpath(os.path.join(MODULE_PATH, '..', 'data'))

DEFAULT_LANGUAGE_MODELS = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm',
    'es': 'es_core_news_sm',
    'pt': 'pt_core_news_sm',
    'it': 'it_core_news_sm',
    'nl': 'nl_core_news_sm',
    'el': 'el_core_news_sm',
    'nb': 'nb_core_news_sm',
    'lt': 'lt_core_news_sm',
}

LANGUAGE_LABELS = {
    'en': 'english',
    'de': 'german',
    'fr': 'french',
    'es': 'spanish',
    'pt': 'portuguese',
    'it': 'italian',
    'nl': 'dutch',
    'el': 'greek',
    'nb': 'norwegian-bokmal',
    'lt': 'lithuanian',
}


def load_stopwords(language):
    """
    Load stopwords for language code `language`.

    :param language: two-letter ISO 639-1 language code
    :return: list of stopword strings or None if loading failed
    """

    if not isinstance(language, str) or len(language) != 2:
        raise ValueError('`language` must be a two-letter ISO 639-1 language code')

    stopwords_pickle = os.path.join(DATAPATH, language, 'stopwords.pickle')
    try:
        with open(stopwords_pickle, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, IOError):
        return None


def clean_tokens(docs, remove_punct=True, remove_stopwords=True, remove_empty=True,
                 remove_shorter_than=None, remove_longer_than=None, remove_numbers=False, language=None):
    """
    Apply several token cleaning steps to documents `docs` and optionally documents metadata `docs_meta`, depending on
    the given parameters.

    :param docs: list of tokenized documents
    :param remove_punct: if True, remove all tokens that match the characters listed in ``string.punctuation`` from the
                         documents; if arg is a list, tuple or set, remove all tokens listed in this arg from the
                         documents; if False do not apply punctuation token removal
    :param remove_stopwords: if True, remove stop words for the given `language` as loaded via
                             `~tmtoolkit.preprocess.load_stopwords` ; if arg is a list, tuple or set, remove all tokens
                             listed in this arg from the documents; if False do not apply stop word token removal
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
    if isinstance(remove_punct, (tuple, list, set)):
        tokens_to_remove.extend(remove_punct)

    # add stopwords to list of tokens to remove
    if remove_stopwords is True:
        # default stopword list from NLTK
        tokens_to_remove.extend(load_stopwords(language))
    elif isinstance(remove_stopwords, (tuple, list, set)):
        tokens_to_remove.extend(remove_stopwords)

    # the "remove masks" list holds a binary array for each document where `True` signals a token to be removed
    remove_masks = [np.repeat(False, len(doc)) for doc in docs]

    # update remove mask for punctuation
    if remove_punct is True:
        remove_masks = [mask | doc.to_array('is_punct').astype(np.bool_)
                        for mask, doc in zip(remove_masks, docs)]

    # update remove mask for tokens shorter/longer than a certain number of characters
    if remove_shorter_than is not None or remove_longer_than is not None:
        token_lengths = [np.fromiter(map(len, doc), np.int, len(doc)) for doc in docs]

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
        remove_masks = [mask | doc.to_array('like_num').astype(np.bool_)
                        for mask, doc in zip(remove_masks, docs)]

    # update remove mask for general list of tokens to be removed
    if tokens_to_remove:
        tokens_to_remove = set(tokens_to_remove)
        # this is actually much faster than using np.isin:
        remove_masks = [mask | np.array([t in tokens_to_remove for t in doc.user_data['tokens']], dtype=bool)
                        for mask, doc in zip(remove_masks, docs)]

    # apply the mask
    return _apply_matches_array(docs, remove_masks, invert=True)


def filter_tokens_by_mask(docs, mask, inverse=False):
    """
    Filter tokens in `docs` according to a binary mask specified by `mask`.

    :param docs: list of tokenized documents
    :param mask: a list containing a mask list for each document in `docs`; each mask list contains boolean values for
                 each token in that document, where `True` means keeping that token and `False` means removing it;
    :param inverse: inverse the mask for filtering, i.e. keep all tokens with a mask set to `False` and remove all those
                    with `True`
    :return: either list of processed documents or optional tuple with (processed documents, document meta data)
    """

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

    :param docs: list of tokenized documents
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param by_meta: if not None, this should be a string of a meta data key in `docs_meta`; this meta data will then be
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
    :return: either list of processed documents or optional tuple with (processed documents, document meta data)
    """
    require_listlike(docs)

    matches = _token_pattern_matches(_match_against(docs, by_meta), search_tokens, match_type=match_type,
                                     ignore_case=ignore_case, glob_method=glob_method)

    return filter_tokens_by_mask(docs, matches, inverse=inverse)


def filter_tokens_with_kwic(docs, search_tokens, context_size=2, match_type='exact', ignore_case=False,
                            glob_method='match', inverse=False):
    """
    Filter tokens in `docs` according to Keywords-in-Context (KWIC) context window of size `context_size` around
    `search_tokens`. Works similar to :func:`~tmtoolkit.preprocess.kwic`, but returns result as list of tokenized
    documents, i.e. in the same structure as `docs` whereas :func:`~tmtoolkit.preprocess.kwic` returns result as
    list of *KWIC windows* into `docs`.

    .. seealso:: :func:`~tmtoolkit.preprocess.kwic`

    :param docs: list of tokenized documents
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
    :return: either list of processed documents or optional tuple with (processed documents, document meta data)
    """
    require_listlike(docs)

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


def remove_tokens(docs, search_tokens, by_meta=None, match_type='exact', ignore_case=False,
                  glob_method='match'):
    """
    Same as :func:`~tmtoolkit.preprocess.filter_tokens` but with ``inverse=True``.

    .. seealso:: :func:`~tmtoolkit.preprocess.filter_tokens`  and :func:`~tmtoolkit.preprocess.token_match`
    """
    return filter_tokens(docs, search_tokens=search_tokens, by_meta=by_meta, match_type=match_type,
                         ignore_case=ignore_case, glob_method=glob_method, inverse=True)


def filter_documents(docs, search_tokens, by_meta=None, matches_threshold=1,
                     match_type='exact', ignore_case=False, glob_method='match', inverse_result=False,
                     inverse_matches=False):
    """
    This function is similar to :func:`~tmtoolkit.preprocess.filter_tokens` but applies at document level. For each
    document, the number of matches is counted. If it is at least `matches_threshold` the document is retained,
    otherwise removed. If `inverse_result` is True, then documents that meet the threshold are *removed*.

    .. seealso:: :func:`~tmtoolkit.preprocess.remove_documents`

    :param docs: list of tokenized documents
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param by_meta: if not None, this should be a string of a meta data key in `docs_meta`; this meta data will then be
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
    :return: list of filtered documents
    """
    require_listlike(docs)

    matches = _token_pattern_matches(_match_against(docs, by_meta), search_tokens, match_type=match_type,
                                     ignore_case=ignore_case, glob_method=glob_method)

    if inverse_matches:
        matches = [~m for m in matches]

    new_docs = []
    for i, (doc, n_matches) in enumerate(zip(docs, map(np.sum, matches))):
        thresh_met = n_matches >= matches_threshold
        if thresh_met or (inverse_result and not thresh_met):
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


def filter_documents_by_name(docs, name_patterns, match_type='exact', ignore_case=False,
                             glob_method='match', inverse=False):
    """
    Filter documents by their name (i.e. document label). Keep all documents whose name matches `name_pattern`
    according to additional matching options. If `inverse` is True, drop all those documents whose name matches,
    which is the same as calling :func:`~tmtoolkit.preprocess.remove_documents_by_name`.

    :param docs: list of tokenized documents
    :param name_patterns: either single search string or sequence of search strings
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: invert the matching results
    :return: list of filtered documents
    """
    require_listlike(docs)

    if isinstance(name_patterns, str):
        name_patterns = [name_patterns]

    matches = np.repeat(True, repeats=len(docs))
    doc_labels = _get_docs_attr(docs, 'label')

    for pat in name_patterns:
        pat_match = token_match(pat, doc_labels, match_type=match_type, ignore_case=ignore_case,
                                glob_method=glob_method)

        if inverse:
            pat_match = ~pat_match

        matches &= pat_match

    assert len(doc_labels) == len(matches)

    return [doc for doc, m in zip(docs, matches) if m]


def remove_documents_by_name(docs, name_patterns, match_type='exact', ignore_case=False,
                             glob_method='match'):
    """
    Same as :func:`~tmtoolkit.preprocess.filter_documents_by_name` but with ``inverse=True``.

    .. seealso:: :func:`~tmtoolkit.preprocess.filter_documents_by_name`
    """

    return filter_documents_by_name(docs, name_patterns, match_type=match_type,
                                    ignore_case=ignore_case, glob_method=glob_method)


def filter_for_pos(docs, required_pos, simplify_pos=True, tagset='ud', inverse=False):
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

    :param docs: list of tokenized documents
    :param required_pos: single string or list of strings with POS tag(s) used for filtering
    :param simplify_pos: before matching simplify POS tags in documents to forms shown above
    :param tagset: POS tagset used while tagging; necessary for simplifying POS tags when `simplify_pos` is True
    :param inverse: inverse the matching results, i.e. *remove* tokens that match the POS tag
    :return: filtered documents
    """
    require_listlike(docs)

    docs_pos = _get_docs_tokenattrs(docs, 'pos_', custom_attr=False)

    if not isinstance(required_pos, (tuple, list, set, str)) \
            and required_pos is not None:
        raise ValueError('`required_pos` must be a string, tuple, list, set or None')

    if required_pos is None or isinstance(required_pos, str):
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


def remove_tokens_by_doc_frequency(docs, which, df_threshold, docs_meta=None, absolute=False, return_blacklist=False,
                                   return_mask=False):
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
    :param return_mask: if True return a list of token masks where each occurrence of True signals a token to
                        be removed
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
    mask = [[comp(doc_freqs[t], df_threshold) for t in dtok] for dtok in docs]

    if return_blacklist:
        blacklist = set(t for t, f in doc_freqs.items() if comp(f, df_threshold))
        if return_mask:
            return blacklist, mask

    if return_mask:
        return mask

    return remove_tokens_by_mask(docs, mask, docs_meta)


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


def simplified_pos(pos, tagset='ud', default=''):
    """
    Return a simplified POS tag for a full POS tag `pos` belonging to a tagset `tagset`.

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

    Does the following conversion by with ``tagset=='ud'``:

    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all JJ... (adjective) tags to 'ADJ'
    - all RB... (adverb) tags to 'ADV'
    - all other to `default`

    :param pos: a POS tag
    :param tagset: tagset used for `pos`; can be ``'wn'`` (WordNet), ``'penn'`` (Penn tagset)
                   or ``'ud'`` (universal dependencies – default)
    :param default: default return value when tag could not be simplified
    :return: simplified tag
    """

    if tagset == 'ud':
        if pos in ('NOUN', 'PROPN'):
            return 'N'
        elif pos == 'VERB':
            return 'V'
        elif pos in ('ADJ', 'ADV'):
            return pos
        else:
            return default
    elif tagset == 'penn':
        if pos.startswith('N') or pos.startswith('V'):
            return pos[0]
        elif pos.startswith('JJ'):
            return 'ADJ'
        elif pos.startswith('RB'):
            return 'ADV'
        else:
            return default
    elif tagset == 'wn':   # default: WordNet
        if pos.startswith('N') or pos.startswith('V'):
            return pos[0]
        elif pos.startswith('ADJ') or pos.startswith('ADV'):
            return pos[:3]
        else:
            return default
    else:
        raise ValueError('unknown tagset "%s"' % tagset)


#%% helper functions and classes


def _replace_doc_tokens(doc, new_tok):
    if isinstance(doc, list):
        return new_tok
    else:
        # replace all non-filtered tokens
        assert sum(doc.user_data['mask']) == len(new_tok)
        doc.user_data['tokens'][doc.user_data['mask']] = new_tok
        return doc


def _get_docs_attr(docs, attr_name, custom_attr=True):
    return [getattr(doc._, attr_name) if custom_attr else getattr(doc, attr_name) for doc in docs]

