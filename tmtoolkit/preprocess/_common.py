"""
Common functions and constants.
"""
import re
from collections import Counter, OrderedDict

import globre
import numpy as np
import datatable as dt
import nltk

from .. import defaults
from ..bow.dtm import create_sparse_dtm
from ..utils import flatten_list, require_listlike, empty_chararray, require_listlike_or_set


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


def kwic(docs, search_token, context_size=2, match_type='exact', ignore_case=False, glob_method='match',
         inverse=False, with_metadata=False, as_data_table=False, non_empty=False, glue=None,
         highlight_keyword=None):
    """
    Perform keyword-in-context (kwic) search for `search_token`. Uses similar search parameters as
    `filter_tokens()`.

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
                        behavior as Python's `re.match` or `re.search`).
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
    Shortcut for `get_kwic` to directly return a data frame table with highlighted keywords in context.

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
                        behavior as Python's `re.match` or `re.search`).
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
    Match N *subsequent* tokens to the N patterns in `patterns` using match options like in `filter_tokens`.
    Join the matched tokens by glue string `glue`. Replace these tokens in the documents.

    If there is metadata, the respective entries for the joint tokens are set to None.

    Return a set of all joint tokens.

    :param docs: list of tokenized documents, optionally as 2-tuple where each element in `docs` is a tuple
                 of (tokens list, tokens metadata dict)
    :param patterns: A sequence of search patterns as excepted by `filter_tokens`.
    :param glue: String for joining the subsequent matches.
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's `re.match` or `re.search`).
    :param inverse: Invert the matching results.
    :param return_glued_tokens: If True, additionally return a set of tokens that were glued
    :return: List of transformed documents or, if `return_glued_tokens` is True, a 2-tuple with
             the list of transformed documents and a set of tokens that were glued
    """

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
    if not chars:
        raise ValueError('`chars` must be a non-empty sequence')

    del_chars = str.maketrans('', '', ''.join(chars))

    return [[t.translate(del_chars) for t in dtok] for dtok in docs]


#%% functions that operate on single document tokens

def token_match(pattern, tokens, match_type='exact', ignore_case=False, glob_method='match'):
    """
    Return a boolean NumPy array signaling matches between `pattern` and `tokens`. `pattern` is a string that will be
    compared with each element in sequence `tokens` either as exact string equality (`match_type` is 'exact') or
    regular expression (`match_type` is 'regex') or glob pattern (`match_type` is 'glob').
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
    token matching options via `kwargs` as `token_match`. The results are returned as list of NumPy arrays with indices
    into `tokens`.

    Example:

    ```
    # indices:   0        1        2         3        4       5       6
    tokens = ['hello', 'world', 'means', 'saying', 'hello', 'world', '.']

    token_match_subsequent(['hello', 'world'], tokens)
    # [array([0, 1]), array([4, 5])]

    token_match_subsequent(['world', 'hello'], tokens)
    # []

    token_match_subsequent(['world', '*'], tokens, match_type='glob')
    # [array([1, 2]), array([5, 6])]
    ```

    :param patterns: A sequence of search patterns as excepted by `token_match`
    :param tokens: A sequence of tokens to be used for matching.
    :param kwargs: Token matching options as passed to `token_match`
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
    Select subsequent tokens as defined by list of indices `matches` (e.g. output of `token_match_subsequent`) and
    join those by string `glue`. Return a list of tokens where the subsequent matches are replaced by the joint tokens.
    **Important**: Only works correctly when matches contains indices of *subsequent* tokens.

    Example:

    ```
    token_glue_subsequent(['a', 'b', 'c', 'd', 'd', 'a', 'b', 'c'], [np.array([1, 2]), np.array([6, 7])])
    # ['a', 'b_c', 'd', 'd', 'a', 'b_c']
    ```

    :param tokens: A sequence of tokens.
    :param matches: List of NumPy arrays with *subsequent* indices into `tokens` (e.g. output of
                    `token_match_subsequent`)
    :param glue: String for joining the subsequent matches or None if no joint tokens but a None object should be placed
                 in the result list.
    :param return_glued: If yes, return also a list of joint tokens.
    :return: Either two-tuple or list. If `return_glued` is True, return a two-tuple with 1) list of tokens where the
             subsequent matches are replaced by the joint tokens and 2) a list of joint tokens. If `return_glued` is
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

    [i-left, i-left+1, ..., i, ..., i+right-1, i+right, i+right+1]

    If `flatten` is True, then a flattened NumPy 1D array is returned. Otherwise, a list of NumPy arrays is returned,
    where each array contains the window indices.

    `remove_overlaps` is only applied when `flatten` is True.

    Example with left=1 and right=1, flatten=False:

    ```
    input:
    #   0      1      2      3     4      5      6      7     8
    [True, True, False, False, True, False, False, False, True]
    output (matches *highlighted*):
    [[0, *1*], [0, *1*, 2], [3, *4*, 5], [7, *8*]]
    ```

    Example with left=1 and right=1, flatten=True, remove_overlaps=True:

    ```
    input:
    #   0      1      2      3     4      5      6      7     8
    [True, True, False, False, True, False, False, False, True]
    output (matches *highlighted*, other values belong to the respective "windows"):
    [*0*, *1*, 2, 3, *4*, 5, 7, *8*]
    ```
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


def tokens2ids(tok, return_counts=False):
    """
    Convert a character token array `tok` to a numeric token ID array.
    Return the vocabulary array (char array where indices correspond to token IDs) and token ID array.
    Optionally return the counts of each token in the token ID array when `return_counts` is True.

    Use `ids2tokens(<vocab>, <token IDs>)` to reverse this operation.
    """
    if not tok:
        if return_counts:
            return empty_chararray(), [], np.array([], dtype=int)
        else:
            return empty_chararray(), []

    if not isinstance(tok[0], np.ndarray):
        tok = list(map(np.array, tok))

    res = np.unique(np.concatenate(tok), return_inverse=True, return_counts=return_counts)

    if return_counts:
        vocab, all_tokids, vocab_counts = res
    else:
        vocab, all_tokids = res

    vocab = vocab.astype(np.str)
    doc_tokids = np.split(all_tokids, np.cumsum(list(map(len, tok))))[:-1]

    if return_counts:
        return vocab, doc_tokids, vocab_counts
    else:
        return vocab, doc_tokids


def ids2tokens(vocab, tokids):
    """
    Convert numeric token ID array `tokids` to a character token array with the help of the vocabulary array `vocab`.
    Returns result as list.
    Use `tokens2ids(tokens)` to reverse this operation.
    """
    return [vocab[ids] for ids in tokids]


#%% Part-of-Speech tag handling


def pos_tag_convert_penn_to_wn(tag):
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
    Does the following conversion by with `tagset=='penn'`:
    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all JJ... (adjective) tags to 'ADJ'
    - all RB... (adverb) tags to 'ADV'
    - all other to `default`
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


#%% functions that operate on single tokens / strings


def expand_compound_token(t, split_chars=('-',), split_on_len=2, split_on_casechange=False):
    if not isinstance(t, (list, tuple, set, np.ndarray)):
        raise ValueError('`t` must be a list, tuple, set or NumPy array of strings')

    if not split_on_len and not split_on_casechange:
        raise ValueError('At least one of the arguments `split_on_len` and `split_on_casechange` must evaluate to True')

    if len(t) == 0:
        return []

    if isinstance(split_chars, str):
        split_chars = (split_chars,)

    require_listlike_or_set(split_chars)

    split_chars = set(split_chars)

    if len(split_chars) == 1:
        split_t = np.char.split(t, next(iter(split_chars)))
    else:
        split_t = map(lambda x: str_multisplit(x, split_chars), t)

    res = []
    for t_parts, orig_t in zip(split_t, t):  # for each part p in compound token t
        n_parts = len(t_parts)
        assert n_parts > 0
        if n_parts == 1:
            res.append(t_parts)
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

            res.append(parts or [orig_t])  # if parts is empty, return unchanged input

    return res


def str_multisplit(s, split_chars):
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



#%% helper functions


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
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's `re.match` or `re.search`).
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
            tok_win = dtok_arr[win]

            if highlight_keyword is not None:
                highlight_mask = win == match_ind
                assert np.sum(highlight_mask) == 1
                new_tok = highlight_keyword + tok_win[highlight_mask][0] + highlight_keyword
                if len(new_tok) > np.char.str_len(tok_win[highlight_mask]).max():  # may need to create more space
                    tok_win = tok_win.astype('<U' + str(len(new_tok)))             # for this token
                tok_win[highlight_mask] = new_tok

            win_res = {'token': tok_win.tolist()}

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
