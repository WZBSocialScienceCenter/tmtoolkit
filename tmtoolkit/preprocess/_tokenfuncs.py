"""
Functions that operate on token strings.
"""

import re

import numpy as np
import globre

from ..utils import require_listlike, require_listlike_or_set, flatten_list, empty_chararray


#%% functions that operate on lists of string tokens


def remove_chars(docs, chars):
    """
    Remove all characters listed in `chars` from all tokens.

    :param docs: list of string tokens documents
    :param chars: list of characters to remove
    :return: list of processed string tokens documents
    """
    require_tokendocs(docs)

    if not chars:
        raise ValueError('`chars` must be a non-empty sequence')

    del_chars = str.maketrans('', '', ''.join(chars))

    return [[t.translate(del_chars) for t in dtok] for dtok in docs]


def transform(docs, func, **kwargs):
    """
    Apply `func` to each token in each document of `docs` and return the result.

    :param docs: list of string tokens documents
    :param func: function to apply to each token; should accept a string as first arg. and optional `kwargs`
    :param kwargs: keyword arguments passed to `func`
    :return: list of processed documents
    """
    require_tokendocs(docs)

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

    :param docs: list of string tokens documents
    :return: list of processed documents
    """
    return transform(docs, str.lower)


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

    ignore_case_flag = dict(flags=re.IGNORECASE) if ignore_case else {}

    if match_type == 'exact':
        return np.char.lower(tokens) == pattern.lower() if ignore_case else tokens == pattern
    elif match_type == 'regex':
        if isinstance(pattern, str):
            pattern = re.compile(pattern, **ignore_case_flag)
        vecmatch = np.vectorize(lambda x: bool(pattern.search(x)))
        return vecmatch(tokens)
    else:
        if glob_method not in {'search', 'match'}:
            raise ValueError("`glob_method` must be one of `'search', 'match'`")

        if isinstance(pattern, str):
            pattern = globre.compile(pattern, **ignore_case_flag)

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

    :param patterns: a sequence of search patterns as excepted by :func:`~tmtoolkit.preprocess.token_match`
    :param tokens: a sequence of tokens to be used for matching
    :param kwargs: token matching options as passed to :func:`~tmtoolkit.preprocess.token_match`
    :return: list of NumPy arrays with subsequent indices into `tokens`
    """
    require_listlike(patterns)

    n_pat = len(patterns)

    if n_pat < 2:
        raise ValueError('`patterns` must contain at least two strings')

    n_tok = len(tokens)

    if n_tok == 0:
        return []

    if not isinstance(tokens, np.ndarray):
        require_listlike(tokens)
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

    if isinstance(split_chars, str):
        split_chars = (split_chars,)

    require_listlike_or_set(split_chars)

    if split_on_len is not None and split_on_len < 1:
        raise ValueError('`split_on_len` must be greater or equal 1')

    if split_on_casechange and not split_chars:
        t_parts = str_shapesplit(t, min_part_length=split_on_len)
    else:
        split_chars = set(split_chars)
        t_parts = str_multisplit(t, split_chars)

        if split_on_casechange:
            t_parts = flatten_list([str_shapesplit(p, min_part_length=split_on_len) for p in t_parts])

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
            parts_.extend(p.split(c))
        parts = parts_

    return parts


def str_shape(s, lower=0, upper=1, as_str=False):
    """
    Generate a sequence that reflects the "shape" of string `s`.

    :param s: input string
    :param lower: shape element marking a lower case letter
    :param upper: shape element marking an upper case letter
    :param as_str: join the sequence to a string
    :return: shape list or string if `as_str` is True
    """
    shape = [lower if c.islower() else upper for c in s]

    if as_str:
        if not isinstance(lower, str) or not isinstance(upper, str):
            shape = map(str, shape)

        return ''.join(shape)

    return shape


def str_shapesplit(s, shape=None, min_part_length=2):
    """
    Split string `s` according to its "shape" which is either given by `shape` (see
    :func:`~tmtoolkit.preprocess.str_shape`).

    :param s: string to split
    :param shape: list where 0 denotes a lower case character and 1 an upper case character; if `shape` is None,
                  it is computed via :func:`~tmtoolkit.preprocess.str_shape()`
    :param min_part_length: minimum length of a chunk (as long as ``len(s) >= min_part_length``)
    :return: list of substrings of `s`; returns ``['']`` if `s` is empty string
    """

    if not isinstance(s, str):
        raise ValueError('`s` must be string')

    if min_part_length is None:
        min_part_length = 2

    if min_part_length < 1:
        raise ValueError('`min_part_length` must be greater or equal 1')

    if not s:
        return ['']

    if shape is None:
        shape = str_shape(s)
    elif len(shape) != len(s):
        raise ValueError('`shape` must have same length as `s`')

    shapechange = np.abs(np.diff(shape, prepend=[shape[0]])).tolist()
    assert len(s) == len(shape) == len(shapechange)

    parts = []
    n = 0
    while shapechange and n < len(s):
        if n == 0:
            begin = 0
        else:
            begin = shapechange.index(1, n)

        try:
            offset = n + 1 if n == 0 and shape[0] == 0 else n + min_part_length
            end = shapechange.index(1, offset)
            #end = shapechange.index(1, n+min_part_length)
            n += end - begin
        except ValueError:
            end = None
            n = len(s)

        chunk = s[begin:end]

        if (parts and len(parts[-1]) >= min_part_length and len(chunk) >= min_part_length) or not parts:
            parts.append(chunk)
        else:
            parts[-1] += chunk

    return parts


#%% other functions


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


def require_tokendocs(docs, types=(list, np.ndarray),
                      error_msg='the argument must be a list of string token documents'):
    require_listlike(docs)

    if docs:
        first_doc = next(iter(docs))
        if not isinstance(first_doc, types):
            raise ValueError(error_msg)
