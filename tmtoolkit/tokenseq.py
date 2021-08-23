import re
from typing import Union, List, Any, Sequence, Optional

import globre
import numpy as np


def token_lengths(tokens: Union[List[str], np.ndarray]) -> List[int]:
    """
    Token lengths (number of characters of each token) in `tokens`.

    :param tokens: list or NumPy array of string tokens
    :return: list of token lengths
    """
    return list(map(len, tokens))


def token_match_multi_pattern(search_tokens: Union[Any, Sequence[Any]], tokens: Union[List[str], np.ndarray],
                              match_type: str = 'exact', ignore_case=False, glob_method: str = 'match') -> np.ndarray:
    """
    Return a boolean NumPy array signaling matches between any pattern in `search_tokens` and `tokens`. Works the
    same as :func:`token_match`, but accepts multiple patterns as `search_tokens` argument.

    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param tokens: list or NumPy array of string tokens
    :param match_type: one of: 'exact', 'regex', 'glob'; if 'regex', `search_token` must be RE pattern; if `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre)
    :param ignore_case: if True, ignore case for matching
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's `re.match` or `re.search`)
    :return: 1D boolean NumPy array of length ``len(tokens)`` where elements signal matches
    """
    if not isinstance(search_tokens, (list, tuple, set)):
        search_tokens = [search_tokens]
    elif isinstance(search_tokens, (list, tuple, set)) and not search_tokens:
        raise ValueError('`search_tokens` must not be empty')

    matches = np.repeat(False, repeats=len(tokens))
    for pat in search_tokens:
        matches |= token_match(pat, tokens, match_type=match_type, ignore_case=ignore_case, glob_method=glob_method)

    return matches


def token_match(pattern: Any, tokens: Union[List[str], np.ndarray],
                match_type: str = 'exact', ignore_case=False, glob_method: str = 'match') -> np.ndarray:
    """
    Return a boolean NumPy array signaling matches between `pattern` and `tokens`. `pattern` will be
    compared with each element in sequence `tokens` either as exact equality (`match_type` is ``'exact'``) or
    regular expression (`match_type` is ``'regex'``) or glob pattern (`match_type` is ``'glob'``). For the last two
    options, `pattern` must be a string or compiled RE pattern, otherwise it can be of any type that allows equality
    checking.

    See :func:`token_match_multi_pattern` for a version of this function that accepts multiple search patterns.

    :param pattern: string or compiled RE pattern used for matching against `tokens`; when `match_type` is ``'exact'``,
                    `pattern` may be of any type that allows equality checking
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


def token_match_subsequent(patterns: Union[Any, list], tokens: List[str], **match_opts) -> List[np.ndarray]:
    """
    Using N patterns in `patterns`, return each tuple of N matching subsequent tokens from `tokens`. Excepts the same
    token matching options via `match_opts` as :func:`token_match`. The results are returned as list
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

    .. seealso:: :func:`token_match`

    :param patterns: a sequence of search patterns as excepted by :func:`token_match`
    :param tokens: a sequence of string tokens to be used for matching
    :param match_opts: token matching options as passed to :func:`token_match`
    :return: list of NumPy arrays with subsequent indices into `tokens`
    """
    n_pat = len(patterns)

    if n_pat < 2:
        raise ValueError('`patterns` must contain at least two strings')

    n_tok = len(tokens)

    if n_tok == 0:
        return []

    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens, dtype=str)

    # iterate through the patterns
    for i_pat, pat in enumerate(patterns):
        if i_pat == 0:   # initial matching on full token array
            next_indices = np.arange(n_tok)
        else:  # subsequent matching uses previous match indices + 1 to match on tokens right after the previous matches
            next_indices = match_indices + 1
            next_indices = next_indices[next_indices < n_tok]   # restrict maximum index

        # do the matching with the current subset of "tokens"
        pat_match = token_match(pat, tokens[next_indices], **match_opts)

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


def token_join_subsequent(tokens: Union[List[str], np.ndarray], matches: List[np.ndarray], glue: Optional[str] = '_',
                          return_glued=False, return_mask=False) -> Union[list, tuple]:
    """
    Select subsequent tokens as defined by list of indices `matches` (e.g. output of
    :func:`token_match_subsequent`) and join those by string `glue`. Return a list of tokens
    where the subsequent matches are replaced by the joint tokens.

    .. warning:: Only works correctly when matches contains indices of *subsequent* tokens.

    Example::

        token_glue_subsequent(['a', 'b', 'c', 'd', 'd', 'a', 'b', 'c'], [np.array([1, 2]), np.array([6, 7])])
        # ['a', 'b_c', 'd', 'd', 'a', 'b_c']

    .. seealso:: :func:`token_match_subsequent`

    :param tokens: a sequence of tokens
    :param matches: list of NumPy arrays with *subsequent* indices into `tokens` (e.g. output of
                    :func:`token_match_subsequent`)
    :param glue: string for joining the subsequent matches or None if no joint tokens but a None object should be placed
                 in the result list
    :param return_glued: if True, return also a list of joint tokens
    :param return_mask: if True, return also a binary NumPy array with the length of the input `tokens` list that masks
                        all joint tokens but the first one
    :return: either two-tuple, three-tuple or list depending on `return_glued` and `return_mask`
    """
    if return_glued and glue is None:
        raise ValueError('if `glue` is None, `return_glued` must be False')

    n_tok = len(tokens)

    if n_tok == 0 or not matches:
        if return_glued:
            if return_mask:
                return [], [], np.repeat(1, n_tok).astype('uint8')
            return [], []
        else:
            if return_mask:
                return [], np.repeat(1, n_tok).astype('uint8')
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
            if not return_mask:
                res.append(tokens[i_t])
            i_t += 1

    if return_mask:
        mask = np.repeat(1, n_tok).astype('uint8')
        try:
            set_zero_ind = np.unique(np.concatenate([m[1:] for m in matches]))
            mask[set_zero_ind] = 0
        except ValueError:
            pass  # ignore "zero-dimensional arrays cannot be concatenated"

        mask[np.array(list(start_ind.keys()))] = 2
        assert len(res) == np.sum(mask == 2)

        if return_glued:
            return res, glued, mask
        else:
            return res, mask

    if return_glued:
        return res, glued
    else:
        return res


def ngrams_from_tokenlist(tok: List[str], n: int, join=True, join_str=' ') -> List[Union[str, List[str]]]:
    if len(tok) == 0:
        ng = []
    else:
        if len(tok) < n:
            ng = [tok]
        else:
            ng = [[tok[i + j] for j in range(n)]
                  for i in range(len(tok) - n + 1)]

    if join:
        return list(map(lambda x: join_str.join(x), ng))
    else:
        return ng


def index_windows_around_matches(matches: np.ndarray, left: int, right: int,
                                 flatten=False, remove_overlaps=True) -> Union[List[List[int]], np.ndarray]:
    """
    Take a boolean 1D array `matches` of length N and generate an array of indices, where each occurrence of a True
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
            return np.array([], dtype=int)

        window_ind = np.concatenate(nested_ind)
        window_ind = window_ind[(window_ind >= 0) & (window_ind < len(matches))]

        if remove_overlaps:
            return np.sort(np.unique(window_ind))
        else:
            return window_ind
    else:
        return [w[(w >= 0) & (w < len(matches))] for w in nested_ind]
