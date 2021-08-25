import re
from collections import Counter
from functools import partial
from typing import Union, List, Any, Sequence, Optional, Callable

import globre
import numpy as np

from tmtoolkit.utils import flatten_list


def token_lengths(tokens: Union[List[str], np.ndarray]) -> List[int]:
    """
    Token lengths (number of characters of each token) in `tokens`.

    :param tokens: list or NumPy array of string tokens
    :return: list of token lengths
    """
    return list(map(len, tokens))


def pmi(p_x: np.ndarray, p_y: np.ndarray, p_xy: np.ndarray, logfn: Callable = np.log, k: int = 1, normalize=False) \
        -> np.ndarray:
    """
    Calculate pointwise mutual information measure (PMI) from probabilities p(x), p(y) and p(x, y) given as `p_x`, `p_y`
    and `p_xy`, respectively. Setting `k` > 1 gives PMI^k variants. Setting `normalized` to True gives normalized
    PMI (NPMI) as in [Bouma2009]_. See [RoleNadif2011]_ for a comparison of PMI variants.

    Probabilities should be such that ``p(x, y) <= min(p(x), p(y))``.

    .. seealso:: Use :func:`~pmi_from_counts` to calculate this measure from raw counts.

    .. [RoleNadif2011] Role, François & Nadif, Mohamed. (2011). Handling the Impact of Low Frequency Events on
                       Co-occurrence based Measures of Word Similarity - A Case Study of Pointwise Mutual Information.
    .. [Bouma2009] Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. Proceedings
                   of GSCL, 30, 31-40.

    :param p_x: probabilities p(x)
    :param p_y: probabilities p(y)
    :param p_xy: probabilities p(x, y)
    :param logfn: logarithm function to use (default: ``np.log`` – natural logarithm)
    :param k: if `k` > 1, calculate PMI^k variant
    :param normalize: if True, normalize to range [-1, 1]; gives NPMI measure
    :return: array with same length as inputs containing (N)PMI measures for each input probability
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError('`k` must be a strictly positive integer')

    if k > 1 and normalize:
        raise ValueError('normalization is only implemented for standard PMI with `k=1`')

    pmi = logfn(p_xy / (p_x * p_y))

    if k > 1:
        return pmi - (1-k) * logfn(p_xy)
    else:
        if normalize:
            return pmi / -logfn(p_xy)
        else:
            return pmi


def pmi_from_counts(n_x: np.ndarray, n_y: np.ndarray, n_xy: np.ndarray, n_total: int, logfn: Callable = np.log,
                    k: int = 1, normalize=False) -> np.ndarray:
    """
    Calculate pointwise mutual information measure (PMI) as explained in :func:`~pmi`, but use raw counts instead
    of probabilities.

    :param n_x: counts for tokens *x*
    :param n_y: counts for tokens *y*
    :param n_xy: counts for collocations of *x* and *y*
    :param n_total: total number of tokens (strictly positive)
    :param logfn: logarithm function to use (default: ``np.log`` – natural logarithm)
    :param k: if `k` > 1, calculate PMI^k variant
    :param normalize: if True, normalize to range [-1, 1]; gives NPMI measure
    :return: array with same length as inputs containing (N)PMI measures for each input probability
    """
    if n_total < 1:
        raise ValueError('`n_total` must be strictly positive')
    return pmi(n_x/n_total, n_y/n_total, n_xy/n_total, logfn=logfn, k=k, normalize=normalize)


npmi = partial(pmi, k=1, normalize=True)
npmi_from_counts = partial(pmi_from_counts, k=1, normalize=True)
pmi2 = partial(pmi, k=2, normalize=False)
pmi2_from_counts = partial(pmi_from_counts, k=2, normalize=False)
pmi3 = partial(pmi, k=3, normalize=False)
pmi3_from_counts = partial(pmi_from_counts, k=3, normalize=False)


def simple_collocation_counts(n_x: np.ndarray, n_y: np.ndarray, n_xy: np.ndarray, n_total: int):
    """
    "Statistic" function that can be used in :func:`~token_collocations` and will simply return the number of
    collocations between tokens *x* and *y* passed as `n_xy`. Mainly useful for debugging purposes.

    :param n_x: unused
    :param n_y: unused
    :param n_xy: counts for collocations of *x* and *y*
    :param n_total: total number of tokens (strictly positive)
    :return: simply returns `n_xy`
    """
    return n_xy.astype(float)


def token_collocations(sentences: List[list], threshold: Optional[float] = None,
                       min_count: int = 1, embed_tokens: Optional[Union[set, tuple, list]] = None,
                       statistic: Callable = npmi_from_counts, vocab_counts: Optional[Counter] = None,
                       glue: Optional[str] = None, return_statistic=True, rank: Optional[str] = 'desc',
                       **statistic_kwargs) \
        -> List[Union[tuple, str]]:
    """
    Identify token collocations (frequently co-occurring token series) in a list of tokens given by `tokens`. Currently
    only supports bigram collocations.

    :param sentences: list of sentences containing lists of tokens; tokens can be items of any type if `glue` is None
    :param threshold: minimum statistic value for a collocation to enter the results; if None, results are not filtered
    :param min_count: ignore collocations with number of occurrences below this threshold
    :param embed_tokens: tokens that, if occurring inside an n-gram, are not counted; see :func:`token_ngrams`
    :param statistic: function to calculate the statistic measure from the token counts; use one of the
                      ``[n]pmi[2,3]_from_counts`` functions provided in this module or provide your own function which
                      must accept parameters ``n_x, n_y, n_xy, n_total``; see :func:`~pmi_from_counts` and :func:`~pmi`
                      for more information
    :param vocab_counts: pass already computed token type counts to prevent computing these again in this function
    :param glue: if not None, provide a string that is used to join the collocation tokens
    :param return_statistic: also return computed statistic
    :param rank: if not None, rank the results according to the computed statistic in ascending (``rank='asc'``) or
                 descending (``rank='desc'``) order
    :param statistic_kwargs: additional arguments passed to `statistic` function
    :return: list of tuples ``(collocation tokens, score)`` if `return_statistic` is True, otherwise only a list of
             collocations
    """

    # TODO: extend this to accept parameter n for arbitrary n-gram collocations, not only bigrams;
    # requires implementing multivariate mutual information https://en.wikipedia.org/wiki/Interaction_information
    # or other measures
    # TODO: add more measures, esp. t-test
    # (see https://en.wikipedia.org/wiki/Collocation#Statistically_significant_collocation);
    # this requires an additional threshold comparison relation argument

    tokens_flat = flatten_list(sentences)
    n_tok = len(tokens_flat)

    if vocab_counts is None:
        vocab_counts = Counter(tokens_flat)

    del tokens_flat

    # ngram_container must be tuple because they're hashable (needed for Counter)
    bigrams = [token_ngrams(sent_tokens, n=2, join=False, ngram_container=tuple, embed_tokens=embed_tokens)
               for sent_tokens in sentences]
    del sentences

    bg_counts = Counter(flatten_list(bigrams))
    del bigrams
    if min_count > 1:
        bg_counts = {bg: count for bg, count in bg_counts.items() if count >= min_count}

    # unigram vocabulary as list
    vocab = list(vocab_counts.keys())       #vocab = np.array(list(vocab_counts.keys()))
    # counts for token types in vocab as array
    n_vocab = np.fromiter(vocab_counts.values(), dtype='uint32', count=len(vocab_counts))

    # bigram counts as array
    n_bigrams = np.fromiter(bg_counts.values(), dtype='uint32', count=len(bg_counts))

    # first and last token in bigrams -- because of `embed_tokens` we may actually have more than two tokens per bigram
    bg_first, bg_last = zip(*((bg[0], bg[-1]) for bg in bg_counts.keys()))

    # token counts for first and last tokens in bigrams
    # alternative via broadcasting (but probably more memory intensive):
    # np.where(vocab[:, np.newaxis] == bg_first)[0]
    n_first = n_vocab[[vocab.index(t) for t in bg_first]]
    n_last = n_vocab[[vocab.index(t) for t in bg_last]]

    # apply scoring function
    scores = statistic(n_x=n_first, n_y=n_last, n_xy=n_bigrams, n_total=n_tok, **statistic_kwargs)
    assert len(scores) == len(bg_counts), 'length of scores array must match number of unique bigrams'

    # build result
    res = []
    for bg, s in zip(bg_counts.keys(), scores):
        if glue is not None:
            bg = glue.join(bg)

        if threshold is None or s >= threshold:
            res.append((bg, s))

    if rank in {'asc', 'desc'}:
        res = sorted(res, key=lambda x: x[1], reverse=rank == 'desc')

    if not return_statistic:
        res = list(zip(*res))[0]

    return res


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


def token_ngrams(tokens: list, n: int, join=True, join_str: str = ' ', ngram_container: Callable = list,
                 embed_tokens: Optional[Union[set, list, tuple]] = None) -> list:
    """
    Generate n-grams of length `n` from list of tokens `tokens`. Either join the n-grams when `join` is True
    using `join_str` so that a list of joined n-gram strings is returned or, if `join` is False, return a list
    of n-gram lists (or other sequences depending on `ngram_container`).
    For the latter option, the tokens in `tokens` don't have to be strings but can by of any type.

    Optionally pass a set/list/tuple `embed_tokens` which contains tokens that, if occurring inside an n-gram, are
    not counted. See for example how a trigram ``'bank of america'`` is generated when the token ``'of'``
    is set as `embed_tokens`, although we ask to generate bigrams:

    .. code-block:: text

        > ngrams_from_tokenlist("I visited the bank of america".split(), n=2)
        ['I visited', 'visited the', 'the bank', 'bank of', 'of america']
        > ngrams_from_tokenlist("I visited the bank of america".split(), n=2, embed_tokens={'of'})
        ['I visited', 'visited the', 'the bank', 'bank of america', 'of america']

    :param tokens: list of tokens; if `join` is True, this must be a list of strings
    :param n: size of the n-grams to generate
    :param join: if True, join n-grams by `join_str`
    :param join_str: string to join n-grams if `join` is True
    :param ngram_container: if `join` is False, use this function to create the n-gram sequences
    :param embed_tokens: tokens that, if occurring inside an n-gram, are not counted
    :return: list of joined n-gram strings or list of n-grams that are n-sized sequences
    """
    if len(tokens) == 0:
        ng = []
    else:
        if len(tokens) < n:
            ng = [tokens]
        else:
            if embed_tokens:
                ng = []
                for i in range(len(tokens) - n + 1):
                    j = 0
                    stop = n   # original stop mark
                    g = []
                    while j < stop:
                        t = tokens[i + j]
                        g.append(t)
                        if t in embed_tokens and i > 0 and i + stop < len(tokens):
                            stop += 1   # increase stop mark when the current token is an "embedded token"
                        j += 1
                    ng.append(ngram_container(g))
            else:  # faster approach when not using `embed_tokens`
                ng = [ngram_container(tokens[i + j] for j in range(n))
                      for i in range(len(tokens) - n + 1)]

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
