"""
Module for functions that work with text represented as *token sequences*, e.g. ``["A", "test", "document", "."]``
and single tokens (i.e. strings).

Tokens don't have to be represented as strings -- for many functions, they may also be token hashes (as integers).
Most functions also accept NumPy arrays instead of lists / tuples.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import itertools
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from functools import partial
from typing import Union, List, Any, Optional, Callable, Iterable, Dict, Sequence

import globre
import numpy as np
from bidict import bidict

from .utils import flatten_list
from .types import UnordCollection


#%% functions that operate on single tokens

def numbertoken_to_magnitude(numbertoken: str, char: str = '0', firstchar: str = '1', below_one: str = '0',
                             zero: str = '0', decimal_sep: str = '.', thousands_sep: str = ',',
                             drop_sign: bool = False) -> str:
    """
    Convert a string token `numbertoken` that represents a number (e.g. "13", "1.3" or "-1313") to a string token that
    represents the magnitude of that number by repeating `char` ("10", "1", "-1000" for the mentioned examples). A
    different first character can be set via `firstchar`. The sign can be dropped via `drop_sign`.

    If `numbertoken` cannot be converted to a float, an empty string is returned.

    :param numbertoken: token that represents a number
    :param char: character string used to represent single orders of magnitude
    :param firstchar: special character used for first character in the output
    :param below_one: special character used for numbers with absolute value below 1 (would otherwise return `''`)
    :param zero: if `numbertoken` evaluates to zero, return this string
    :param decimal_sep: decimal separator used in `numbertoken`; this is language-specific
    :param thousands_sep: thousands separator used in `numbertoken`; this is language-specific
    :param drop_sign: if True, drop the sign in number `numbertoken`, i.e. use absolute value
    :return: string that represents the magnitude of the input or an empty string
    """
    if decimal_sep != '.':
        numbertoken = numbertoken.replace(decimal_sep, '.')

    if thousands_sep:
        numbertoken = numbertoken.replace(thousands_sep, '')

    try:
        number = float(numbertoken)
    except ValueError:  # catches float conversion error
        return ''

    prefix = '-' if not drop_sign and number < 0 else ''
    abs_number = abs(number)

    if abs_number < 1:
        return prefix + below_one

    try:
        magn = math.floor(math.log10(abs_number)) + 1    # absolute magnitude, sign is discarded here
    except ValueError:  # catches domain error when taking log10(0)
        return zero

    if firstchar != char:
        return prefix + firstchar + char * (magn-1)
    else:
        return prefix + char * magn


def simplify_unicode_chars(token: str, method: str = 'icu', ascii_encoding_errors: str = 'ignore') -> str:
    """
    *Simplify* unicode characters in string `token`, i.e. remove diacritics, underlines and
    other marks. Requires `PyICU <https://pypi.org/project/PyICU/>`_ to be installed when using
    ``method="icu"``.

    :param docs: a Corpus object
    :param method: either ``"icu"`` which uses `PyICU <https://pypi.org/project/PyICU/>`_ for "proper"
                   simplification or ``"ascii"`` which tries to encode the characters as ASCII; the latter
                   is not recommended and will simply dismiss any characters that cannot be converted
                   to ASCII after decomposition
    :param ascii_encoding_errors: only used if `method` is ``"ascii"``; what to do when a character cannot be
                                  encoded as ASCII character; can be either ``"ignore"`` (default – replace by empty
                                  character), ``"replace"`` (replace by ``"???"``) or ``"strict"`` (raise a
                                  ``UnicodeEncodeError``)
    :return: simplyfied string
    """

    method = method.lower()
    if method == 'icu':
        try:
            from icu import UnicodeString, Transliterator, UTransDirection
        except ImportError:
            raise RuntimeError('package PyICU (https://pypi.org/project/PyICU/) must be installed to use this method')

        u = UnicodeString(token)
        trans = Transliterator.createInstance("NFD; [:M:] Remove; NFC", UTransDirection.FORWARD)
        trans.transliterate(u)
        return str(u)
    elif method == 'ascii':
        return unicodedata.normalize('NFKD', token).encode('ASCII', errors=ascii_encoding_errors).decode('utf-8')
    else:
        raise ValueError('`method` must be either "icu" or "ascii"')


#%% functions that operate on token sequences

def token_lengths(tokens: Union[Iterable[str], np.ndarray]) -> List[int]:
    """
    Token lengths (number of characters of each token) in `tokens`.

    :param tokens: list or NumPy array of string tokens
    :return: list of token lengths
    """
    return list(map(len, tokens))


def collapse_tokens(tokens: Union[Iterable[str], np.ndarray], collapse: Union[str, Iterable[str], np.ndarray] = ' ') \
        -> str:
    if isinstance(collapse, str):
        return collapse.join(tokens)
    else:
        if len(tokens) != len(collapse):
            raise ValueError('if `collapse` is given as sequence, it must have the same length as `tokens`')

        interleaved = itertools.chain(*zip(tokens, collapse))
        return ''.join(interleaved)


def pmi(x: np.ndarray, y: np.ndarray, xy: np.ndarray, n_total: Optional[int] = None, logfn: Callable = np.log,
        k: int = 1, normalize: bool = False) -> np.ndarray:
    """
    Calculate pointwise mutual information measure (PMI) either from probabilities p(x), p(y), p(x, y) given as `x`,
    `y`, `xy`, or from total counts `x`, `y`, `xy` and additionally `n_total`. Setting `k` > 1 gives PMI^k variants.
    Setting `normalized` to True gives normalized PMI (NPMI) as in [Bouma2009]_. See [RoleNadif2011]_ for a comparison
    of PMI variants.

    Probabilities should be such that ``p(x, y) <= min(p(x), p(y))``.

    .. [RoleNadif2011] Role, François & Nadif, Mohamed. (2011). Handling the Impact of Low Frequency Events on
                       Co-occurrence based Measures of Word Similarity - A Case Study of Pointwise Mutual Information.
    .. [Bouma2009] Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. Proceedings
                   of GSCL, 30, 31-40.

    :param x: probabilities p(x) or count of occurrence of x (interpreted as count if `n_total` is given)
    :param y: probabilities p(y) or count of occurrence of y (interpreted as count if `n_total` is given)
    :param xy: probabilities p(x, y) or count of occurrence of x *and* y (interpreted as count if `n_total` is given)
    :param n_total: if given, `x`, `y` and `xy` are interpreted as counts with `n_total` as size of the sample space
    :param logfn: logarithm function to use (default: ``np.log`` – natural logarithm)
    :param k: if `k` > 1, calculate PMI^k variant
    :param normalize: if True, normalize to range [-1, 1]; gives NPMI measure
    :return: array with same length as inputs containing (N)PMI measures for each input probability
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError('`k` must be a strictly positive integer')

    if k > 1 and normalize:
        raise ValueError('normalization is only implemented for standard PMI with `k=1`')

    if n_total is not None:
        if n_total < 1:
            raise ValueError('`n_total` must be strictly positive')
        x = x/n_total
        y = y/n_total
        xy = xy/n_total

    pmi_val = logfn(xy) - logfn(x * y)

    if k > 1:
        return pmi_val - (1-k) * logfn(xy)
    else:
        if normalize:
            return pmi_val / -logfn(xy)
        else:
            return pmi_val


npmi = partial(pmi, k=1, normalize=True)
pmi2 = partial(pmi, k=2, normalize=False)
pmi3 = partial(pmi, k=3, normalize=False)


def simple_collocation_counts(x: Optional[np.ndarray], y: Optional[np.ndarray], xy: np.ndarray, n_total: Optional[int]):
    """
    "Statistic" function that can be used in :func:`~token_collocations` and will simply return the number of
    collocations between tokens *x* and *y* passed as `xy`. Mainly useful for debugging purposes.

    :param x: unused
    :param y: unused
    :param xy: counts for collocations of *x* and *y*
    :param n_total: total number of tokens (strictly positive)
    :return: simply returns `xy`
    """
    return xy.astype(float)


def token_collocations(sentences: List[List[Union[str, int]]], threshold: Optional[float] = None,
                       min_count: int = 1, embed_tokens: Optional[UnordCollection] = None,
                       statistic: Callable = npmi, vocab_counts: Optional[Mapping] = None,
                       glue: Optional[str] = None, return_statistic: bool = True, rank: Optional[str] = 'desc',
                       tokens_as_hashes: bool = False, hashes2tokens: Optional[Union[Dict[int, str], bidict]] = None,
                       **statistic_kwargs) \
        -> List[Union[tuple, str]]:
    """
    Identify token collocations (frequently co-occurring token series) in a list of sentences of tokens given by
    `sentences`. Currently only supports bigram collocations.

    :param sentences: list of sentences containing lists of tokens; tokens can be items of any type if `glue` is None
    :param threshold: minimum statistic value for a collocation to enter the results; if None, results are not filtered
    :param min_count: ignore collocations with number of occurrences below this threshold
    :param embed_tokens: tokens that, if occurring inside an n-gram, are not counted; see :func:`token_ngrams`
    :param statistic: function to calculate the statistic measure from the token counts; use one of the
                      ``[n]pmi`` functions provided in this module or provide your own function which
                      must accept parameters ``x, y, xy, n_total``; see :func:`~pmi` for more information
    :param vocab_counts: pass already computed token type counts to prevent computing these again in this function
    :param glue: if not None, provide a string that is used to join the collocation tokens
    :param return_statistic: also return computed statistic
    :param rank: if not None, rank the results according to the computed statistic in ascending (``rank='asc'``) or
                 descending (``rank='desc'``) order
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
    :param hashes2tokens: if tokens are given as integer hashes, this table is used to generate textual representations
                          for the results
    :param statistic_kwargs: additional arguments passed to `statistic` function
    :return: list of tuples ``(collocation tokens, score)`` if `return_statistic` is True, otherwise only a list of
             collocations; collocations are either a string (if `glue` is given) or a tuple of strings
    """

    # TODO: extend this to accept parameter n for arbitrary n-gram collocations, not only bigrams;
    # requires implementing multivariate mutual information https://en.wikipedia.org/wiki/Interaction_information
    # or other measures
    # TODO: add more measures, esp. t-test
    # (see https://en.wikipedia.org/wiki/Collocation#Statistically_significant_collocation);
    # this requires an additional threshold comparison relation argument

    if min_count < 0:
        raise ValueError('`min_count` must be non-negative')

    n_tok = sum(len(sent) for sent in sentences)

    if n_tok < 2:       # can't possibly have any collocations with fewer than 2 tokens
        return []

    if vocab_counts is None:
        vocab_counts = Counter(flatten_list(sentences))

    ngramsize = 2
    bigrams = []
    for sent_tokens in sentences:
        if len(sent_tokens) >= ngramsize:
            bigrams.extend(token_ngrams(sent_tokens, n=ngramsize, join=False, embed_tokens=embed_tokens,
                                        keep_embed_tokens=False))

    if tokens_as_hashes:
        bigrams = np.array(bigrams, dtype='uint64')
    else:
        bigrams = np.array(bigrams, dtype='str')

    bigrams, n_bigrams = np.unique(bigrams, return_counts=True, axis=0)

    if min_count > 1:   # filter bigrams
        mask = n_bigrams >= min_count
        bigrams = bigrams[mask]
        n_bigrams = n_bigrams[mask]

    if len(n_bigrams) == 0:       # can't possibly have any collocations with no bigrams after filtering
        return []

    # first and last token of bigrams
    bg_first = bigrams[:, 0]
    bg_last = bigrams[:, 1]

    # num. of occurrences for first and last token of bigrams
    n_first = np.array([vocab_counts[t] for t in bg_first], dtype=n_bigrams.dtype)
    n_last = np.array([vocab_counts[t] for t in bg_last], dtype=n_bigrams.dtype)

    # apply scoring function
    scores = statistic(x=n_first, y=n_last, xy=n_bigrams, n_total=n_tok, **statistic_kwargs)
    assert len(scores) == len(bigrams), 'length of scores array must match number of unique bigrams'

    # build result
    res = []
    for bg, s in zip(bigrams, scores):
        if hashes2tokens is None:
            bg = tuple(bg)
        else:
            bg = tuple(hashes2tokens[h] for h in bg)

        if glue is not None:
            bg = glue.join(bg)

        if threshold is None or s >= threshold:
            res.append((bg, s))

    if rank in {'asc', 'desc'}:
        res = sorted(res, key=lambda x: x[1], reverse=rank == 'desc')

    if not return_statistic:
        if res:
            res = list(list(zip(*res))[0])
        else:
            return []

    return res


def token_match(pattern: Any, tokens: Union[List[str], np.ndarray],
                match_type: str = 'exact', ignore_case: bool = False, glob_method: str = 'match',
                inverse: bool = False) -> np.ndarray:
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
    :param inverse: invert the matching results
    :return: 1D boolean NumPy array of length ``len(tokens)`` where elements signal matches between `pattern` and the
             respective token from `tokens`
    """
    if match_type not in {'exact', 'regex', 'glob'}:
        raise ValueError("`match_type` must be one of `'exact', 'regex', 'glob'`")

    if len(tokens) == 0:
        return np.array([], dtype=bool)

    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens)

    ignore_case_flag = re.IGNORECASE if ignore_case else 0

    if match_type == 'exact':
        return np.char.lower(tokens) == pattern.lower() if ignore_case else tokens == pattern
    elif match_type == 'regex':
        if isinstance(pattern, str):
            pattern = re.compile(pattern, flags=ignore_case_flag)
        vecmatch = np.vectorize(lambda x: bool(pattern.search(x)))
        return vecmatch(tokens)
    else:
        if glob_method not in {'search', 'match'}:
            raise ValueError("`glob_method` must be one of `'search', 'match'`")

        if isinstance(pattern, str):
            # using separator " " instead of default seperator "/" since this cannot occur in a token;
            # also adding "EXACT" flag so that the pattern must match the whole token
            pattern = globre.compile(pattern, sep=' ', flags=ignore_case_flag|globre.EXACT)

        if glob_method == 'search':
            vecmatch = np.vectorize(lambda x: bool(pattern.search(x)))
        else:
            vecmatch = np.vectorize(lambda x: bool(pattern.match(x)))

        res = vecmatch(tokens) if len(tokens) > 0 else np.array([], dtype=bool)

        if inverse:
            return ~res
        else:
            return res


def token_match_multi_pattern(search_tokens: Any, tokens: Union[List[str], np.ndarray],
                              match_type: str = 'exact', ignore_case: bool = False, glob_method: str = 'match') \
        -> np.ndarray:
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


def token_match_subsequent(patterns: UnordCollection, tokens: Union[list, np.ndarray], **match_opts) \
        -> List[np.ndarray]:
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
    if not isinstance(patterns, (list, tuple, set)):
        raise ValueError('`patterns` must be a list, tuple or set')

    n_pat = len(patterns)

    if n_pat < 2:
        raise ValueError('`patterns` must contain at least two strings')

    n_tok = len(tokens)

    if n_tok == 0:
        return []

    if not isinstance(tokens, np.ndarray):  # required since we need multiple item indexing
        tokens = np.array(tokens)

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
                          tokens_dtype: Optional[Union[str, np.dtype]] = None, return_glued: bool = False,
                          return_mask: bool = False) -> Union[list, tuple]:
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
    :param glue: string for joining the subsequent matches or None to keep them as separate items in a list
    :param tokens_dtype: if tokens is not a NumPy array, it will be converted as such; use this dtype for the array
    :param return_glued: if True, return also a list of joint tokens
    :param return_mask: if True, return also a NumPy integer array with the length of the input `tokens` list that marks
                        the original input `tokens` in three ways: 0 means mask that original token, 1 means retain
                        that original token, 2 means replace original token by newly generated joint token;
                        if True, also only return newly generated joint subsequent tokens and *not* also the original
                        tokens
    :return: either two-tuple, three-tuple or list depending on `return_glued` and `return_mask`
    """
    if return_glued and glue is None:
        raise ValueError('if `glue` is None, `return_glued` must be False')

    if not isinstance(matches, list):
        raise ValueError('`matches` must be a list')

    n_tok = len(tokens)

    # handle empty token list or no matches
    if n_tok == 0 or (not matches and return_mask):
        if return_glued:
            if return_mask:
                return [], [], np.repeat(1, n_tok).astype('uint8')
            return [], []
        else:
            if return_mask:
                return [], np.repeat(1, n_tok).astype('uint8')
            return []

    if not isinstance(tokens, np.ndarray):  # we require an array for multi-indexing
        tokens = np.array(tokens, dtype=tokens_dtype)

    # map match subsequence start index to all indices of that subsequence
    start_ind = dict(zip(map(lambda x: x[0], matches), matches))
    res = []
    glued = []

    i_t = 0             # current token index
    subseq_until = 0    # token match subsequence until this index
    while i_t < n_tok:
        if i_t in start_ind:    # a token match subsequence starts
            seq = tokens[start_ind[i_t]]    # get the full subsequence
            t = seq.tolist() if glue is None else glue.join(seq)    # join using `glue` or keep separated (but as list)
            if return_glued:
                glued.append(t)
            res.append(t)
            subseq_until = i_t + len(seq)   # mark the end of the subsequence
        else:   # not a subsequence *start*
            # if we don't return the mask and we're not inside a subsequence, add this original token to the result list
            if not return_mask and i_t >= subseq_until:
                res.append(tokens[i_t])
        i_t += 1

    if return_mask:
        # generate a mask
        mask = np.repeat(1, n_tok).astype('uint8')   # 1 means use original token
        try:
            set_zero_ind = np.unique(np.concatenate([m[1:] for m in matches]))
            mask[set_zero_ind] = 0   # 0 means mask original token
        except ValueError:
            pass  # ignore "zero-dimensional arrays cannot be concatenated"

        mask[np.array(list(start_ind.keys()))] = 2   # 2 means use newly generated joint token
        assert len(res) == np.sum(mask == 2)

        if return_glued:
            return res, glued, mask
        else:
            return res, mask

    if return_glued:
        return res, glued
    else:
        return res


def token_ngrams(tokens: Sequence, n: int, join: bool = True, join_str: str = ' ',
                 ngram_container: Callable = list, embed_tokens: Optional[Iterable] = None,
                 keep_embed_tokens: bool = True) -> list:
    """
    Generate n-grams of length `n` from list of tokens `tokens`. Either join the n-grams when `join` is True
    using `join_str` so that a list of joined n-gram strings is returned or, if `join` is False, return a list
    of n-gram lists (or other sequences depending on `ngram_container`).
    For the latter option, the tokens in `tokens` don't have to be strings but can by of any type.

    Optionally pass a set/list/tuple `embed_tokens` which contains tokens that, if occurring inside an n-gram, are
    not counted. See for example how a trigram ``'bank of america'`` is generated when the token ``'of'``
    is set as `embed_tokens`, although we ask to generate bigrams:

    .. code-block:: text

        > token_ngrams("I visited the bank of america".split(), n=2)
        ['I visited', 'visited the', 'the bank', 'bank of', 'of america']
        > token_ngrams("I visited the bank of america".split(), n=2, embed_tokens={'of'})
        ['I visited', 'visited the', 'the bank', 'bank of america', 'of america']

    :param tokens: sequence of tokens; if `join` is True, this must be a list of strings
    :param n: size of the n-grams to generate
    :param join: if True, join n-grams by `join_str`
    :param join_str: string to join n-grams if `join` is True
    :param ngram_container: if `join` is False, use this function to create the n-gram sequences
    :param embed_tokens: tokens that, if occurring inside an n-gram, are not counted
    :param keep_embed_tokens: if True, keep embedded tokens in the result
    :return: list of joined n-gram strings or list of n-grams that are n-sized sequences
    """
    if n < 2:
        raise ValueError('`n` must be at least 2')

    if len(tokens) == 0:
        ng = []
    else:
        if len(tokens) < n:
            ng = [ngram_container(tokens)]
        else:
            if embed_tokens:
                ng = []
                i = 0
                while i < len(tokens) - n + 1:
                    stop = n   # original stop mark
                    g = []
                    j = 0

                    while j < stop and (len(g) < n or keep_embed_tokens):
                        t = tokens[i + j]
                        is_embed = t in embed_tokens
                        if not is_embed or keep_embed_tokens:
                            g.append(t)
                        if is_embed and i + stop < len(tokens):
                            stop += 1   # increase stop mark when the current token is an "embedded token"
                        j += 1

                    if len(g) == n or (keep_embed_tokens and len(g) >= n):
                        ng.append(ngram_container(g))

                    i += 1
                    if not keep_embed_tokens:
                        i += (stop - n)   # 1 plus number of skipped tokens
            else:  # faster approach when not using `embed_tokens`
                ng = [ngram_container(tokens[i + j] for j in range(n))
                      for i in range(len(tokens) - n + 1)]

    if join:
        return list(map(lambda x: join_str.join(x), ng))
    else:
        return ng


def index_windows_around_matches(matches: np.ndarray, left: int, right: int,
                                 flatten: bool = False, remove_overlaps: bool = True) \
        -> Union[List[List[int]], np.ndarray]:
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
