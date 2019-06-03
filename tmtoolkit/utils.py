import pickle
import re

import globre
import numpy as np
from deprecation import deprecated


#%% pickle / unpickle

def pickle_data(data, picklefile):
    """Helper function to pickle `data` in `picklefile`."""
    with open(picklefile, 'wb') as f:
        pickle.dump(data, f, protocol=2)


def unpickle_file(picklefile, **kwargs):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)


#%% arg type check functions


def require_types(x, valid_types):
    if all(isinstance(x, t) is False for t in valid_types):
        raise ValueError('requires one of those types:', str(valid_types))


def require_attrs(x, req_attrs):
    avail_attrs = dir(x)
    if any(a not in avail_attrs for a in req_attrs):
        raise ValueError('requires attributes:', str(req_attrs))


def require_listlike(x):
    require_types(x, (tuple, list))


def require_listlike_or_set(x):
    require_types(x, (set, tuple, list))


def require_dictlike(x):
    require_attrs(x, ('__len__', '__getitem__', '__setitem__', '__delitem__', '__iter__', '__contains__',
                      'items', 'keys', 'get'))


#%% NumPy array/matrices related helper functions


def empty_chararray():
    """Create empty NumPy character array"""
    return np.array([], dtype='<U1')


def mat2d_window_from_indices(mat, row_indices=None, col_indices=None, copy=False):
    """
    Select an area/"window" inside of a 2D array/matrix `mat` specified by either a sequence of
    row indices `row_indices` and/or a sequence of column indices `col_indices`.
    Returns the specified area as a *view* of the data if `copy` is False, else it will return a copy.
    """
    if not isinstance(mat, np.ndarray) or mat.ndim != 2:
        raise ValueError('`mat` must be a 2D NumPy array')

    if mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError('invalid shape for `mat`: %s' % str(mat.shape))

    if row_indices is None:
        row_indices = slice(None)   # a ":" slice
    elif len(row_indices) == 0:
        raise ValueError('`row_indices` must be non-empty')

    if col_indices is None:
        col_indices = slice(None)   # a ":" slice
    elif len(col_indices) == 0:
        raise ValueError('`col_indices` must be non-empty')

    view = mat[row_indices, :][:, col_indices]

    if copy:
        return view.copy()
    else:
        return view


def normalize_to_unit_range(values):
    """Bring a 1D NumPy array with at least two values in `values` to a linearly normalized range of [0, 1]."""
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise ValueError('`values` must be a 1D NumPy array')

    if len(values) < 2:
        raise ValueError('`values` must contain at least two values')

    min_ = np.min(values)
    max_ = np.max(values)
    range_ = max_ - min_

    if range_ == 0:
        raise ValueError('range of `values` is 0 -- cannot normalize')

    return (values - min_) / range_



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


#%% Token (character array / token ID array) handling


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
    :return: List of NumPy arrays with indices into `tokens`
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


@deprecated(deprecated_in='0.9.0', removed_in='0.10.0', details='Method was renamed to `remove_chars_in_tokens`.')
def remove_special_chars_in_tokens(tokens, special_chars):
    return remove_chars_in_tokens(tokens, special_chars)


def remove_chars_in_tokens(tokens, chars):
    if not chars:
        raise ValueError('`chars` must be a non-empty sequence')

    del_chars = str.maketrans('', '', ''.join(chars))

    return [t.translate(del_chars) for t in tokens]


def create_ngrams(tokens, n, join=True, join_str=' '):
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


#%% misc functions


def flatten_list(l):
    """
    Flatten a 2D sequence `l` to a 1D list that is returned.
    Although `return sum(l, [])` looks like a very nice one-liner, it turns out to be much much slower than what is
    implemented below.
    """
    flat = []
    for x in l:
        flat.extend(x)

    return flat


def greedy_partitioning(elems_dict, k, return_only_labels=False):
    """
    Implementation of greed partitioning algorithm as explained in https://stackoverflow.com/a/6670011
    for a dict `elems_dict` containing elements with label -> weight mapping. The elements are placed in
    `k` bins such that the difference of sums of weights in each bin is minimized. The algorithm
    does not always find the optimal solution.
    If `return_only_labels` is False, returns a list of `k` dicts with label -> weight mapping,
    else returns a list of `k` lists containing only the labels.
    """
    if k <= 0:
        raise ValueError('`k` must be at least 1')
    elif k == 1:
        return elems_dict
    elif k >= len(elems_dict):
        # if k is bigger than the number of elements, return `len(elems_dict)` bins with each
        # bin containing only a single element
        if return_only_labels:
            return [[k] for k in elems_dict.keys()]
        else:
            return [{k: v} for k, v in elems_dict.items()]

    sorted_elems = sorted(elems_dict.items(), key=lambda x: x[1], reverse=True)
    bins = [[sorted_elems.pop(0)] for _ in range(k)]
    bin_sums = [sum(x[1] for x in b) for b in bins]

    for pair in sorted_elems:
        argmin = min(enumerate(bin_sums), key=lambda x: x[1])[0]
        bins[argmin].append(pair)
        bin_sums[argmin] += pair[1]

    if return_only_labels:
        return [[x[1] for x in b] for b in bins]
    else:
        return [dict(b) for b in bins]


def argsort(seq):
    """Same as NumPy argsort but for Python lists"""
    return sorted(range(len(seq)), key=seq.__getitem__)
