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
        raise ValueError('requires type:', str(valid_types))


def require_attrs(x, req_attrs):
    avail_attrs = dir(x)
    if any(a not in avail_attrs for a in req_attrs):
        raise ValueError('requires attributes:', str(req_attrs))


def require_listlike(x):
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
        raise ValueError('invalid shape for `mat`: %s' % mat.shape)

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


def simplified_pos(pos, tagset=None):
    """
    Return a simplified POS tag for a full POS tag `pos` belonging to a tagset `tagset`. By default the WordNet
    tagset is assumed.
    Does the following conversion by default:
    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all ADJ... (adjective) tags to 'ADJ'
    - all ADV... (adverb) tags to 'ADV'
    - all other to None
    Does the following conversion by with `tagset=='penn'`:
    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all JJ... (adjective) tags to 'ADJ'
    - all RB... (adverb) tags to 'ADV'
    - all other to None
    """
    if tagset == 'penn':
        if pos.startswith('N') or pos.startswith('V'):
            return pos[0]
        elif pos.startswith('JJ'):
            return 'ADJ'
        elif pos.startswith('RB'):
            return 'ADV'
        else:
            return None
    else:   # default: WordNet, STTS or unknown
        if pos.startswith('N') or pos.startswith('V'):
            return pos[0]
        elif pos.startswith('ADJ') or pos.startswith('ADV'):
            return pos[:3]
        else:
            return None


#%% Token (character array / token ID array) handling


def token_match(pattern, tokens, match_type='exact', ignore_case=False, glob_method='match'):
    """
    Return a boolean NumPy array signaling matches between `pattern` and `tokens`. `pattern` is a string that will be
    compared with each element in sequence `tokens` either as exact string equality (`match_type` is 'exact') or
    regular expression (`match_type` is 'regex') or glob pattern (`match_type` is 'glob').
    """
    if match_type not in {'exact', 'regex', 'glob'}:
        raise ValueError("`match_type` must be one of `'exact', 'regex', 'glob'`")

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


def make_vocab_unique_and_update_token_ids(vocab, tokids):
    """
    Make a vocabulary `vocab` with potentially repeated terms unique, which requires a remapping of IDs in list of
    token documents `tokids` (a list of NumPy arrays with token IDs). The remapping is applied and the function returns
    a tuple `new_vocab, new_tokids`. If `vocab` is already unique, `vocab` and `tokids` are returned unchanged.

    This function is useful to apply when your vocab might not be unique after applying a transformation to it. E.g.:

    ```
    vocab = np.array(['A', 'a', 'b'])
    vocab_lower = np.char.lower(vocab)  # -> results in ['a', 'a', 'b'], which is not a vocab of unique terms any more
    ```

    `tokids` are documents with token IDs into `vocab` and `vocab_lower`, e.g. `[[2, 2, 0, 1]]` which is
    `[['b', 'b', 'A', 'a']]` into `vocab` or `[['b', 'b', 'a', 'a']]` into `vocab_lower`
    We now remove the duplicate terms in `vocab_lower` and update `tokids` so that it contains token IDs into
    a newly constructed `new_vocab` which is unique:

    ```
    new_vocab, new_tokid = make_vocab_unique_and_update_token_ids(vocab_lower, tokids)
    ```

    `new_vocab` is now `['a', 'b']` and `new_tokids` is `[[1, 1, 0, 0]]`. `new_tokids` reflects the change in the
    vocab.
    """
    new_vocab, ind, inv_ind = np.unique(vocab, return_index=True, return_inverse=True)
    n_old_vocab = len(vocab)
    n_new_vocab = len(new_vocab)

    if n_old_vocab != n_new_vocab:   # vocab was not unique before, update token IDs
        mapper = dict(zip(ind, np.arange(n_new_vocab)))         # recoding of existing IDs
        rm_ind = np.setdiff1d(np.arange(len(vocab)), ind)       # removed IDs
        mapper.update(dict(zip(rm_ind, inv_ind[rm_ind])))       # map removed IDs to new IDs

        def replace(val):
            return mapper[val]
        replace = np.vectorize(replace)

        if not isinstance(next(iter(tokids)), np.ndarray):
            raise ValueError('`tokids` must be a sequence of NumPy arrays')

        new_tokids = [replace(ids) if len(ids) > 0 else ids for ids in tokids]
    else:    # vocab was already unique, don't change anything
        new_vocab = vocab
        new_tokids = tokids

    return new_vocab, new_tokids


def str_multisplit(s, split_chars):
    if not isinstance(s, (str, bytes)):
        raise ValueError('`s` must be of type `str` or `bytes`')

    require_listlike(split_chars)

    split_chars = set(split_chars)
    parts = [s]
    for c in split_chars:
        parts_ = []
        for p in parts:
            parts_.extend(p.split(c))
        parts = parts_

    return parts


def expand_compound_token(t, split_chars=('-',), split_on_len=2, split_on_casechange=False):
    if not split_on_len and not split_on_casechange:
        raise ValueError('At least one of the arguments `split_on_len` and `split_on_casechange` must evaluate to True')

    if isinstance(split_chars, str):
        split_chars = (split_chars,)

    require_listlike(split_chars)

    split_chars = set(split_chars)

    parts = []
    add = False   # signals if current part should be appended to previous part

    for p in str_multisplit(t, split_chars):  # for each part p in compound token t
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

    return parts or [t]    # if parts is empty, return unchanged input


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
    """Flatten a 2D sequence `l` to a 1D list that is returned"""
    return sum(l, [])


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
