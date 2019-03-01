"""
Preprocessing utility functions.
"""

from deprecation import deprecated
import numpy as np


def empty_chararray():
    return np.array([], dtype='<U1')


def tokens2ids(tok, return_counts=False):
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

    doc_tokids = np.split(all_tokids, np.cumsum(list(map(len, tok))))[:-1]

    if return_counts:
        return vocab, doc_tokids, vocab_counts
    else:
        return vocab, doc_tokids


def ids2tokens(vocab, tokids):
    return [vocab[ids] for ids in tokids]


def str_multisplit(s, split_chars):
    parts = [s]
    for c in split_chars:
        parts_ = []
        for p in parts:
            parts_.extend(p.split(c))
        parts = parts_

    return parts


def expand_compound_token(t, split_chars=('-',), split_on_len=2, split_on_casechange=False):
    #print('expand_compound_token', t)
    if not split_on_len and not split_on_casechange:
        raise ValueError('At least one of the arguments `split_on_len` and `split_on_casechange` must evaluate to True')

    if not any(isinstance(split_chars, type_) for type_ in (list, set, tuple)):
        split_chars = [split_chars]

    parts = []
    add = False   # signals if current part should be appended to previous part

    t_parts = [t]
    for c in split_chars:
        t_parts_ = []
        for t in t_parts:
            t_parts_.extend(t.split(c))
        t_parts = t_parts_

    for p in str_multisplit(t, split_chars):  # for each part p in compound token t
        if not p: continue  # skip empty part
        if add and parts:   # append current part p to previous part
            parts[-1] += p
        else:               # add p as separate token
            parts.append(p)

        if split_on_len:
            add = len(p) < split_on_len   # if p only consists of `split_on_len` characters -> append the next p to it

        if split_on_casechange:
            # alt. strategy: if p is all uppercase ("US", "E", etc.) -> append the next p to it
            add = add and p.isupper() if split_on_len else p.isupper()

    if add and len(parts) >= 2:
        parts = parts[:-2] + [parts[-2] + parts[-1]]

    return parts


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
