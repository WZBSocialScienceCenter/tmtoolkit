import re
from typing import Dict, Union, List, Optional, Any

import globre
import numpy as np
from spacy.vocab import Vocab
from spacy.tokens import Doc


def token_match(pattern: Any, tokens: Union[List[str], np.ndarray],
                match_type: str = 'exact', ignore_case=False, glob_method: str = 'match') -> np.ndarray:
    """
    Return a boolean NumPy array signaling matches between `pattern` and `tokens`. `pattern` will be
    compared with each element in sequence `tokens` either as exact equality (`match_type` is ``'exact'``) or
    regular expression (`match_type` is ``'regex'``) or glob pattern (`match_type` is ``'glob'``). For the last two
    options, `pattern` must be a string or compiled RE pattern, otherwise it can be of any type that allows equality
    checking.

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


def spacydoc_from_tokens_with_metadata(tokens_w_meta: Dict[str, List], label: str,
                                       vocab: Optional[Union[Vocab, List[str]]] = None):
    otherattrs = {}
    if 'pos' in tokens_w_meta:
        otherattrs['pos'] = tokens_w_meta['pos']
    if 'lemma' in tokens_w_meta:
        otherattrs['lemmas'] = tokens_w_meta['lemma']

    userdata = {k: v for k, v in tokens_w_meta.items() if k.startswith('meta_')}

    if 'mask' in tokens_w_meta:
        mask = tokens_w_meta['mask']
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
    else:
        mask = None

    return spacydoc_from_tokens(tokens_w_meta['token'], label=label, vocab=vocab,
                                spaces=tokens_w_meta['whitespace'],
                                mask=mask, otherattrs=otherattrs, userdata=userdata)


def spacydoc_from_tokens(tokens: List[str], label: str,
                         vocab: Optional[Union[Vocab, List[str]]] = None,
                         spaces: Optional[List[bool]] = None,
                         mask: Optional[np.ndarray] = None,
                         otherattrs: Optional[Dict[str, List[str]]] = None,
                         userdata: Optional[Dict[str, np.ndarray]] = None):
    """
    Create a new spaCy ``Doc`` document with tokens `tokens`.
    """
    from ._helpers import _init_spacy_doc

    # spaCy doesn't accept empty tokens
    nonempty_tok = np.array([len(t) > 0 for t in tokens])
    has_nonempty = np.sum(nonempty_tok) < len(tokens)

    if has_nonempty:
        tokens = np.asarray(tokens)[nonempty_tok].tolist()

    if vocab is None:
        vocab = Vocab(strings=set(tokens))
    elif not isinstance(vocab, Vocab):
        vocab = Vocab(strings=vocab)

    if spaces is not None:
        if has_nonempty:
            spaces = np.asarray(spaces)[nonempty_tok].tolist()
        assert len(spaces) == len(tokens), '`tokens` and `spaces` must have same length'

    if mask is not None:
        if has_nonempty:
            mask = mask[nonempty_tok]
        assert len(mask) == len(tokens), '`tokens` and `mask` must have same length'

    for attrs in (otherattrs, userdata):
        if attrs is not None:
            if has_nonempty:
                for k in attrs.keys():
                    if isinstance(attrs[k], np.ndarray):
                        attrs[k] = attrs[k][nonempty_tok]
                    else:
                        attrs[k] = np.asarray(attrs[k])[nonempty_tok].tolist()

            which = 'otherattrs' if attrs == otherattrs else 'userdata'
            for k, v in attrs.items():
                assert len(v) == len(tokens), f'all attributes in `{which}` must have the same length as `tokens`; ' \
                                              f'this failed for attribute {k}'

    new_doc = Doc(vocab, words=tokens, spaces=spaces, **(otherattrs or {}))
    assert len(new_doc) == len(tokens), 'created Doc object must have same length as `tokens`'

    _init_spacy_doc(new_doc, label, mask=mask, additional_attrs=userdata)

    return new_doc
