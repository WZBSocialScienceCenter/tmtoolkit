import re
from typing import Dict, Union, List, Optional, Any, Sequence

import globre
import numpy as np
from spacy.vocab import Vocab
from spacy.tokens import Doc


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


def spacydoc_from_tokens_with_attrdata(tokens_w_attr: Dict[str, List], label: str,
                                       vocab: Optional[Union[Vocab, List[str]]] = None,
                                       doc_attr_names: Sequence = (),
                                       token_attr_names: Sequence = ()):
    spacytokenattrs = {}
    if 'pos' in tokens_w_attr:
        spacytokenattrs['pos'] = tokens_w_attr['pos']
    if 'lemma' in tokens_w_attr:
        spacytokenattrs['lemmas'] = tokens_w_attr['lemma']

    tokenattrs = {k: tokens_w_attr[k] for k in token_attr_names}
    docattrs = {k: tokens_w_attr[k] for k in doc_attr_names}

    if 'mask' in tokens_w_attr:
        mask = tokens_w_attr['mask']
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
    else:
        mask = None

    return spacydoc_from_tokens(tokens_w_attr['token'], label=label, vocab=vocab,
                                spaces=tokens_w_attr['whitespace'], mask=mask,
                                docattrs=docattrs,
                                spacytokenattrs=spacytokenattrs,
                                tokenattrs=tokenattrs)


def spacydoc_from_tokens(tokens: List[str], label: str,
                         vocab: Optional[Union[Vocab, List[str]]] = None,
                         spaces: Optional[List[bool]] = None,
                         mask: Optional[np.ndarray] = None,
                         docattrs: Optional[Dict[str, np.ndarray]] = None,
                         spacytokenattrs: Optional[Dict[str, List[str]]] = None,
                         tokenattrs: Optional[Dict[str, np.ndarray]] = None):
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

    # prepare token attributes
    for attrs in (spacytokenattrs, tokenattrs):
        if attrs is not None:
            if has_nonempty:
                for k in attrs.keys():
                    if isinstance(attrs[k], np.ndarray):
                        attrs[k] = attrs[k][nonempty_tok]
                    else:
                        attrs[k] = np.asarray(attrs[k])[nonempty_tok].tolist()

            # check length
            which = 'spacytokenattrs' if attrs == spacytokenattrs else 'tokenattrs'
            for k, v in attrs.items():
                assert len(v) == len(tokens), f'all attributes in `{which}` must have the same length as `tokens`; ' \
                                              f'this failed for attribute {k}'

    # create new Doc object
    new_doc = Doc(vocab, words=tokens, spaces=spaces, **(spacytokenattrs or {}))
    assert len(new_doc) == len(tokens), 'created Doc object must have same length as `tokens`'

    # set initial attributes / token attributes
    _init_spacy_doc(new_doc, label, mask=mask, additional_attrs=tokenattrs)

    # set additional document attributes
    for k, v in docattrs.items():
        if isinstance(v, (np.ndarray, list, tuple)):
            reduced = set(v)
            assert len(reduced) == 1, f'value of document attribute "{k}" is not a single scalar: "{reduced}"'
            v = reduced.pop()

        setattr(new_doc._, k, v)

    return new_doc


def make_index_window_around_matches(matches: np.ndarray, left: int, right: int, flatten=False, remove_overlaps=True):
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
