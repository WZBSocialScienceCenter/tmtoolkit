from typing import Dict, Union, List, Optional, Any

import numpy as np
from spacy.tokens import Doc

from .._pd_dt_compat import FRAME_TYPE, pd_dt_colnames, pd_dt_frame_to_list

from ._corpus import Corpus
from ._tokenfuncs import token_match, spacydoc_from_tokens, spacydoc_from_tokens_with_metadata


def _corpus_from_tokens_metadata(corp: Corpus, tokens: Dict[str, Dict[str, list]]):  # TODO: also handle document attributes
    """
    Create SpaCy docs from tokens (with metadata) for Corpus `corp`. Modifies `corp` in-place.
    """
    spacydocs = {}
    for label, tok in tokens.items():
        if isinstance(tok, (list, tuple)):                          # tokens alone (no metadata)
            doc = spacydoc_from_tokens(tok, label=label, vocab=corp.nlp.vocab)
        else:
            if isinstance(tok, FRAME_TYPE):  # each document is a datatable
                tok = {col: coldata for col, coldata in zip(pd_dt_colnames(tok), pd_dt_frame_to_list(tok))}
            elif not isinstance(tok, dict):
                raise ValueError(f'data for document `{label}` is of unknown type `{type(tok)}`')

            doc = spacydoc_from_tokens_with_metadata(tok, label=label, vocab=corp.nlp.vocab)

        spacydocs[label] = doc

    corp.spacydocs = spacydocs


def _init_spacy_doc(doc: Doc, doc_label: str,
                    mask: Optional[np.ndarray] = None,
                    additional_attrs: Optional[Dict[str, Union[list, tuple, np.ndarray, int, float, str]]] = None):
    n = len(doc)
    doc._.label = doc_label
    if mask is None:
        doc.user_data['mask'] = np.repeat(True, n)
    else:
        doc.user_data['mask'] = mask

    doc.user_data['processed'] = np.fromiter((t.orth for t in doc), dtype='uint64', count=n)

    if additional_attrs:
        for k, default in additional_attrs.items():
            # default can be sequence (list, tuple or array) ...
            if isinstance(default, (list, tuple)):
                v = np.array(default)
            elif isinstance(default, np.ndarray):
                v = default
            else:  # ... or single value -> repeat this value to fill the array
                v = np.repeat(default, n)
            assert len(v) == n, 'user data array must have the same length as the tokens'
            doc.user_data[k] = v


def _filtered_doc_tokens(doc: Doc, tokens_as_hashes=False, apply_filter=True):
    hashes = doc.user_data['processed']

    if apply_filter:
        hashes = hashes[doc.user_data['mask']]

    if tokens_as_hashes:
        return hashes
    else:
        return list(map(lambda hash: doc.vocab.strings[hash], hashes))


def _filtered_doc_attr(doc: Doc, attr: str, custom: Optional[bool] = None, stringified: Optional[bool] = None,
                       apply_filter=True):
    if custom is None:   # this means "auto" – we first check if `attr` is a custom attrib.
        custom = attr in doc.user_data.keys()

    if custom:
        res = doc.user_data[attr]
        if apply_filter:
            return res[doc.user_data['mask']]
        else:
            return res
    else:
        if stringified is None:  # this means "auto" – we first check if a stringified attr. exists,
                                 # if not we try the original attr. name
            getattrfn = lambda t, a: getattr(t, a + '_') if hasattr(t, a + '_') else getattr(t, a)
        else:
            if stringified is True:
                attr += '_'
            getattrfn = getattr

        if apply_filter:
            return [getattrfn(t, attr) for t, m in zip(doc, doc.user_data['mask']) if m]
        else:
            return [getattrfn(t, attr) for t in doc]


def _token_pattern_matches(tokens: Dict[str, List[Any]], search_tokens: Union[Any, List[Any]],
                           match_type: str, ignore_case: bool, glob_method: str):
    """
    Helper function to apply `token_match` with multiple patterns in `search_tokens` to `docs`.
    The matching results for each pattern in `search_tokens` are combined via logical OR.
    Returns a list of length `docs` containing boolean arrays that signal the pattern matches for each token in each
    document.
    """

    # search tokens may be of any type (e.g. bool when matching against token meta data)
    if not isinstance(search_tokens, (list, tuple, set)):
        search_tokens = [search_tokens]
    elif isinstance(search_tokens, (list, tuple, set)) and not search_tokens:
        raise ValueError('`search_tokens` must not be empty')

    matches = [np.repeat(False, repeats=len(dtok)) for dtok in tokens.values()]

    for dtok, dmatches in zip(tokens.values(), matches):
        for pat in search_tokens:
            dmatches |= token_match(pat, dtok, match_type=match_type, ignore_case=ignore_case, glob_method=glob_method)

    return dict(zip(tokens.keys(), matches))


def _apply_matches_array(docs: Corpus, matches: Dict[str, np.ndarray] = None, invert=False):
    assert set(docs.keys()) == set(matches.keys()), 'the document labels in `matches` and `docs` must match'

    if invert:
        matches = [~m for m in matches]

    assert len(matches) == len(docs), '`matches` and `docs` must have same length'

    # simply set the new filter mask to previously unfiltered elements; changes document masks in-place
    for lbl, mask in matches.items():
        doc = docs.spacydocs[lbl]
        assert len(mask) == sum(doc.user_data['mask']), \
            'length of matches mask must equal the number of unmasked tokens in the document'
        doc.user_data['mask'] = _ensure_writable_array(doc.user_data['mask'])
        doc.user_data['mask'][doc.user_data['mask']] = mask


def _ensure_writable_array(arr: np.ndarray) -> np.ndarray:
    if not arr.flags.writeable:
        return np.copy(arr)
    else:
        return arr


def _check_filter_args(**kwargs):
    if 'match_type' in kwargs and kwargs['match_type'] not in {'exact', 'regex', 'glob'}:
        raise ValueError("`match_type` must be one of `'exact', 'regex', 'glob'`")

    if 'glob_method' in kwargs and kwargs['glob_method'] not in {'search', 'match'}:
        raise ValueError("`glob_method` must be one of `'search', 'match'`")


def _match_against(docs: Dict[str, Doc], by_meta: Optional[str] = None):
    """Return the list of values to match against in filtering functions."""
    if by_meta:
        return {lbl: _filtered_doc_attr(doc, attr=by_meta) for lbl, doc in docs.items()}
    else:
        return {lbl: _filtered_doc_tokens(doc) for lbl, doc in docs.items()}
