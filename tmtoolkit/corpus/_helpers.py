from typing import Dict, Union, List, Optional, Any, Sequence

import numpy as np
from spacy.tokens import Doc
from spacy.vocab import Vocab

from .._pd_dt_compat import FRAME_TYPE, pd_dt_colnames, pd_dt_frame_to_list
from ..tokenseq import token_match

from ._corpus import Corpus


def _corpus_from_tokens(corp: Corpus, tokens: Dict[str, Dict[str, list]],
                        doc_attr_names: Optional[Sequence] = None,
                        token_attr_names: Optional[Sequence] = None):
    """
    Create SpaCy docs from tokens (with doc/tokens attributes) for Corpus `corp`.

    Modifies `corp` in-place.
    """

    if doc_attr_names is None and token_attr_names is None:  # guess whether attribute is doc or token attr.
        doc_attr_names = set()
        token_attr_names = set()
        for tok in tokens.values():
            if isinstance(tok, dict):
                for k, v in tok.items():
                    if isinstance(v, (tuple, list, np.ndarray)):
                        token_attr_names.add(k)
                    else:
                        doc_attr_names.add(k)
            elif isinstance(tok, FRAME_TYPE):
                raise RuntimeError('cannot guess attribute level (i.e. document or token level attrib.) '
                                   'from datatables / dataframes')

    spacydocs = {}
    for label, tok in tokens.items():
        if isinstance(tok, (list, tuple)):                          # tokens alone (no attributes)
            doc = spacydoc_from_tokens(tok, label=label, vocab=corp.nlp.vocab)
        else:
            if isinstance(tok, FRAME_TYPE):  # each document is a datatable
                tok = {col: coldata for col, coldata in zip(pd_dt_colnames(tok), pd_dt_frame_to_list(tok))}
            elif not isinstance(tok, dict):
                raise ValueError(f'data for document `{label}` is of unknown type `{type(tok)}`')

            doc = spacydoc_from_tokens_with_attrdata(tok, label=label, vocab=corp.nlp.vocab,
                                                     doc_attr_names=doc_attr_names or (),
                                                     token_attr_names=token_attr_names or ())

        spacydocs[label] = doc

    corp.spacydocs = spacydocs
    if doc_attr_names:
        corp._doc_attrs_defaults = {k: None for k in doc_attr_names}        # cannot infer default value
        for k in doc_attr_names:
            if not Doc.has_extension(k):
                Doc.set_extension(k, default=None, force=True)
    if token_attr_names:
        corp._token_attrs_defaults = {k: None for k in token_attr_names}    # cannot infer default value


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


def _filtered_doc_tokens(doc: Doc, tokens_as_hashes=False, apply_filter=True):
    hashes = doc.user_data['processed']

    if apply_filter:
        hashes = hashes[doc.user_data['mask']]

    if tokens_as_hashes:
        return hashes
    else:
        return list(map(lambda hash: doc.vocab.strings[hash], hashes))


def _filtered_doc_token_attr(doc: Doc, attr: str, custom: Optional[bool] = None, stringified: Optional[bool] = None,
                             apply_filter=True, **kwargs):
    if custom is None:   # this means "auto" – we first check if `attr` is a custom attrib.
        custom = attr in doc.user_data.keys()

    if custom:
        if 'default' in kwargs and attr not in doc.user_data:
            n = np.sum(doc.user_data['mask']) if apply_filter else len(doc.user_data['mask'])
            return np.repeat(kwargs['default'], n)
        else:
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
                           match_type: str = 'exact', ignore_case=False, glob_method: str = 'match'):
    """
    Helper function to apply `token_match` with multiple patterns in `search_tokens` to `docs`.
    The matching results for each pattern in `search_tokens` are combined via logical OR.
    Returns a dict mapping keys in `tokens` to boolean arrays that signal the pattern matches for each token in each
    document.
    """

    # search tokens may be of any type (e.g. bool when matching against token attributes)
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


def _match_against(docs: Dict[str, Doc], by_attr: Optional[str] = None, **kwargs):
    """Return the list of values to match against in filtering functions."""
    if by_attr:
        return {lbl: _filtered_doc_token_attr(doc, attr=by_attr, **kwargs) for lbl, doc in docs.items()}
    else:
        return {lbl: _filtered_doc_tokens(doc) for lbl, doc in docs.items()}
