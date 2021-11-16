"""
Helper functions for text processing in the :mod:`tmtoolkit.corpus` module.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""
from collections import defaultdict
from typing import Dict, Union, List, Optional, Any, Sequence

import numpy as np
import pandas as pd
from spacy.tokens import Doc
from spacy.vocab import Vocab

from ._common import Document
from ..tokenseq import token_match
from ..types import OrdCollection, UnordCollection, UnordStrCollection
from ..utils import empty_chararray, flatten_list

from ._corpus import Corpus


#%% public functions for creating SpaCy Doc objects

def spacydoc_from_tokens_with_attrdata(tokens_w_attr: Dict[str, list],
                                       label: str,
                                       vocab: Optional[Union[Vocab, List[str]]] = None,
                                       sent_borders: Optional[np.array] = None,
                                       doc_attr_names: UnordCollection = (),
                                       token_attr_names: UnordCollection = ()) -> Doc:
    """
    Create a `SpaCy Doc <https://spacy.io/api/doc/>`_ object from a dict of tokens with document/token
    attributes.

    :param tokens_w_attr: dict with token attributes; must at least contain the attributes "token" and "whitespace"
    :param label: document label
    :param vocab: optional `SpaCy Vocab <https://spacy.io/api/vocab>`_ object or list of token type strings
    :param sent_borders: optional integer list or integer array that denotes the indices of tokens that mark the
                         beginning of a sentence; should always include 0 as first index for non-empty documents
    :param doc_attr_names: document attribute names
    :param token_attr_names: token attribute names
    :return: `SpaCy Doc <https://spacy.io/api/doc/>`_ object created from the passed data
    """
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

    return spacydoc_from_tokens(tokens_w_attr['token'], label=label, vocab=vocab, sent_borders=sent_borders,
                                spaces=tokens_w_attr['whitespace'], mask=mask,
                                docattrs=docattrs,
                                spacytokenattrs=spacytokenattrs,
                                tokenattrs=tokenattrs)


def spacydoc_from_tokens(tokens: List[str],
                         label: str,
                         vocab: Optional[Union[Vocab, List[str]]] = None,
                         sent_borders: Optional[Union[List[int], np.ndarray]] = None,
                         spaces: Optional[List[bool]] = None,
                         mask: Optional[np.ndarray] = None,
                         docattrs: Optional[Dict[str, Any]] = None,
                         spacytokenattrs: Optional[Dict[str, list]] = None,
                         tokenattrs: Optional[Dict[str, OrdCollection]] = None) -> Doc:
    """
    Create a `SpaCy Doc <https://spacy.io/api/doc/>`_ object from a list of tokens and optional attributes.

    :param tokens: list of tokens
    :param label: document label
    :param vocab: optional `SpaCy Vocab <https://spacy.io/api/vocab>`_ object or list of token type strings
    :param sent_borders: optional integer list or integer array that denotes the indices of tokens that mark the
                         beginning of a sentence; should always include 0 as first index for non-empty documents
    :param spaces: optional boolean list denoting spaces after tokens
    :param mask: optional boolean array defining the token mask
    :param docattrs: optional document attributes as dict mapping attribute name to attribute value; the
                     attribute value can either be a scalar value or an array/list/tuple which must contain
                     only one unique value
    :param spacytokenattrs: optional dict of token attributes that are set as
                            `SpaCy Token <https://spacy.io/api/token/>`_ attributes
    :param tokenattrs: optional dict of token attributes that are set as custom attributes
    :return: `SpaCy Doc <https://spacy.io/api/doc/>`_ object created from the passed data
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
                                              f'this failed for attribute "{k}"'

    spacytokenattrs = spacytokenattrs or {}
    if 'sent_starts' not in spacytokenattrs and sent_borders is not None:
        if tokens:
            sent_starts = np.repeat(False, len(tokens))
            sent_starts[sent_borders] = True
            sent_starts = sent_starts.tolist()
        else:
            sent_starts = []

        spacytokenattrs['sent_starts'] = sent_starts

    # create new Doc object
    new_doc = Doc(vocab, words=tokens, spaces=spaces, **spacytokenattrs)
    assert len(new_doc) == len(tokens), 'created Doc object must have same length as `tokens`'

    # set initial attributes / token attributes
    _init_spacy_doc(new_doc, label, mask=mask, additional_attrs=tokenattrs, skip_sent_borders=sent_borders is None)

    # set additional document attributes
    if docattrs:
        for k, v in docattrs.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                reduced = set(v)
                assert len(reduced) == 1, f'value of document attribute "{k}" is not a single scalar: "{reduced}"'
                v = reduced.pop()

            setattr(new_doc._, k, v)

    return new_doc


#%% various internal helper functions


def _corpus_from_tokens(corp: Corpus, tokens: Dict[str, Dict[str, list]],
                        sentences: bool,
                        doc_attr_names: Optional[UnordStrCollection] = None,
                        token_attr_names: Optional[UnordStrCollection] = None):
    """
    Create SpaCy docs from tokens (with doc/tokens attributes) for Corpus `corp`.

    Modifies `corp` in-place.
    """

    if doc_attr_names is None or token_attr_names is None:  # guess whether attribute is doc or token attr.
        new_doc_attr_names = set()
        new_token_attr_names = set()
        for sents in tokens.values():
            if not sentences:
                sents = [sents]

            for tok in sents:
                if isinstance(tok, dict):
                    for k, v in tok.items():
                        if isinstance(v, (tuple, list, np.ndarray)):
                            if token_attr_names is None:
                                new_token_attr_names.add(k)
                        else:
                            if doc_attr_names is None:
                                new_doc_attr_names.add(k)
                elif isinstance(tok, pd.DataFrame):
                    raise RuntimeError('cannot guess attribute level (i.e. document or token level attrib.) '
                                       'from dataframes')

        if doc_attr_names is None:
            doc_attr_names = new_doc_attr_names
        if token_attr_names is None:
            token_attr_names = new_token_attr_names

    spacydocs = {}
    for label, sents in tokens.items():
        if sentences:
            if len(sents) > 0:
                if isinstance(sents, pd.DataFrame):
                    tok = sents
                    sent_borders = np.cumsum(np.array([0] + tok.groupby('sent').count()['token'].to_list()[:-1],
                                                      dtype='uint32'))
                else:
                    first_sent = next(iter(sents))
                    if isinstance(first_sent, (list, tuple)):                  # tokens alone (no attributes)
                        tok = flatten_list(sents)
                        sent_borders = np.cumsum([0] + list(map(len, sents))[:-1], dtype='uint32')
                    elif isinstance(first_sent, dict):
                        tok = defaultdict(list)
                        sent_borders = [0]
                        for sent in sents:
                            for k, v in sent.items():
                                if k == 'token' and len(v) > 0:
                                    sent_borders.append(sent_borders[-1] + len(v))

                                if isinstance(v, (list, tuple, np.ndarray)):
                                    if len(v) > 0:
                                        tok[k].extend(v)
                                else:
                                    if k not in tok:
                                        tok[k] = v
                                    else:
                                        assert tok[k] == v, \
                                            'expecting non-sequence value to be constant across sentences'
                        sent_borders = np.array(sent_borders[:-1], dtype='uint32')
                    else:
                        raise ValueError(f'data for document `{label}` is of unknown type `{type(first_sent)}`')
            else:
                tok = []
                sent_borders = np.array([], dtype='uint32')
        else:
            tok = sents
            sent_borders = None

        if isinstance(tok, (list, tuple)):                          # tokens alone (no attributes)
            doc = spacydoc_from_tokens(tok, label=label, vocab=corp.nlp.vocab, sent_borders=sent_borders)
        else:
            if isinstance(tok, pd.DataFrame):  # each document is a dataframe
                tok = {col: coldata for col, coldata in zip(tok.columns, tok.to_numpy().T.tolist())
                       if col != 'sent'}
            elif not isinstance(tok, dict):
                raise ValueError(f'data for document `{label}` is of unknown type `{type(tok)}`')

            doc = spacydoc_from_tokens_with_attrdata(tok, label=label, vocab=corp.nlp.vocab,
                                                     doc_attr_names=doc_attr_names or (),
                                                     token_attr_names=token_attr_names or (),
                                                     sent_borders=sent_borders)

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
                    additional_attrs: Optional[Dict[str, Union[OrdCollection, np.ndarray, int, float, str]]] = None,
                    skip_sent_borders: bool = False):
    """Initialize a SpaCy document with a label and optionally a preset token mask and other token attributes."""
    n = len(doc)

    # set label
    doc._.label = doc_label

    # set token mask
    if mask is None:
        doc.user_data['mask'] = np.repeat(True, n)
    else:
        doc.user_data['mask'] = mask

    # generate token type hashes for "processed" array
    doc.user_data['processed'] = np.fromiter((t.orth for t in doc), dtype='uint64', count=n)

    # generate sentence borders: each item represents the index of the last token in the respective sentence
    if not skip_sent_borders:
        try:
            doc.user_data['sent_borders'] = np.cumsum([len(s) for s in doc.sents], dtype='uint32')
        except ValueError:
            # happens when sentence boundaries are not set in `doc.sents`, e.g. when sentencizer component was disabled
            pass

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


def _init_document(vocab: Vocab, spacydoc: Doc, label: str,
                   doc_attrs: Optional[Dict[str, Any]] = None,
                   token_attrs: Optional[Sequence[str]] = None):
    whitespace = np.array([vocab.strings[t.whitespace_] for t in spacydoc], dtype='uint64').reshape((len(spacydoc), 1))
    load_token_attrs = ['orth']
    if spacydoc.is_sentenced:
        load_token_attrs.append('sent_start')
    if token_attrs:
        load_token_attrs.extend(token_attrs)

    return Document(label,
                    has_sents=spacydoc.is_sentenced,
                    tokens=np.hstack((whitespace, spacydoc.to_array(load_token_attrs))),
                    doc_attrs=doc_attrs,
                    token_attrs=list(token_attrs))


def _chop_along_sentences(tok: Union[List[Union[str, int]], np.ndarray], doc: Doc,
                          sentences: bool, apply_filter: bool, as_array: bool,
                          tokens_as_hashes: bool = False) \
        -> Union[List[Union[str, int]], List[List[Union[str, int]]], np.ndarray, List[np.ndarray]]:
    if sentences:
        if 'sent_borders' not in doc.user_data.keys():
            raise RuntimeError('sentence borders not set; Corpus documents probably not parsed with sentence '
                               'recognition')

        if apply_filter:
            prev_idx = None
            sent_borders = []
            adjust = 0
            for idx in doc.user_data['sent_borders']:
                adjust += np.sum(~doc.user_data['mask'][prev_idx:idx])
                sent_borders.append(idx - adjust)
                prev_idx = idx
        else:
            sent_borders = doc.user_data['sent_borders']

        sent = []
        prev_idx = None
        for idx in sent_borders:
            if prev_idx is None or prev_idx < idx:   # make sure to skip "empty" sentences
                sent.append(tok[prev_idx:idx])
                prev_idx = idx

        if sent:
            return sent
        else:
            if as_array:
                if tokens_as_hashes:
                    return [np.array([], dtype='uint64')]
                else:
                    return [empty_chararray()]
            else:
                return [[]]
    else:
        return tok


def _filtered_doc_tokens(doc: Doc, sentences: bool = False, tokens_as_hashes: bool = False,
                         apply_filter: bool = True, as_array: bool = False) \
        -> Union[List[Union[str, int]], List[List[Union[str, int]]], np.ndarray, List[np.ndarray]]:
    """
    If `apply_filter` is True, apply token mask and return filtered tokens from `doc`.
    """
    hashes = doc.user_data['processed']

    if apply_filter:    # apply mask
        hashes = hashes[doc.user_data['mask']]

    if tokens_as_hashes:
        if as_array:
            tok = np.copy(hashes)
        else:
            tok = [int(h) for h in hashes]     # converts "np.uint64" types to Python "int"
    else:   # convert token hashes to token strings using the SpaCy Vocab object
        if as_array:
            tok = np.array([doc.vocab.strings[hash] for hash in hashes]) if len(hashes) > 0 else empty_chararray()
        else:
            tok = list(map(lambda hash: doc.vocab.strings[hash], hashes))

    return _chop_along_sentences(tok, doc, sentences=sentences, apply_filter=apply_filter,
                                 tokens_as_hashes=tokens_as_hashes, as_array=as_array)


def _filtered_doc_token_attr(doc: Doc, attr: str, custom: Optional[bool] = None, stringified: Optional[bool] = None,
                             sentences: bool = False, apply_filter=True, **kwargs) \
        -> Union[List, List[List], np.ndarray, List[np.ndarray]]:
    """
    If `apply_filter` is True, apply token mask and return filtered token attribute `attr` from `doc`.
    """
    if custom is None:   # this means "auto" – we first check if `attr` is a custom attrib.
        custom = attr in doc.user_data.keys()

    if custom:  # a custom token attribute in Doc.user_data
        if 'default' in kwargs and attr not in doc.user_data:   # return default if default avail. and attrib. not set
            n = np.sum(doc.user_data['mask']) if apply_filter else len(doc.user_data['mask'])
            res = np.repeat(kwargs['default'], n)
        else:   # the attribute is set
            res = doc.user_data[attr]

            if apply_filter:
                res = res[doc.user_data['mask']]
    else:   # a SpaCy token attribute
        if stringified is None:  # this means "auto" – we first check if a stringified attr. exists,
                                 # if not we try the original attr. name
            getattrfn = lambda t, a: getattr(t, a + '_') if hasattr(t, a + '_') else getattr(t, a)
        else:
            if stringified is True:
                attr += '_'
            getattrfn = getattr

        if apply_filter:
            res = [getattrfn(t, attr) for t, m in zip(doc, doc.user_data['mask']) if m]
        else:
            res = [getattrfn(t, attr) for t in doc]

    return _chop_along_sentences(res, doc, sentences=sentences, apply_filter=apply_filter, as_array=False)


def _token_pattern_matches(tokens: Dict[str, List[Any]], search_tokens: Any,
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
    """Set a new filter mask for document tokens."""
    assert set(docs.keys()) == set(matches.keys()), 'the document labels in `matches` and `docs` must match'

    if invert:
        matches = [~m for m in matches]

    # simply set the new filter mask to previously unfiltered elements; changes document masks in-place
    for lbl, mask in matches.items():
        doc = docs.spacydocs[lbl]
        assert len(mask) == sum(doc.user_data['mask']), \
            'length of matches mask must equal the number of unmasked tokens in the document'
        doc.user_data['mask'] = _ensure_writable_array(doc.user_data['mask'])
        doc.user_data['mask'][doc.user_data['mask']] = mask


def _ensure_writable_array(arr: np.ndarray) -> np.ndarray:
    """Make sure that `arr` is writable; if it's not, copy it."""
    if not arr.flags.writeable:
        return np.copy(arr)
    else:
        return arr


def _check_filter_args(**kwargs):
    """Helper function to check common filtering arguments match_type and glob_method."""
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
