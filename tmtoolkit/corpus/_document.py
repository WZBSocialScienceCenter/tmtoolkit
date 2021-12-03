from __future__ import annotations   # req. for classmethod return type; see https://stackoverflow.com/a/49872353
from typing import Optional, Dict, Any, List, Union, Sequence

import numpy as np
from bidict import bidict
from spacy.strings import hash_string

from ..tokenseq import token_ngrams
from ..types import UnordStrCollection
from ..utils import empty_chararray, flatten_list

from ._common import SPACY_TOKEN_ATTRS, BOOLEAN_SPACY_TOKEN_ATTRS


TOKENMAT_ATTRS = ('whitespace', 'token', 'sent_start') + SPACY_TOKEN_ATTRS
PROTECTED_ATTRS = TOKENMAT_ATTRS + ('sent',)


#%% document class


class Document:
    # TODO: cached string tokens array/list ?
    # TODO: how handle unknown attributes (i.e. when corp['bla'] = Document(...) and attributes don't match with
    #       existing docs' attributes)
    def __init__(self, bimaps: Optional[Dict[str, bidict]],
                 label: str, has_sents: bool,
                 tokenmat: np.ndarray,
                 tokenmat_attrs: List[str] = None,
                 custom_token_attrs: Optional[Dict[str, Union[Sequence, np.ndarray]]] = None,
                 doc_attrs: Optional[Dict[str, Any]] = None):
        doc_attrs = doc_attrs or {}
        doc_attrs['label'] = label
        doc_attrs['has_sents'] = has_sents

        # contains standard attrib. "label", "has_sents"
        self.doc_attrs = doc_attrs

        # bimaps for hash <-> string token conversion
        self.bimaps = bimaps

        # uint64 matrix of shape (N, M) for N tokens and with M attributes, where M is
        # len(tokenmat_attrs) + 2 or 3 (without or with information about sentences)
        self.tokenmat = tokenmat
        self.custom_token_attrs = {}   # type: Dict[str, np.ndarray]

        if custom_token_attrs:
            for attr, val in custom_token_attrs.items():
                self[attr] = val

        # labels of additional token attributes in `tokens` after base attributes
        tokenmat_attrs = tokenmat_attrs or []
        base_token_attrs = ('whitespace', 'token', 'sent_start') if has_sents else ('whitespace', 'token')
        base_token_attrs = [tokenmat_attrs.pop(tokenmat_attrs.index(a)) if a in tokenmat_attrs else a
                            for a in base_token_attrs]
        self.tokenmat_attrs = base_token_attrs + (tokenmat_attrs or [])

        assert self.tokenmat.ndim == 2, '`tokenmat` must be 2D-array'
        assert self.tokenmat.shape[1] == 2 + int(has_sents) + len(tokenmat_attrs), \
            '`tokens` must contain 3+len(token_attrs) columns'

    def __len__(self) -> int:
        return self.tokenmat.shape[0]

    def __repr__(self) -> str:
        return f'Document "{self.label}" ({len(self)} tokens, {len(self.token_attrs)} token attributes, ' \
               f'{len(self.doc_attrs)} document attributes)'

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, attr: str) -> list:
        return document_token_attr(self, attr)

    def __setitem__(self, attr: str, values: Union[Sequence, np.ndarray]):
        if not isinstance(values, (Sequence, np.ndarray)):
            raise ValueError('`values` must be sequence or NumPy array')

        if len(values) != len(self):
            raise ValueError(f'number of token attribute values ({len(values)}) does not match number of tokens '
                             f'({len(self)})')

        if not isinstance(values, np.ndarray):
            values = np.array(values, dtype='uint64' if attr in TOKENMAT_ATTRS else None)

        if values.ndim != 1:
            raise ValueError('`values` must be one-dimensional')

        if attr in TOKENMAT_ATTRS:
            # add/replace token matrix attribute
            if not np.issubdtype(values.dtype, 'uint64'):
                raise ValueError('`values` must be uint64 array')

            if attr in self.tokenmat_attrs:   # replace
                self.tokenmat[:, self.tokenmat_attrs.index(attr)] = values
            else:                             # add
                self.tokenmat = np.hstack((self.tokenmat, values.reshape((len(self), 1))))
                self.tokenmat_attrs.append(attr)
        else:
            # add/replace custom token attribute
            self.custom_token_attrs[attr] = values

    def __delitem__(self, attr: str):
        if attr in TOKENMAT_ATTRS:
            # remove token matrix attribute
            self.tokenmat = np.delete(self.tokenmat, self.tokenmat_attrs.index(attr), axis=1)
            self.tokenmat_attrs.remove(attr)
        else:
            # remove custom token attribute
            del self.custom_token_attrs[attr]

    def __copy__(self) -> Document:
        """
        Make a copy of this Document, returning a new object with the same data but using the *same* SpaCy instance.

        :return: new Corpus object
        """
        return self._deserialize(self._serialize(store_bimaps_pointer=True))

    @property
    def label(self) -> str:
        return self.doc_attrs['label']

    @property
    def has_sents(self) -> bool:
        return self.doc_attrs['has_sents']

    @property
    def token_attrs(self) -> List[str]:
        return self.tokenmat_attrs + list(self.custom_token_attrs.keys())

    def _serialize(self, store_bimaps_pointer: bool) -> Dict[str, Any]:
        serdata = {}
        for attr in ('doc_attrs', 'tokenmat', 'tokenmat_attrs', 'custom_token_attrs'):
            serdata[attr] = getattr(self, attr).copy()

        if store_bimaps_pointer:
            serdata['bimaps'] = self.bimaps

        return serdata

    @classmethod
    def _deserialize(cls, data: Dict[str, Any], **kwargs) -> Document:
        init_args = {
            'bimaps': data.get('bimaps', kwargs.pop('bimaps', None)),
            'label': data['doc_attrs'].pop('label'),
            'has_sents': data['doc_attrs'].pop('has_sents'),
            'tokenmat': data['tokenmat'],
            'tokenmat_attrs': data['tokenmat_attrs'],
            'custom_token_attrs': data['custom_token_attrs'],
            'doc_attrs': data['doc_attrs'],
        }

        init_args.update(kwargs)

        return cls(**init_args)


#%% document functions


def document_token_attr(d: Document,
                        attr: Union[str, Sequence[str]] = 'token',
                        default: Optional[Any, Dict[str, Any]] = None,
                        sentences: bool = False,
                        ngrams: int = 1,
                        ngrams_join: str = ' ',
                        as_hashes: bool = False,
                        as_array: bool = False) \
        -> Union[list,
                 List[list],                        # sentences
                 np.ndarray,
                 List[np.ndarray],                  # sentences
                 # w/ multiple attributes
                 Dict[str, list],
                 Dict[str, List[list]],             # sentences
                 Dict[str, np.ndarray],
                 Dict[str, List[np.ndarray]]]:      # sentences
    if not as_hashes and d.bimaps is None:
        raise ValueError('tokens as string representation requested, but no bimaps instance set for document `d`')

    if sentences and not d.has_sents:
        raise ValueError('sentences requested, but sentence borders not set; '
                         'Corpus documents probably not parsed with sentence recognition')

    if ngrams > 1 and as_hashes:
        raise ValueError('cannot join ngrams as hashes; either set `as_hashes` to False or use unigrams')

    if isinstance(attr, str):
        single_attr = attr
        attr = [attr]
        if default:
            default = {attr: default}
    else:
        single_attr = None
        if default is not None and not isinstance(default, dict):
            raise ValueError('`default` must be a dict mapping attribute names to default values')

    if ngrams > 1 and 'sent_start' in attr:
        raise ValueError('cannot join ngrams for sent_start')

    if 'sent' in attr and not d.has_sents:
        raise ValueError('sentence numbers requested, but sentence borders not set; '
                         'Corpus documents probably not parsed with sentence recognition')

    res = {}   # token attributes per attribute
    for a in attr:   # iterate through requested attributes
        if a in TOKENMAT_ATTRS:
            # token matrix attribute
            tok = d.tokenmat[:, d.tokenmat_attrs.index(a)]   # token attribute hashes

            if a in BOOLEAN_SPACY_TOKEN_ATTRS:
                tok = tok.astype(bool)

            if as_hashes and not as_array:
                tok = tok.tolist()
            elif not as_hashes:
                if a == 'sent_start':
                    if as_array:
                        tok = tok == 1
                    else:
                        tok = [t == 1 for t in tok]
                else:
                    if a in BOOLEAN_SPACY_TOKEN_ATTRS:
                        if not as_array or ngrams > 1:
                            tok = tok.tolist()
                            if ngrams > 1:
                                tok = list(map(str, tok))
                    else:
                        tok = [d.bimaps[a][t] for t in tok]

                    if ngrams > 1:
                        tok = token_ngrams(tok, n=ngrams, join=True, join_str=ngrams_join)

                    if as_array and (a not in BOOLEAN_SPACY_TOKEN_ATTRS or ngrams > 1):
                        tok = np.array(tok, dtype=str)
        elif a == 'sent':
            sent_start = d.tokenmat[:, d.tokenmat_attrs.index('sent_start')]
            tok = np.cumsum(sent_start == 1)
            if not as_array:
                tok = tok.tolist()
        else:
            # custom attribute
            if as_hashes:
                raise ValueError('cannot return hashes for a custom token attribute')

            if default is None or a in d.custom_token_attrs.keys():
                tok = d.custom_token_attrs[a]
            else:
                tok = np.repeat(default[a], len(d))

            if not as_array:
                tok = tok.tolist()

            if ngrams > 1:
                if as_array:
                    tok = tok.tolist()

                try:
                    tok = token_ngrams(tok, n=ngrams, join=True, join_str=ngrams_join)
                except TypeError:   # tok is not a list of str
                    tok = token_ngrams(list(map(str, tok)), n=ngrams, join=True, join_str=ngrams_join)

                if as_array:
                    tok = np.array(tok, dtype=str)

        if sentences:
            tok = _chop_along_sentences(tok, d.tokenmat[:, d.tokenmat_attrs.index('sent_start')],
                                        as_array=as_array, as_hashes=as_hashes, skip=-ngrams+1)

        res[a] = tok

    if single_attr is None:
        return res
    else:
        return res[single_attr]


def document_from_attrs(bimaps: Dict[str, bidict], label: str, tokens_w_attr: Dict[str, Union[list, np.ndarray]],
                        sentences: bool,
                        doc_attr_names: Optional[UnordStrCollection] = None,
                        token_attr_names: Optional[UnordStrCollection] = None) \
        -> Document:
    def uint64arr_from_strings(strings):
        hashes = []
        upd = []

        for s in strings:
            h = hash_string(s)
            if h not in bimaps['token']:
                upd.append((h, s))
            hashes.append(h)

        bimaps['token'].update(upd)

        return np.array([hashes], dtype='uint64')

    def values_as_uint64arr(val):
        if isinstance(val, np.ndarray):
            if np.issubdtype(val.dtype, str):    # this is an array of strings -> convert to hashes
                return uint64arr_from_strings(val.tolist())
            else:
                return val.astype('uint64')
        elif isinstance(val, list):
            try:
                return np.array(val, dtype='uint64')
            except ValueError:   # this is a list of strings -> convert to hashes
                return uint64arr_from_strings(val)
        else:
            raise ValueError('`tokens_w_attr` must be a dict that contains lists or NumPy arrays')

    def flatten_if_sents(val):
        if sentences:
            if val and isinstance(next(iter(val)), np.ndarray):
                return np.concatenate(val)
            else:
                return flatten_list(val)
        return val

    sent_start = None

    if sentences:
        sent_borders = np.cumsum(list(map(len, tokens_w_attr['token'])))
        if len(sent_borders) > 0 and sent_borders[-1] > 0:  # non-empty doc.
            sent_start = np.repeat(0, sent_borders[-1])
            sent_start[0] = 1
            sent_start[sent_borders[:-1]] = 1
            sent_start = sent_start.astype(dtype='uint64')
        else:   # empty document
            sent_start = np.array([], dtype='uint64')

    if 'sent_start' in tokens_w_attr:
        sent_start = tokens_w_attr['sent_start']

    if sent_start is None:
        base_attrs = ('whitespace', 'token')
    else:
        base_attrs = ('whitespace', 'token', 'sent_start')

    tokenmat_arrays = []
    tokenmat_attrs = []
    for attr in base_attrs:
        if attr == 'sent_start':
            val = sent_start
        else:
            try:
                val = tokens_w_attr[attr]
            except KeyError:
                raise ValueError(f'at least the following base token attributes must be given: {base_attrs}')

            val = flatten_if_sents(val)

        tokenmat_arrays.append(values_as_uint64arr(val))
        tokenmat_attrs.append(attr)

    custom_token_attrs = {}
    doc_attrs = {}
    for attr, val in tokens_w_attr.items():
        if attr in base_attrs:
            continue

        val = flatten_if_sents(val)

        if attr in TOKENMAT_ATTRS:
            tokenmat_arrays.append(values_as_uint64arr(val))
            tokenmat_attrs.append(attr)
        else:
            if (token_attr_names is not None and attr in token_attr_names) or \
                    (token_attr_names is None and isinstance(val, (list, np.ndarray))):
                custom_token_attrs[attr] = val
            elif (doc_attr_names is not None and attr in doc_attr_names) or \
                    (doc_attr_names is None and not isinstance(val, (list, np.ndarray))):
                doc_attrs[attr] = val
            else:
                raise ValueError(f"don't know how to handle attribute \"{attr}\"")

    return Document(bimaps, label=label, has_sents=sent_start is not None,
                    tokenmat=np.vstack(tokenmat_arrays).T, tokenmat_attrs=tokenmat_attrs,
                    custom_token_attrs=custom_token_attrs,
                    doc_attrs=doc_attrs)


#%% internal helper functions

def _chop_along_sentences(tok: Union[list, np.ndarray],
                          sent_start: np.ndarray,
                          as_array: bool,
                          as_hashes: bool,
                          skip: int = 0) \
        -> Union[List[Union[str, int]], List[List[Union[str, int]]], np.ndarray, List[np.ndarray]]:
    # generate sentence borders: each item represents the index of the last token in the respective sentence;
    # only the last token index is always missing
    sent_borders = np.nonzero(sent_start == 1)[0][1:]

    sent = []
    prev_idx = 0
    for idx in sent_borders:
        if prev_idx < idx+skip:  # make sure to skip "empty" sentences
            sent.append(tok[prev_idx:idx+skip])
            prev_idx = idx

    sent.append(tok[prev_idx:len(tok)])  # last token index is always missing

    if sent:
        return sent
    else:
        if as_array:
            if as_hashes:
                return [np.array([], dtype='uint64')]
            else:
                return [empty_chararray()]
        else:
            return [[]]
