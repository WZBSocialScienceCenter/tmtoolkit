from functools import partial
from typing import Optional, Dict, Any, List, Union, Sequence

import numpy as np
from spacy import Vocab

from tmtoolkit.utils import empty_chararray

TOKENMAT_ATTRS = ('whitespace', 'token', 'sent_start', 'pos', 'lemma')   # TODO add all other possible SpaCy attributes


#%% document class


class Document:
    # TODO: cached string tokens array/list ?
    # TODO: how handle unknown attributes (i.e. when corp['bla'] = Document(...) and attributes don't match with
    #       existing docs' attributes)
    def __init__(self, vocab: Vocab, label: str, has_sents: bool,
                 tokenmat: np.ndarray,
                 tokenmat_attrs: List[str] = None,
                 custom_token_attrs: Optional[Dict[str, Union[Sequence, np.ndarray]]] = None,
                 doc_attrs: Optional[Dict[str, Any]] = None):
        doc_attrs = doc_attrs or {}
        doc_attrs['label'] = label
        doc_attrs['has_sents'] = has_sents

        # SpaCy Vocab instance
        self.vocab = vocab

        # uint64 matrix of shape (N, M) for N tokens and with M attributes, where M is
        # len(token_attrs) + 2 or 3 (without or with information about sentences)
        self.tokenmat = tokenmat
        self.custom_token_attrs = {}   # type: Dict[str, np.ndarray]

        if custom_token_attrs:
            for attr, val in custom_token_attrs.items():
                self[attr] = val

        # contains standard attrib. "label", "has_sents"
        self.doc_attrs = doc_attrs

        # labels of additional token attributes in `tokens` after base attributes
        base_token_attrs = ['whitespace', 'token']
        if has_sents:
            base_token_attrs.append('sent_start')
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

    @property
    def label(self) -> str:
        return self.doc_attrs['label']

    @property
    def has_sents(self) -> bool:
        return self.doc_attrs['has_sents']

    @property
    def token_attrs(self) -> List[str]:
        return self.tokenmat_attrs + list(self.custom_token_attrs.keys())


#%% document functions


def document_token_attr(d: Document, attr: str,
                        sentences: bool = False,
                        as_hashes: bool = False,
                        as_array: bool = False) \
        -> Union[List[Union[str, int]], List[List[Union[str, int]]], np.ndarray, List[np.ndarray]]:
    if sentences and not d.has_sents:
        raise RuntimeError('sentence borders not set; Corpus documents probably not parsed with sentence recognition')

    if attr in TOKENMAT_ATTRS:
        # return token matrix attribute
        tok = d.tokenmat[:, d.tokenmat_attrs.index(attr)]   # token attribute hashes

        if as_hashes and not as_array:
            tok = tok.tolist()
        elif not as_hashes:
            if attr == 'sent_start':
                if as_array:
                    tok = tok == 1
                else:
                    tok = [t == 1 for t in tok]
            else:
                tok = [d.vocab.strings[t] for t in tok]
                if as_array:
                    tok = np.array(tok)
    else:
        if as_hashes:
            raise ValueError('cannot return hashes for a custom token attribute')

        # return custom attribute
        tok = d.custom_token_attrs[attr]
        if not as_array:
            tok = tok.tolist()

    if sentences:
        return _chop_along_sentences(tok, d.tokenmat[:, d.tokenmat_attrs.index('sent_start')],
                                     as_array=as_array, as_hashes=as_hashes)
    else:
        return tok


document_tokens = partial(document_token_attr, attr='token')


#%% internal helper functions

def _chop_along_sentences(tok: Union[list, np.ndarray],
                          sent_start: np.ndarray,
                          as_array: bool,
                          as_hashes: bool) \
        -> Union[List[Union[str, int]], List[List[Union[str, int]]], np.ndarray, List[np.ndarray]]:
    # generate sentence borders: each item represents the index of the last token in the respective sentence;
    # only the last token index is always missing
    sent_borders = np.nonzero(sent_start == 1)[0][1:]

    sent = []
    prev_idx = None
    for idx in sent_borders:
        if prev_idx is None or prev_idx < idx:  # make sure to skip "empty" sentences
            sent.append(tok[prev_idx:idx])
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
