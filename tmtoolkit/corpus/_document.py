"""
Internal module that implements :class:`Document` class representing a text document as token sequence.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

from __future__ import annotations   # req. for classmethod return type; see https://stackoverflow.com/a/49872353
from typing import Optional, Dict, Any, List, Union, Sequence

import numpy as np
from bidict import bidict
from spacy import Vocab
from spacy.strings import hash_string

from ..tokenseq import token_ngrams
from ..types import UnordStrCollection
from ..utils import empty_chararray, flatten_list

from ._common import SPACY_TOKEN_ATTRS, BOOLEAN_SPACY_TOKEN_ATTRS

#%% constants

# names of token attributes that can be stored in the token matrix
TOKENMAT_ATTRS = ('whitespace', 'token', 'sent_start') + SPACY_TOKEN_ATTRS
# names of token attributes that are protected, i.e. cannot be used as custom attribute names
PROTECTED_ATTRS = TOKENMAT_ATTRS + ('sent',)


#%% document class

class Document:
    """
    A class that represents text as sequence of tokens. Attributes are also implemented at two levels:

    1. Document attributes like the document label (document name);
    2. Token attributes (e.g. POS, lemma, etc.)

    Token attributes are further divided into "standard" or "SpaCy" token attributes and custom attributes.
    The former are represented as 64 bit unsigned integer hash value and are stored in a "token matrix" in which
    rows represent tokens and columns represent token attributes. The token hash itself is also stored in this matrix
    as "token" attribute. The custom token attributes are stored as NumPy arrays of any dtype.
    """

    def __init__(self, bimaps: Optional[Dict[str, bidict]],
                 label: str, has_sents: bool,
                 tokenmat: np.ndarray,
                 tokenmat_attrs: List[str] = None,
                 custom_token_attrs: Optional[Dict[str, Union[Sequence, np.ndarray]]] = None,
                 doc_attrs: Optional[Dict[str, Any]] = None):
        """
        Create a new :class:`~tmtoolkit.corpus.Document` object that uses the bidirectional dictionaries in `bimaps` for
        hash <-> text conversion, has a document label `label`, has sentences recognized (`has_sents`) and has a token
         matrix `tokenmat`.

        :param bimaps: bidirectional dictionaries for hash <-> text conversion of data in `tokenmat`
        :param label: document label (document name)
        :param has_sents: if True, this document supports sentences
        :param tokenmat: token matrix as uint64 matrix of shape (N, M) for N tokens and with M attributes; the data is
                         *not* copied
        :param tokenmat_attrs: names of token attributes in `tokenmat` with respect to column order
        :param custom_token_attrs: additional custom token attributes
        :param doc_attrs: document attributes
        """
        # set up document attributes
        doc_attrs = {} if doc_attrs is None else doc_attrs.copy()
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
        tokenmat_attrs = [] if tokenmat_attrs is None else tokenmat_attrs.copy()
        base_token_attrs = ('whitespace', 'token', 'sent_start') if has_sents else ('whitespace', 'token')
        base_token_attrs = [tokenmat_attrs.pop(tokenmat_attrs.index(a)) if a in tokenmat_attrs else a
                            for a in base_token_attrs]
        self.tokenmat_attrs = base_token_attrs + (tokenmat_attrs or [])

        assert self.tokenmat.ndim == 2, '`tokenmat` must be 2D-array'
        assert self.tokenmat.shape[1] == 2 + int(has_sents) + len(tokenmat_attrs), \
            '`tokens` must contain 3+len(token_attrs) columns'

    def __len__(self) -> int:
        """
        Length of the document, i.e. number of tokens.

        :return: length of the document, i.e. number of tokens
        """
        return self.tokenmat.shape[0]

    def __repr__(self) -> str:
        """
        Document summary.

        :return: document summary as string
        """
        return f'Document "{self.label}" ({len(self)} tokens, {len(self.token_attrs)} token attributes, ' \
               f'{len(self.doc_attrs)} document attributes)'

    def __str__(self) -> str:
        """
        Document summary.

        :return: document summary as string
        """
        return self.__repr__()

    def __getitem__(self, attr: str) -> list:
        """
        Get list of token attributes for attribute `attr`. E.g. ``d['token']`` returns list of
        text tokens or ``d['pos']`` returns list of POS-tags for each token of a document ``d``.

        .. seealso:: See :func:`~tmtoolkit.corpus.document_token_attr` for more options on retrieving token attributes.

        :param attr: token attribute
        :return: list of token attributes
        """
        return document_token_attr(self, attr)

    def __setitem__(self, attr: str, values: Union[Sequence, np.ndarray]):
        """
        Set a new token attribute `attr` with values `values`. The length of `values` must match the number of tokens
        in this document.

        :param attr: token attribute name
        :param values: token attribute values
        """
        if not isinstance(values, (Sequence, np.ndarray)):
            raise ValueError('`values` must be sequence or NumPy array')

        if len(values) != len(self):
            raise ValueError(f'number of token attribute values ({len(values)}) does not match number of tokens '
                             f'({len(self)})')

        if not isinstance(values, np.ndarray):
            # convert to NumPy array: if attr is a token matrix attribute, convert to uint64, else don't specify a type
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
        """
        Remove a token attribute `attr`.

        :param attr: token attribute to remove
        """
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
        """Document label (document name)."""
        return self.doc_attrs['label']

    @property
    def has_sents(self) -> bool:
        """
        Status on information on borders of sentences.

        :return: True if information on borders of sentences is contained in this document, else False
        """
        return self.doc_attrs['has_sents']

    @property
    def token_attrs(self) -> List[str]:
        """
        Retrieve list of token attribute names (standard *and* custom attributes).

        :return: list of token attribute names
        """
        return self.tokenmat_attrs + list(self.custom_token_attrs.keys())

    def _serialize(self, store_bimaps_pointer: bool) -> Dict[str, Any]:
        """
        Helper function for serializing a Document object as dictionary.

        :param store_bimaps_pointer: if True, add an entry ``'bimaps'`` pointing to the ``self.bimaps`` object
        :return: dictionary with all necessary data to reconstruct this Document object
        """
        serdata = {}
        for attr in ('doc_attrs', 'tokenmat', 'tokenmat_attrs', 'custom_token_attrs'):
            serdata[attr] = getattr(self, attr).copy()

        if store_bimaps_pointer:
            serdata['bimaps'] = self.bimaps

        return serdata

    @classmethod
    def _deserialize(cls, data: Dict[str, Any], **kwargs) -> Document:
        """
        Helper function for de-serializing a Document object from a dictionary.

        :param data: dictionary of document data
        :param kwargs: additional keyword arguments passed to Document constructor
        :return: new Document object
        """
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
    """
    Retrieve one or more token attributes given as `attr` from a :class:`~tmtoolkit.corpus.Document` object `d`.

    :param d: :class:`~tmtoolkit.corpus.Document` object
    :param attr: either single token attribute name or a sequence of token attribute names
    :param default: default value if a token attribute doesn't exist
    :param sentences: divide result into sentences
    :param ngrams: form n-grams if `ngrams` > 1
    :param ngrams_join: use this string to join the n-grams if `ngrams` > 1
    :param as_hashes: return hashes instead of textual representations
    :param as_array: return NumPy arrays instead of lists
    :return: if a single token attribute is given as `attr`, return a list, a NumPy array or a list of lists or NumPy
             arrays depending on `as_array` and `sentences`; if multiple token attributes are given, return a dictionary
             mapping the token attribute name to the respective result
    """
    # check arguments
    if not as_hashes and d.bimaps is None:
        raise ValueError('tokens as string representation requested, but no bimaps instance set for document `d`')

    if sentences and not d.has_sents:
        raise ValueError('sentences requested, but sentence borders not set; '
                         'Corpus documents probably not parsed with sentence recognition')

    if ngrams > 1 and as_hashes:
        raise ValueError('cannot join ngrams as hashes; either set `as_hashes` to False or use unigrams')

    if isinstance(attr, str):  # handle single attribute
        single_attr = attr
        attr = [attr]
        if default:
            default = {attr: default}
    else:  # handle multiple attributes
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

            if a in BOOLEAN_SPACY_TOKEN_ATTRS:   # convert to boolean array
                tok = tok.astype(bool)

            if as_hashes and not as_array:
                tok = tok.tolist()   # list of hashes
            elif not as_hashes:
                # convert hashes to other representations (mostly string representations)
                if a == 'sent_start':    # sentences border marks
                    if as_array:
                        tok = tok == 1   # convert to boolean array
                    else:
                        tok = [t == 1 for t in tok]    # convert to boolean list
                else:   # every other token matrix attribute
                    if a in BOOLEAN_SPACY_TOKEN_ATTRS:
                        if not as_array or ngrams > 1:
                            tok = tok.tolist()   # convert boolean attribute simply to list
                            if ngrams > 1:   # ngrams require strings
                                tok = list(map(str, tok))
                    else:   # use the attribute-specific bimap to convert the hash to a string
                        tok = [d.bimaps[a][t] for t in tok]

                    if ngrams > 1:   # generate ngrams
                        tok = token_ngrams(tok, n=ngrams, join=True, join_str=ngrams_join)

                    # convert (back) to NumPy array
                    if as_array and (a not in BOOLEAN_SPACY_TOKEN_ATTRS or ngrams > 1):
                        tok = np.array(tok, dtype=str)
        elif a == 'sent':
            # special attribute "sent" generates a sentence number for each token
            sent_start = d.tokenmat[:, d.tokenmat_attrs.index('sent_start')]
            tok = np.cumsum(sent_start == 1)
            if not as_array:
                tok = tok.tolist()
        else:
            # custom attribute
            if default is None or a in d.custom_token_attrs.keys():
                tok = d.custom_token_attrs[a]
            else:   # attribute doesn't exist, but default value is given
                tok = np.repeat(default[a], len(d))

            if as_hashes:
                raise ValueError('cannot return hashes for a custom token attribute')

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
            # divide tokens into sentences
            tok = _chop_along_sentences(tok, d.tokenmat[:, d.tokenmat_attrs.index('sent_start')],
                                        as_array=as_array, as_hashes=as_hashes, skip=-ngrams+1)

        res[a] = tok

    if single_attr is None:   # return dict with multiple attributes
        return res
    else:
        return res[single_attr]   # return single attribute values


def document_from_attrs(bimaps: Dict[str, bidict],
                        vocab: Vocab,
                        label: str,
                        tokens_w_attr: Dict[str, Union[list, np.ndarray]],
                        sentences: bool,
                        doc_attr_names: Optional[UnordStrCollection] = None,
                        token_attr_names: Optional[UnordStrCollection] = None) \
        -> Document:
    """
    Create a new :class:`~tmtoolkit.corpus.Document` object from tokens with attributes in `tokens_w_attr`.

    :param bimaps: bidirectional dictionaries for hash <-> text conversion
    :param label: document label
    :param tokens_w_attr: dictionary mapping attribute names to attribute values
    :param sentences: if True, `tokens_w_attr` contains data split by sentences, else sentences are not split
    :param doc_attr_names: names of keys in `tokens_w_attr` that are assumed to be document attributes
    :param token_attr_names: names of keys in `tokens_w_attr` that are assumed to be token attributes
    :return:
    """
    def uint64arr_from_strings(attr, strings):
        """
        Helper function to convert strings of attribute `attr` to array of hashes while updating `bimaps` and `vocab`.
        """

        hashes = []
        upd = []

        for s in strings:
            h = vocab.strings[s] if s in vocab.strings else vocab.strings.add(s)
            if h not in bimaps[attr]:
                upd.append((h, s))
            hashes.append(h)

        bimaps[attr].update(upd)

        return np.array([hashes], dtype='uint64')

    def values_as_uint64arr(attr, val):
        """Helper function that tries to convert `val` to an array of hashes, depending on the type of `val`."""
        if isinstance(val, np.ndarray):
            if np.issubdtype(val.dtype, str):    # this is an array of strings -> convert to hashes
                return uint64arr_from_strings(attr, val.tolist())
            else:
                return val.astype('uint64')
        elif isinstance(val, list):
            try:
                return np.array(val, dtype='uint64')
            except ValueError:   # this is a list of strings -> convert to hashes
                return uint64arr_from_strings(attr, val)
        else:
            raise ValueError('`tokens_w_attr` must be a dict that contains lists or NumPy arrays')

    def flatten_if_sents(val):
        """Helper function to flatten sentences in `val` to a concatenated list or array."""
        if sentences:
            if val and isinstance(next(iter(val)), np.ndarray):
                return np.concatenate(val)
            else:
                return flatten_list(val)
        return val

    if not isinstance(tokens_w_attr, dict):
        raise ValueError('`tokens_w_attr` must be given as dict with token attributes')

    sent_start = None

    if 'sent_start' in tokens_w_attr and tokens_w_attr['sent_start'] is not None:
        sent_start = tokens_w_attr['sent_start']
    elif 'sent' in tokens_w_attr and tokens_w_attr['sent'] is not None:
        # convert sentence indices to sentence start indicators
        sent_start_indices = [0] + (np.flatnonzero(np.diff(tokens_w_attr['sent'])) + 1).tolist()
        sent_start = np.repeat(False, len(tokens_w_attr['token']))
        sent_start[sent_start_indices] = True
    elif sentences:
        # get sentence borders
        sent_borders = np.cumsum(list(map(len, tokens_w_attr['token'])))
        if len(sent_borders) > 0 and sent_borders[-1] > 0:  # non-empty doc.
            # create array filled with 0, where start of sentence is indicated by 1
            sent_start = np.repeat(0, sent_borders[-1])
            sent_start[0] = 1
            sent_start[sent_borders[:-1]] = 1
            sent_start = sent_start.astype(dtype='uint64')
        else:   # empty document
            sent_start = np.array([], dtype='uint64')

    # define "base attributes"
    if sent_start is None:
        base_attrs = ('whitespace', 'token')
    else:
        base_attrs = ('whitespace', 'token', 'sent_start')

    # collect data for "base attributes" in token matrix
    tokenmat_arrays = []
    tokenmat_attrs = []
    for attr in base_attrs:
        if attr in {'sent_start', 'sent'}:
            val = sent_start
        else:
            try:
                val = tokens_w_attr[attr]
            except KeyError:
                raise ValueError(f'at least the following base token attributes must be given: {base_attrs}')

            val = flatten_if_sents(val)

        tokenmat_arrays.append(values_as_uint64arr(attr, val))
        tokenmat_attrs.append(attr)

    # collect data for other token matrix attributes and for custom token attributes and document attributes
    custom_token_attrs = {}
    doc_attrs = {}
    for attr, val in tokens_w_attr.items():
        if attr in base_attrs or attr == 'sent':   # already collected
            continue

        val = flatten_if_sents(val)

        if attr in TOKENMAT_ATTRS:  # a token matrix attribute
            tokenmat_arrays.append(values_as_uint64arr(attr, val))
            tokenmat_attrs.append(attr)
        else:
            if (token_attr_names is not None and attr in token_attr_names) or \
                    (token_attr_names is None and isinstance(val, (list, np.ndarray))):
                custom_token_attrs[attr] = val   # a custom token attribute
            elif (doc_attr_names is not None and attr in doc_attr_names) or \
                    (doc_attr_names is None and not isinstance(val, (list, np.ndarray))):
                doc_attrs[attr] = val   # a document attribute
            else:
                raise ValueError(f"don't know how to handle attribute \"{attr}\"")

    # create Document instance
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
