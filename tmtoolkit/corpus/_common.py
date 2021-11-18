"""
Internal module with common functions and constants for text processing in the :mod:`tmtoolkit.corpus` module.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import os
from typing import Optional, Dict, Any, List, Union, Sequence

import numpy as np
from spacy import Vocab

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.normpath(os.path.join(MODULE_PATH, '..', 'data'))

#: Default SpaCy language models used for a given two-letter ISO 639-1 language code.
#: These model names will be appended with model size suffix like "_sm", "_md" or "_lg".
DEFAULT_LANGUAGE_MODELS = {
    'en': 'en_core_web',
    'de': 'de_core_news',
    'fr': 'fr_core_news',
    'es': 'es_core_news',
    'pt': 'pt_core_news',
    'it': 'it_core_news',
    'nl': 'nl_core_news',
    'el': 'el_core_news',
    'nb': 'nb_core_news',
    'lt': 'lt_core_news',
    'zh': 'zh_core_web',
    'ja': 'ja_core_news',
}

#: Map two-letter ISO 639-1 language code to language name.
LANGUAGE_LABELS = {
    'en': 'english',
    'de': 'german',
    'fr': 'french',
    'es': 'spanish',
    'pt': 'portuguese',
    'it': 'italian',
    'nl': 'dutch',
    'el': 'greek',
    'nb': 'norwegian-bokmal',
    'lt': 'lithuanian',
    'zh': 'chinese',
    'ja': 'japanese',
}

TOKENMAT_ATTRS = ('whitespace', 'token', 'sent_start', 'pos', 'lemma')   # TODO add all other possible SpaCy attributes


class Document:
    # TODO: cached string tokens array/list ?
    # TODO: move to _document module
    # TODO: allow custom (non-string?) token attributes
    # TODO: setitem to add custom token attributes
    # TODO: delitem to remove (custom) token attributes
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

    def __str__(self) -> str:
        return f'Document "{self.label}" ({len(self)} tokens, {len(self.token_attrs)} token attributes, ' \
               f'{len(self.doc_attrs)} document attributes)'

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, attr: str) -> list:
        if attr in TOKENMAT_ATTRS:
            # return token matrix attribute
            hashes = self.tokenmat[:, self.tokenmat_attrs.index(attr)]
            if attr == 'sent_start':
                return [h == 1 for h in hashes]
            else:
                return [self.vocab.strings[h] for h in hashes]
        else:
            # return custom attribute
            return self.custom_token_attrs[attr]

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


def simplified_pos(pos: str, tagset: str = 'ud', default: str = '') -> str:
    """
    Return a simplified POS tag for a full POS tag `pos` belonging to a tagset `tagset`.

    Does the following conversion by default:

    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all ADJ... (adjective) tags to 'ADJ'
    - all ADV... (adverb) tags to 'ADV'
    - all other to `default`

    Does the following conversion by with ``tagset=='penn'``:

    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all JJ... (adjective) tags to 'ADJ'
    - all RB... (adverb) tags to 'ADV'
    - all other to `default`

    Does the following conversion by with ``tagset=='ud'``:

    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all JJ... (adjective) tags to 'ADJ'
    - all RB... (adverb) tags to 'ADV'
    - all other to `default`

    :param pos: a POS tag as string
    :param tagset: tagset used for `pos`; can be ``'wn'`` (WordNet), ``'penn'`` (Penn tagset)
                   or ``'ud'`` (universal dependencies – default)
    :param default: default return value when tag could not be simplified
    :return: simplified tag string
    """

    if pos and not isinstance(pos, str):
        raise ValueError('`pos` must be a string or None')

    if tagset == 'ud':
        if pos in ('NOUN', 'PROPN'):
            return 'N'
        elif pos == 'VERB':
            return 'V'
        elif pos in ('ADJ', 'ADV'):
            return pos
        else:
            return default
    elif tagset == 'penn':
        if pos.startswith('N') or pos.startswith('V'):
            return pos[0]
        elif pos.startswith('JJ'):
            return 'ADJ'
        elif pos.startswith('RB'):
            return 'ADV'
        else:
            return default
    elif tagset == 'wn':
        if pos.startswith('N') or pos.startswith('V'):
            return pos[0]
        elif pos.startswith('ADJ') or pos.startswith('ADV'):
            return pos[:3]
        else:
            return default
    else:
        raise ValueError('unknown tagset "%s"' % tagset)
