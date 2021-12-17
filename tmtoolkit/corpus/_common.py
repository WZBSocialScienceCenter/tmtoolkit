"""
Internal module with common functions and constants for text processing in the :mod:`tmtoolkit.corpus` module.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import os


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

STD_TOKEN_ATTRS = ('pos', 'lemma')
BOOLEAN_SPACY_TOKEN_ATTRS = ('is_punct', 'is_stop', 'like_num',)
SPACY_TOKEN_ATTRS = STD_TOKEN_ATTRS + BOOLEAN_SPACY_TOKEN_ATTRS


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
