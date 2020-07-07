"""
Some additional functions that are only available when the `NLTK <https://www.nltk.org/>`_ package is installed.
"""

from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn

from ._docfuncs import transform


def stem(docs, language, stemmer_instance=None):
    """
    Apply stemming to all tokens in `docs` using a stemmer `stemmer_instance`.

    .. note: This requires that the `NLTK <https://www.nltk.org/>`_ package is installed.

    :param docs: list of string tokens or spaCy documents
    :param language: language in which `docs` is given; note that this is not an ISO language code but a language
                     label like "english" or "german" that NLTK accepts
    :param stemmer_instance: a stemmer instance; it must implement a method `stem` that accepts a single string;
                             default is :class:`nltk.stem.SnowballStemmer`
    :return: list of string tokens or spaCy documents, depending on `docs`
    """

    if stemmer_instance is None:
        stemmer_instance = SnowballStemmer(language)

    return transform(docs, stemmer_instance.stem)


def pos_tag_convert_penn_to_wn(tag):
    """
    Convert POS tag from Penn tagset to WordNet tagset.

    .. note: This requires that the `NLTK <https://www.nltk.org/>`_ package is installed.

    :param tag: a tag from Penn tagset
    :return: a tag from WordNet tagset or None if no corresponding tag could be found
    """

    if tag in ['JJ', 'JJR', 'JJS']:
        return wn.ADJ
    elif tag in ['RB', 'RBR', 'RBS']:
        return wn.ADV
    elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return wn.NOUN
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return wn.VERB
    return None