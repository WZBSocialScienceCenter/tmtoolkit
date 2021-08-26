"""
Internal module with some additional functions that are only available when the `NLTK <https://www.nltk.org/>`_ package
is installed.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""
from typing import Optional

from ._corpus import Corpus
from ._common import LANGUAGE_LABELS
from ._corpusfuncs import transform_tokens


def stem(docs: Corpus, /, language: Optional[str] = None,
         stemmer_instance: Optional[object] = None, inplace=True):
    """
    Apply stemming to all tokens in `docs` using a stemmer `stemmer_instance`.

    .. note: This requires that the `NLTK <https://www.nltk.org/>`_ package is installed.

    :param docs: a Corpus object
    :param language: language in which `docs` is given; if None, will be detected from the ``language`` property of
                     `docs`; note that this is not an ISO language code but a language
                     label like "english" or "german" that NLTK accepts
    :param stemmer_instance: a stemmer instance; it must implement a method `stem` that accepts a single string;
                             default is :class:`nltk.stem.SnowballStemmer`
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either original Corpus object `docs` or a modified copy of it
    """

    from nltk.stem import SnowballStemmer

    if stemmer_instance is None:
        if language is None:
            language = LANGUAGE_LABELS[docs.language]
        stemmer_instance = SnowballStemmer(language)

    return transform_tokens(docs, stemmer_instance.stem, inplace=inplace)
