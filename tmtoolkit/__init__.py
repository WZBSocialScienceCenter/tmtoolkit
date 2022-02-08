"""
tmtoolkit – Text Mining and Topic Modeling Toolkit for Python

Markus Konrad <markus.konrad@wzb.eu>
"""

from importlib.util import find_spec
import logging

__title__ = 'tmtoolkit'
__version__ = '0.11.0'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'

logger = logging.getLogger(__title__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)   # set default level


from . import bow, topicmod, tokenseq, types, utils

if find_spec('spacy') and find_spec('globre'):
    from . import corpus
