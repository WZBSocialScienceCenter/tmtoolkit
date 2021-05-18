"""
tmtoolkit – Text Mining and Topic Modeling Toolkit for Python

Markus Konrad <markus.konrad@wzb.eu>
"""

from importlib.util import find_spec

from . import topicmod, bow

if find_spec('spacy') and find_spec('globre'):
    from . import preprocess


__title__ = 'tmtoolkit'
__version__ = '0.11.0-dev'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'
