"""
tmtoolkit – Text Mining and Topic Modeling Toolkit for Python

Markus Konrad <markus.konrad@wzb.eu>
"""

import logging

from . import topicmod, bow


__title__ = 'tmtoolkit'
__version__ = '0.9.0'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'


# logger used in whole tmtoolkit package
logger = logging.getLogger(__title__)
logger.addHandler(logging.NullHandler())
