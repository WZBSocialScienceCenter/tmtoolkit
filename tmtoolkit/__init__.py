import logging

from . import topicmod


__title__ = 'tmtoolkit'
__version__ = '0.6.0'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'


logger = logging.getLogger(__title__)
logger.addHandler(logging.NullHandler())
