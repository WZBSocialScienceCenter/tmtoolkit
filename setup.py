"""
tmtoolkit setuptools based setup module
"""

import os
from codecs import open

from setuptools import setup, find_packages

__title__ = 'tmtoolkit'
__version__ = '0.11.0-dev'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'


GITHUB_URL = 'https://github.com/WZBSocialScienceCenter/tmtoolkit'

DEPS_BASE = ['numpy>=1.22.0', 'scipy>=1.7.0', 'globre>=0.1.5',
             'pandas>=1.3.0', 'xlrd>=2.0.0', 'openpyxl>=3.0.0',
             'matplotlib>=3.5.0']

DEPS_EXTRA = {
    'textproc': ['spacy>=3.2.0', 'bidict>=0.21.0', 'loky>=3.0.0'],
    'textproc_extra': ['PyICU>=2.8', 'nltk>=3.6.0'],
    'wordclouds': ['wordcloud>=1.8.0,<1.9', 'Pillow>=9.0.0'],
    'lda': ['lda>=2.0'],
    'sklearn': ['scikit-learn>=1.0.0'],
    'gensim': ['gensim>=4.1.0'],
    'topic_modeling_eval_extra': ['gmpy2>=2.1.0'],
    'test': ['pytest>=6.2.0', 'hypothesis>=6.35.0'],
    'doc': ['Sphinx>=4.3.0', 'sphinx-rtd-theme>=1.0.0', 'nbsphinx>=0.8.0'],
    'dev': ['coverage>=6.2', 'coverage-badge>=1.1.0', 'pytest-cov>=3.0.0', 'twine>=3.7.0',
            'ipython>=7.31.0', 'jupyter>=1.0.0', 'notebook>=6.4.0', 'tox>=3.24.0'],
}

# DEPS_EXTRA['minimal'] = DEPS_BASE   # doesn't work with extras_require and pip currently
# see https://github.com/pypa/setuptools/issues/1139

DEPS_EXTRA['recommended'] = DEPS_EXTRA['textproc'] + DEPS_EXTRA['wordclouds']
DEPS_EXTRA['all'] = []
for k, deps in DEPS_EXTRA.items():
    if k not in {'recommended', 'all'}:
        DEPS_EXTRA['all'].extend(deps)

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=__title__,
    version=__version__,
    description='Text Mining and Topic Modeling Toolkit',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url=GITHUB_URL,
    project_urls={
        'Bug Reports': GITHUB_URL + '/issues',
        'Source': GITHUB_URL,
    },

    author=__author__,
    author_email='markus.konrad@wzb.eu',

    license=__license__,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',

        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],

    keywords='textmining textanalysis text mining analysis preprocessing topicmodeling topic modeling evaluation',

    packages=find_packages(exclude=['tests', 'examples']),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=DEPS_BASE,
    extras_require=DEPS_EXTRA
)
