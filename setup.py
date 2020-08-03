"""
tmtoolkit setuptools based setup module
"""

import os
from codecs import open

from setuptools import setup, find_packages

__title__ = 'tmtoolkit'
__version__ = '0.10.0'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'


GITHUB_URL = 'https://github.com/WZBSocialScienceCenter/tmtoolkit'

DEPS_BASE = ['numpy>=1.19.0,<2', 'scipy>=1.5.0,<1.6', 'pandas>=1.1.0,<1.2', 'xlrd>=1.2.0',
             'globre>=0.1.5,<0.2', 'matplotlib>=3.3.0,<3.4', 'spacy>=2.3.0,<2.4']

DEPS_EXTRA = {
    'datatable': ['datatable>=0.10.0,<0.11'],
    'nltk': ['nltk>=3.5.0,<3.6'],
    'excel_export': ['openpyxl>=3.0.0'],
    'wordclouds': ['wordcloud>=1.7.0,<1.8', 'Pillow>=7.2.0,<7.3'],
    'lda': ['ldafork>=1.2.0,<1.3'],
    'sklearn': ['scikit-learn>=0.23,<0.24'],
    'gensim': ['gensim>=3.8.0,<3.9'],
    'topic_modeling_eval_extra': ['gmpy2>=2.0.0,<3'],
    'test': ['pytest>=6.0.0,<7', 'hypothesis>=5.23.0<5.24', 'decorator>=4.4.0,<4.5'],
    'dev': ['Sphinx>=3.1.0', 'nbsphinx>=0.7.0', 'sphinx-rtd-theme>=0.5.0',
            'coverage>=5.2', 'coverage-badge>=1.0.0', 'pytest-cov>=2.10.0', 'twine>=3.2.0',
            'ipython>=7.16.0', 'jupyter>=1.0.0', 'notebook>=6.0.0', 'tox>=3.18.0'],
}

DEPS_EXTRA['recommended'] = DEPS_EXTRA['excel_export'] + DEPS_EXTRA['wordclouds']
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],

    keywords='textmining textanalysis text mining analysis preprocessing topicmodeling topic modeling evaluation',

    packages=find_packages(exclude=['tests', 'examples']),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=DEPS_BASE,
    extras_require=DEPS_EXTRA
)
