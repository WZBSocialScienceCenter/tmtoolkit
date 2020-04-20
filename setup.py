"""
tmtoolkit setuptools based setup module
"""

import os
from codecs import open

from setuptools import setup, find_packages

__title__ = 'tmtoolkit'
__version__ = '0.10.0-dev'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'


GITHUB_URL = 'https://github.com/WZBSocialScienceCenter/tmtoolkit'

DEPS_BASE = ['numpy>=1.18.0', 'scipy>=1.4.0', 'pandas>=1.0.0', 'xlrd>=1.2.0',
             'globre>=0.1.5', 'matplotlib>=3.2.0', 'deprecation>=2.0.0', 'spacy>=2.2.0']

DEPS_EXTRA = {
    'datatable': ['datatable>=0.10.0'],
    'nltk': ['nltk>=3.5.0'],
    'excel_export': ['openpyxl>=3.0.0'],
    'wordclouds': ['wordcloud>=1.6.0', 'Pillow>=7.1.0'],
    'lda': ['lda>=1.1.0'],
    'sklearn': ['scikit-learn>=0.22'],
    'gensim': ['gensim>=3.8.0'],
    'topic_modeling_eval_extra': ['gmpy2>=2.0.0'],
    'test': ['pytest>=5.4.0', 'hypothesis>=5.10.0', 'decorator>=4.4.0'],
    'dev': ['Sphinx>=3.0.0', 'nbsphinx>=0.6.1', 'sphinx-rtd-theme>=0.4.0',
            'coverage>=5.1', 'pytest-cov>=2.8.1', 'twine>=3.1.0',
            'ipython>=7.0.0', 'jupyter>=1.0.0', 'notebook>=6.0.0'],
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
