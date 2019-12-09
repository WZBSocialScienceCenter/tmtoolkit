"""
tmtoolkit setuptools based setup module
"""

import os
from codecs import open

from setuptools import setup, find_packages
import tmtoolkit

GITHUB_URL = 'https://github.com/WZBSocialScienceCenter/tmtoolkit'

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=tmtoolkit.__title__,
    version=tmtoolkit.__version__,
    description='Text Mining and Topic Modeling Toolkit',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url=GITHUB_URL,
    project_urls={
        'Bug Reports': GITHUB_URL + '/issues',
        'Source': GITHUB_URL,
    },

    author=tmtoolkit.__author__,
    author_email='markus.konrad@wzb.eu',

    license=tmtoolkit.__license__,

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

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],

    keywords='textmining textanalysis text mining analysis preprocessing topicmodeling topic modeling evaluation',

    packages=find_packages(exclude=['tests', 'examples']),
    include_package_data=True,
    python_requires='>=3.6, <=3.7',
    install_requires=['numpy>=1.17.0', 'scipy>=1.3.0', 'pandas>=0.25.0', 'datatable>=0.10.0', 'nltk>=3.4.0',
                      'globre>=0.1.5', 'matplotlib>=3.1.0', 'germalemma>=0.1.2', 'deprecation>=2.0.0'],
    extras_require={
        'excel_export': ['openpyxl'],
        'wordclouds': ['wordcloud', 'Pillow'],
        'topic_modeling_lda': ['lda'],
        'topic_modeling_sklearn': ['scikit-learn>=0.22'],
        'topic_modeling_gensim': ['gensim>=3.8.0'],
        'topic_modeling_eval_extra': ['gmpy2'],
    }
)
