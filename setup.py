"""
tmtoolkit setuptools based setup module
"""

from setuptools import setup


setup(
    name='tmtoolkit',
    version='0.1.6',
    description='Text Mining and Topic Modeling Toolkit',
    long_description="""tmtoolkit is a set of tools for text mining and topic modeling with Python. It contains
functions for text preprocessing like lemmatization, stemming or POS tagging especially for English and German
texts. Preprocessing is done in parallel by using all available processors on your machine. The topic modeling
features include topic model evaluation metrics, allowing to calculate models with different parameters in parallel
and comparing them (e.g. in order to find the best number of topics for a given set of documents). Topic models can
be generated in parallel for different copora and/or parameter sets using the LDA implementations either from
lda, scikit-learn or gensim.""",
    url='https://github.com/WZBSocialScienceCenter/tmtoolkit',

    author='Markus Konrad',
    author_email='markus.konrad@wzb.eu',

    license='Apache 2.0',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',

        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],

    keywords='textmining textanalysis text mining analysis preprocessing topicmodeling topic modeling evaluation',

    packages=['tmtoolkit', 'tmtoolkit.lda_utils', 'ClassifierBasedGermanTagger'],
    include_package_data=True,
    python_requires='>=2.7',
    install_requires=['six', 'numpy', 'scipy', 'pandas', 'nltk', 'pyphen'],
    extras_require={
        'improved_german_lemmatization':  ['pattern'],
        'excel_export': ['openpyxl'],
        'plotting': ['matplotlib'],
        'topic_modeling_lda': ['lda'],
        'topic_modeling_sklearn': ['scikit-learn'],
        'topic_modeling_gensim': ['gensim'],
        'topic_modeling_eval_griffiths_2004': ['gmpy2'],
    }
)
