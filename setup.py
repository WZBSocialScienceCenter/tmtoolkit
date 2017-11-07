"""
tmtoolkit setuptools based setup module
"""

from setuptools import setup


setup(
    name='tmtoolkit',
    version='0.1.0',

    description='A set of tools for text mining / text analysis',
    # TODO:
#     long_description="""This repository contains a set of tools written in Python 3 with the aim to extract tabular
# data from scanned and OCR-processed documents available as PDF files. Before these files can be processed they need
# to be converted to XML files in pdf2xml format using poppler utils. Further information and examples can be found
# in the github repository.""",
#
#     url='https://github.com/WZBSocialScienceCenter/pdftabextract',

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

    keywords='textmining textanalysis text mining analysis preprocessing',

    packages=['tmtoolkit'],
    include_package_data=True,
    python_requires='>=2.7',
    install_requires=['six', 'numpy', 'scipy', 'pandas', 'nltk'],
    extras_require={
        'improved_german_lemmatization':  ['pyphen', 'pattern'],
        'excel_export': ['openpyxl'],
        'plotting': ['matplotlib'],
        'topic_modeling_lda': ['lda'],
        'topic_modeling_sklearn': ['scikit-learn'],
        'topic_modeling_gensim': ['gensim'],
        'topic_modeling_eval_griffiths_2004': ['gmpy2'],
    }
)
