.. _development:

Development
===========

This part of the documentation serves as developer documentation, i.e. a help for those who want to contribute to the development of the package.


Project overview
----------------

This project aims to provide a Python package that allows text processing, text mining and topic modeling with

- easy installation,
- extensive documentation,
- clear programming interface,
- good performance on large datasets.

All computations need to be performed in memory. Streaming from disk is not supported so far.

The package is written in Python and uses other packages for key tasks:

- `SpaCy <https://spacy.io/>`_ is used for the text processing and text mining tasks
- `lda <http://pythonhosted.org/lda/>`_, `gensim <https://radimrehurek.com/gensim/>`_ or `scikit-learn <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html>`_ are used for computing topic models

The package's dependencies are only installed on demand. There's a setup routine that provides an interface for easy installation of SpaCy's language models.

Text processing and normalization is often used to construct a Bag-of-Words (BoW) model which in turn is the input for topic models.


Folder structure
----------------

The project's root folder contains files for documentation generation (``.readthedocs.yaml``), testing (``conftest.py``, ``coverage.svg``, ``tox.ini``) as well as project management and package building (``Makefile``, ``MANIFEST.in``, ``setup.py``). The subfolders include:

- ``.github/worflows``: provides CI configuration for GitHub actions,
- ``doc``: documentation source and built documentation files,
- ``examples``: example scripts and data to show some of the features (most features are better explained in the tutorial which is part of the documentation),
- ``scripts``: scripts used for preparing datasets that come along with the package,
- ``tests``: test suite,
- ``tmtoolkit``: package source code.


API style
---------

The tmtoolkit package provides a *functional API*. This is quite different from object-oriented APIs that are found in many other Python packages, where a programmer mainly uses classes and their methods that are exposed by an API. The tmtoolkit API on the other hand mainly exposes data structures and functions that operate on these data structures. Python classes are usually used to implement more complex data structures such as documents or document corpora, but these classes don't provide methods. Rather, they are used as function arguments, for example as in the large set of *corpus functions* that operate on text corpora as explained below.


Implementation details
----------------------

Top-level module and setup routine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``__main__.py`` file provides a command-line interface for the package. It's only purpose is to allow easy installation of SpaCy language models via the :ref:`setup routine <setup>`. The ``tokenseq`` module provides functions that operate on single (string) tokens or sequences of tokens. These functions are used mainly internally in the ``corpus`` module, but are also exposed by the API to be used from a package user. The ``utils.py`` module provides helper functions used internally throughout the package, but also to be possibly used from a package user.

``bow`` module
^^^^^^^^^^^^^^

This module provides functions for generating document-term-matrices (DTMs), which are central to the BoW concept, and some common statistics used for these matrices.

``corpus`` module
^^^^^^^^^^^^^^^^^

This is the central module for text processing and text mining.

At the core of this module, there is the :class:`~tmtoolkit.corpus.Corpus` class. It takes documents with raw text as input (i.e. a dict mapping *document labels* to text strings) and applies a SpaCy NLP pipeline to it. After that, the corpus consists of  :class:`~tmtoolkit.corpus.Document` objects which contain the textual data in tokenized form, i.e. as a sequence of *tokens* (roughly translated as "words" but other text contents such as numbers and punctation also form separate tokens). Each token comes along with several *token attributes* which were estimated using the NLP pipeline. Examples for token attributes include the Part-of-Speech tag or the lemma.

The :class:`~tmtoolkit.corpus.Document` class stores the tokens and their "standard" attributes in a *token matrix*. This matrix is of shape *(N, M)* for *N* tokens and with *M* attributes. There are at least 2 or 3 attributes: ``whitespace`` (boolean – is there a whitespace after the token?), ``token`` (the actual token string) and optionally ``sent_start`` (only when sentence information is parsed in the NLP pipeline).

The token matrix is a *uint64* matrix as it stores all information as *64 bit hash values*. This reduces memory usage and allows faster computations and data modifications. E.g., when you transform a token (lets say "Hello" to "hello"), you only do one transformation, calculate one new hash value and replace each occurrence of the old hash with the new hash. The hashes are calculated with SpaCy's `hash_string <https://spacy.io/api/stringstore#hash_string>`_ function. For fast conversion between token/attribute hashes and strings, the mappings are stored in a *bidirectional dictionary* using the `bidict <https://pypi.org/project/bidict/>`_ package. Each column, i.e. each attribute, in the token matrix has a separate bidict in the  ``bimaps`` dictionary that is shared between a corpus and each Document object. Using bidict proved to be *much* faster than using SpaCy's built in `Vocab / StringStore <https://spacy.io/api/stringstore>`_.

Besides "standard" token attributes that come from the SpaCy NLP pipeline, a user may also add custom token attributes. These are stored in each document's :attr:`~tmtoolkit.corpus.Document.custom_token_attrs` dictionary that map a attribute name to a NumPy array. Besides token attributes, there are also *document attributes*. These are attributes attached to each document, for example the *document label* (unique document identifier). Custom document attributes can be added, e.g. to record the publication year of a document.

Corpus - As all other modules

corpus functions

inplace

decorators

Parallel processing


``topicmod`` module
^^^^^^^^^^^^^^^^^^^

This is the central module for computing, evaluating and analyzing topic models.

Automated testing
-----------------


Generating documentation
------------------------


Continuous integration
----------------------


Release management
------------------
