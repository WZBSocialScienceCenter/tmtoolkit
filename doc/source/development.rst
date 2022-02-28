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
