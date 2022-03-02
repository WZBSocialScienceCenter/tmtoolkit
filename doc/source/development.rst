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


Packaging and dependency management
-----------------------------------

This package uses `setuptools <https://setuptools.pypa.io/en/latest/index.html>`_ for packaging. All package metadata and dependencies are defined in ``setup.py``. Since tmtoolkit allows installing dependencies on demand, there are several installation options defined in ``setup.py``. For development, the most important are:

- ``[dev]``: installs packages for development and packaging
- ``[test]``: installs packages for testing tmtoolkit
- ``[doc]``: installs packages for generating the documentation
- ``[all]``: installs all required and optional packages

The ``requirements.txt`` and ``requirements_doc.txt`` files simply point to the ``[all]`` and ``[doc]`` installation options.

The ``Makefile`` in the root folder contains targets for generating a Python *Wheel* package (``make wheel``) and a Python source distribution package (``make sdist``).


Built-in datasets
-----------------

All built-in datasets reside in ``tmtoolkit/data/<LANGUAGE_CODE>``, where ``LANGUAGE_CODE`` is an ISO language code. For the `ParlSpeech V2 <https://doi.org/10.7910/DVN/L4OAKN>`_ datasets, the samples are generated via the R script ``scripts/prepare_corpora.R``. The `News Articles <https://doi.org/10.7910/DVN/GMFCTR>`_ dataset is used without further processing.


Automated testing
-----------------

The tmtoolkit package relies on the following packages for testing:

- `pytest <https://pytest.org/>`_ as testing framework
- `hypothesis <https://hypothesis.readthedocs.io/>`_ for property-based testing
- `coverage <https://coverage.readthedocs.io/>`_ for measuring test coverage of the code
- `tox <https://tox.wiki/>`_ for checking packaging and running tests in different virtual environments

All tests are implemented in the ``tests`` directory and prefixed by ``test_``. The ``conftest.py`` file contains project-wide test configuration. The ``tox.ini`` file contains configuration for setting up the virtual environments for tox. For each release, tmtoolkit aims to support the last three major Python release versions, e.g. 3.8, 3.9 and 3.10, and all of these are tested with tox along with different dependency configurations from *minimal* to *full*. To use different versions of Python on the same system, it's recommended to use the `deadsnakes repository <https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa>`_ on Ubuntu or Debian Linux.

The ``Makefile`` in the root folder contains a target for generating coverage reports and the coverage badge (``make cov_tests``).


Documentation
-------------

The `Sphinx <https://www.sphinx-doc.org/>`_ package is used for documentation. All objects exposed by the API are documented in the Sphinx format. All other parts of the documentation reside in ``doc/source``. The configuration for Sphinx lies in ``doc/source/conf.py``. The `nbsphinx <https://nbsphinx.readthedocs.io/>`_ package is used for generating the tutorial from Jupyter Notebooks which are also located in ``doc/source``.

The ``Makefile`` in the ``doc`` folder has several targets for generating the documentation. These are:

- ``make notebooks`` – run all notebooks to generate their outputs; these are stored in-place
- ``make clean`` – remove everything under ``doc/build``
- ``make html`` – generate the HTML documentation from the documentation source

The generated documentation then resides under ``doc/build``.

The documentation is published at `tmtoolkit.readthedocs.io <https://tmtoolkit.readthedocs.io/en/latest/>`_. For this, new commits to the master branch of the GitHub project or new tags are automatically built by `readthedocs.org <https://readthedocs.org/>`_. The ``.readthedocs.yaml`` file in the root folder sets up the build process for readthedocs.org.


Continuous integration
----------------------

Continuous integration routines are achieved via `GitHub Actions (GA) <https://docs.github.com/en/actions>`_. For tmtoolkit, this so far only means automatic testing for new commits and releases on different machine configurations.

The GA set up for the tests is done in ``.github/worflows/runtests.yml``. There are "minimal" and "full" test suites for Ubuntu, MacOS and Windows with Python versions 3.8, 3.9 and 3.10 each, which means 18 jobs are spawned. Again, tox is used for running the tests on the machines.


Release management
------------------



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

At the core of this module, there is the :class:`~tmtoolkit.corpus.Corpus` class implemented in ``corpus/_corpus.py``. It takes documents with raw text as input (i.e. a dict mapping *document labels* to text strings) and applies a SpaCy NLP pipeline to it. After that, the corpus consists of  :class:`~tmtoolkit.corpus.Document` (implemented in ``corpus/_document.py``) objects which contain the textual data in tokenized form, i.e. as a sequence of *tokens* (roughly translated as "words" but other text contents such as numbers and punctation also form separate tokens). Each token comes along with several *token attributes* which were estimated using the NLP pipeline. Examples for token attributes include the Part-of-Speech tag or the lemma.

The :class:`~tmtoolkit.corpus.Document` class stores the tokens and their "standard" attributes in a *token matrix*. This matrix is of shape *(N, M)* for *N* tokens and with *M* attributes. There are at least 2 or 3 attributes: ``whitespace`` (boolean – is there a whitespace after the token?), ``token`` (the actual token string) and optionally ``sent_start`` (only when sentence information is parsed in the NLP pipeline).

The token matrix is a *uint64* matrix as it stores all information as *64 bit hash values*. This reduces memory usage and allows faster computations and data modifications. E.g., when you transform a token (lets say "Hello" to "hello"), you only do one transformation, calculate one new hash value and replace each occurrence of the old hash with the new hash. The hashes are calculated with SpaCy's `hash_string <https://spacy.io/api/stringstore#hash_string>`_ function. For fast conversion between token/attribute hashes and strings, the mappings are stored in a *bidirectional dictionary* using the `bidict <https://pypi.org/project/bidict/>`_ package. Each column, i.e. each attribute, in the token matrix has a separate bidict in the  ``bimaps`` dictionary that is shared between a corpus and each Document object. Using bidict proved to be *much* faster than using SpaCy's built in `Vocab / StringStore <https://spacy.io/api/stringstore>`_.

Besides "standard" token attributes that come from the SpaCy NLP pipeline, a user may also add custom token attributes. These are stored in each document's :attr:`~tmtoolkit.corpus.Document.custom_token_attrs` dictionary that map a attribute name to a NumPy array. Besides token attributes, there are also *document attributes*. These are attributes attached to each document, for example the *document label* (unique document identifier). Custom document attributes can be added, e.g. to record the publication year of a document.

The :class:`~tmtoolkit.corpus.Corpus` class implements a data structure for text corpora with named documents. All these documents are stored in the corpus as :class:`~tmtoolkit.corpus.Document` objects. *Corpus functions* allow to operate on Corpus objects. They are implemented in ``corpus/_corpusfuncs.py``. All corpus functions that transform/modify a corpus, have an ``inplace`` argument, by default set to ``True``. If  ``inplace`` is set to ``True``, the corpus will be directly modified in-place, i.e. modifying the input corpus. If ``inplace`` is set to ``False``, a copy of the input corpus is created and all modifications are applied to this copy. The original input corpus is not altered in that case. The ``corpus_func_inplace_opt`` decorator is used to mark corpus functions with the in-place option.

The :class:`~tmtoolkit.corpus.Corpus` class provides parallel processing capabilities for processing large data amounts. This can be controlled with the ``max_workers`` argument. Parallel processing is then enabled at two stages: First, it is simply enabled for the SpaCy NLP pipeline by setting up the pipeline accordingly. Second, a *reusable process pool executor* is created by the means of `loky <https://github.com/joblib/loky/>`_. This process pool is then used in corpus functions whenever parallel execution is beneficial over serial execution. The ``parallelexec`` decorator is used to mark (inner) functions for parallel execution.


``topicmod`` module
^^^^^^^^^^^^^^^^^^^

This is the central module for computing, evaluating and analyzing topic models.

In ``topicmod/evaluate.py`` there are mainly several evaluation metrics for topic models implemented. Topic models can be computed and evaluated in parallel, the base code for that is in ``topicmod/parallel.py``. Three modules use the base classes from ``topicmod/parallel.py`` to implement interfaces to popular topic modeling packages:

- ``topicmod/tm_gensim.py`` for `gensim <https://radimrehurek.com/gensim/>`_
- ``topicmod/tm_lda.py`` for `lda <http://pythonhosted.org/lda/>`_
- ``topicmod/tm_sklearn.py`` for `scikit-learn <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html>`_

