.. _install:

Installation
============

Requirements
------------

**tmtoolkit works with Python 3.8 or newer (tested up to Python 3.10).**

Requirements are automatically installed via *pip* as described below. Additional packages can also be installed
via *pip* for certain use cases (see :ref:`optional_packages`).


Installation instructions
-------------------------

The package *tmtoolkit* is available on `PyPI <https://pypi.org/project/tmtoolkit/>`_ and can be installed via
Python package manager *pip*. It is highly recommended to install tmtoolkit and its dependencies in a separate
`Python Virtual Environment ("venv") <https://docs.python.org/3/tutorial/venv.html>`_ and upgrade to the latest
*pip* version (you may also choose to install
`virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_, which makes managing venvs a lot
easier).

Creating and activating a venv *without* virtualenvwrapper:

.. code-block:: text

    python3 -m venv myenv

    # activating the environment (on Windows type "myenv\Scripts\activate.bat")
    source myenv/bin/activate

Alternatively, creating and activating a venv *with* virtualenvwrapper:

.. code-block:: text

    mkvirtualenv myenv

    # activating the environment
    workon myenv

Upgrading pip (*only* do this when you've activated your venv):

.. code-block:: text

    pip install -U pip

The tmtoolkit package is highly modular and tries to install as few software dependencies as possible. So in order to
install tmtoolkit, you can first choose if you want a minimal installation or install a recommended set of
packages that enable most features. For the recommended installation, you can type **one of the following**, depending
on the preferred package for topic modeling:

.. code-block:: text

    # recommended installation without topic modeling
    pip install -U "tmtoolkit[recommended]"

    # recommended installation with "lda" for topic modeling
    pip install -U "tmtoolkit[recommended,lda]"

    # recommended installation with "scikit-learn" for topic modeling
    pip install -U "tmtoolkit[recommended,sklearn]"

    # recommended installation with "gensim" for topic modeling
    pip install -U "tmtoolkit[recommended,gensim]"

    # you may also select several topic modeling packages
    pip install -U "tmtoolkit[recommended,lda,sklearn,gensim]"

The **minimal** installation will only install a base set of dependencies and will only enable the modules for BoW
statistics, token sequence operations, topic modeling and utility functions. You can install it as follows:

.. code-block:: text

    # alternative installation if you only want to install a minimum set of dependencies
    pip install -U tmtoolkit

.. note:: The tmtoolkit package is about 7MB big, because it contains some example corpora.

.. _setup:

**After that, you should initially run tmtoolkit's setup routine.** This makes sure that all required data files are
present and downloads them if necessary. You should specify a list of languages for which language models should be
downloaded and installed. The list of available language models corresponds with the models provided by
`SpaCy <https://spacy.io/usage/models#languages>`_ (except for "multi-language"). You need to specify the two-letter ISO
language code for the language models that you want to install. **Don't use spaces in the list of languages.**
E.g. in order to install models for English and German:

.. code-block:: text

    python -m tmtoolkit setup en,de

To install *all* available language models, you can run:

.. code-block:: text

    python -m tmtoolkit setup all

.. _optional_packages:

Optional packages
-----------------

For additional features, you can install further packages using the following installation options:

- ``pip install -U tmtoolkit[textproc_extra]`` for Unicode normalization and simplification and for stemming with *nltk*
- ``pip install -U tmtoolkit[wordclouds]`` for generating word clouds
- ``pip install -U tmtoolkit[lda]`` for topic modeling with LDA
- ``pip install -U tmtoolkit[sklearn]`` for topic modeling with scikit-learn
- ``pip install -U tmtoolkit[gensim]`` for topic modeling and additional evaluation metrics with Gensim
- ``pip install -U tmtoolkit[topic_modeling_eval_extra]`` for topic modeling evaluation metrics ``griffiths_2004`` and
  ``held_out_documents_wallach09`` (see further information below)

For LDA evaluation metrics ``griffiths_2004`` and ``held_out_documents_wallach09`` it is necessary to install
`gmpy2 <https://github.com/aleaxit/gmpy>`_ for multiple-precision arithmetic. This in turn requires installing some C
header libraries for GMP, MPFR and MPC. On Debian/Ubuntu systems this is done with:

.. code-block:: text

    sudo apt install libgmp-dev libmpfr-dev libmpc-dev
