.. _install:

Installation
============

The package *tmtoolkit* is available on `PyPI <https://pypi.org/project/tmtoolkit/>`_ and can be installed via
Python package manager *pip*. **It is highly recommended to install tmtoolkit and its dependencies in a
`Python Virtual Environment *("venv")* <https://docs.python.org/3/tutorial/venv.html>`_** and upgrade to the latest
*pip* version (you may also choose to install
*`virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>_`*, which makes managing venvs a lot easier).

Creating and activating a venv *without* virtualenvwrapper:

.. code-block:: text

    python3 -m venv myenv

    # activating the environment
    source myenv/bin/activate

Creating and activating a venv *with* virtualenvwrapper:

.. code-block:: text

    mkvirtualenv myenv

    # activating the environment
    workon myenv

Upgrading pip (*only* do this when you've activated your venv):

.. code-block:: text

    pip install -U pip

Now in order to install tmtoolkit, you can choose if you want a minimal installation or install a recommended set of
packages that enable all features. For the recommended installation, you can type **one of the following**, depending on
the preferred package for topic modeling:

.. code-block:: text

    # recommended installation without topic modeling
    pip install -U tmtoolkit[recommended]

    # recommended installation with "lda" for topic modeling
    pip install -U tmtoolkit[recommended,lda]

    # recommended installation with "scikit-learn" for topic modeling
    pip install -U tmtoolkit[recommended,sklearn]

    # recommended installation with "gensim" for topic modeling
    pip install -U tmtoolkit[recommended,gensim]

    # you may also select several topic modeling packages
    pip install -U tmtoolkit[recommended,lda,sklearn,gensim]

For the minimal installation, you can just do:

.. code-block:: text

    pip install -U tmtoolkit

.. note::
    For Linux and MacOS users, it's also recommended to install the *datatable* package (see :ref:`optional_packages`),
    which makes many operations faster and more memory efficient.

.. note::
    The tmtoolkit package is about 19MB big, because it contains some example corpora and additional German language
    model data for POS tagging.

After that, you should initially run tmtoolkit's setup routine. This makes sure that all required data files are
present and downloads them if necessary:

.. code-block:: text

    python -m tmtoolkit setup


Requirements
------------

**tmtoolkit works with Python 3.6, 3.7 or 3.8.**

Requirements are automatically installed via *pip*. Additional packages can also be installed via *pip* for certain
use cases (see :ref:`optional_packages`).

.. note::

    **A special note for Windows users**: tmtoolkit has been tested on Windows and works well (I recommend using
    the `Anaconda distribution for Python <https://anaconda.org/)>`_ when using Windows). However, you will need to
    wrap all code that uses multi-processing (i.e. all calls to :class:`tmtoolkit.preprocess.TMPreproc` and the
    parallel topic modeling functions) in a ``if __name__ == '__main__'`` block like this:

.. code-block::

    def main():
        # code with multi-processing comes here
        # ...

    if __name__ == '__main__':
        main()


.. _optional_packages:

Optional packages
-----------------

For additional features, you can install further packages from PyPI via pip:

* for faster tabular data creation and access (replaces usage of *pandas* package in most functions): *datatable*.
  note that *datatable* is currently only available for Linux and MacOS on Python 3.6 and 3.7.
* for the word cloud functions: *wordcloud* and *Pillow*.
* for Excel export: *openpyxl*.
* for topic modeling, one of the LDA implementations: *lda*, *scikit-learn* or *gensim*.
* for additional topic model coherence metrics: *gensim*.

For LDA evaluation metrics ``griffiths_2004`` and ``held_out_documents_wallach09`` it is necessary to install
`gmpy2 <https://github.com/aleaxit/gmpy>`_ for multiple-precision arithmetic. This in turn requires installing some C
header libraries for GMP, MPFR and MPC. On Debian/Ubuntu systems this is done with:

.. code-block:: text

    sudo apt install libgmp-dev libmpfr-dev libmpc-dev

After that, gmpy2 can be installed via *pip*.
