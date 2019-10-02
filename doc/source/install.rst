.. _install:

Installation
============

The package is available on `PyPI <https://pypi.org/project/tmtoolkit/>`_ and can be installed via Python package
manager *pip*:

.. code-block:: text

    # recommended installation
    pip install -U tmtoolkit[excel_export,plotting,wordclouds]

    # minimal installation:
    pip install -U tmtoolkit

The package is about 13MB big, because it contains some additional German language model data for POS tagging.

.. note::

    About installation error *mysql_config: not found***: The package *Pattern*, which tmtoolkit uses for lemmatization
    of German-language text, requires to install the *mysqlclient* package (for reasons unknown to me). This in turn
    requires that the program ``mysql_config`` is installed. You can install it with the system package
    *libmysqlclient-dev*, e.g. via ``sudo apt install libmysqlclient-dev`` on Debian based Linux.
    This is sufficient -- **you do not need to install a full MySQL server!**

    In the future, I will try to remove the dependency on Pattern so that this program doesn't need to be installed
    anymore.

.. note::

    If upgrading from an older version to 0.6.0 or above, you will need to uninstall tmtoolkit first
    (run ``pip uninstall tmtoolkit``), before re-installing (using one of the commands described above).

Requirements
------------

**tmtoolkit works with Python 3.5 or above.**

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

Required packages and data files
--------------------------------

All required Python packages are installed automatically along with tmtoolkit when using *pip*. The list of exact
package requirements is in
`requirements.txt <https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/requirements.txt>`_.

.. note::

    You will need to install several corpora and language models from NLTK if you didn't do so yet. You can run the
    following Python code to download all necessary data:

.. code-block::

    import nltk
    nltk.download(['averaged_perceptron_tagger', 'punkt', 'stopwords', 'wordnet'])

Alternatively, you can run ``python -c 'import nltk; nltk.download()'`` in the console. This will open a graphical
downloader interface where you can select the data packages for download.

.. _optional_packages:

Optional packages
-----------------

PyPI packages which can be installed via pip are written *italic*.

* for plotting/visualizations: *matplotlib*
* for the word cloud functions: *wordcloud* and *Pillow*
* for Excel export: *openpyxl*
* for topic modeling, one of the LDA implementations: *lda*, *scikit-learn* or *gensim*
* for additional topic model coherence metrics: *gensim*

For LDA evaluation metrics ``griffiths_2004`` and ``held_out_documents_wallach09`` it is necessary to install

`gmpy2 <https://github.com/aleaxit/gmpy>`_ for multiple-precision arithmetic. This in turn requires installing some C
header libraries for GMP, MPFR and MPC. On Debian/Ubuntu systems this is done with:

.. code-block:: text

    sudo apt install libgmp-dev libmpfr-dev libmpc-dev

After that, gmpy2 can be installed via *pip*.

So for the full set of features, you should run the following (optionally adding gmpy2 if you have installed the
above requirements):

.. code-block:: text

    pip install -U matplotlib wordcloud Pillow openpyxl lda scikit-learn gensim

