tmtookit: Text mining and topic modeling toolkit
================================================

*tmtoolkit* is a set of tools for text mining and topic modeling with Python developed especially for the use in the
social sciences. It aims for easy installation, extensive documentation and a clear programming interface while
offering good performance on large datasets by the means of vectorized operations (via NumPy) and parallel computation
(using Python's *multiprocessing* module). It combines several known and well-tested packages such as
`NLTK <http://www.nltk.org/>`_ or `SciPy <https://scipy.org/>`_.

At the moment, tmtoolkit focuses on methods around the *Bag-of-words* model, but word embeddings may be integrated in
the future.

The documentation for tmtoolkit is available on `tmtoolkit.readthedocs.org <https://tmtoolkit.readthedocs.org>`_ and
the GitHub code repository is on
`github.com/WZBSocialScienceCenter/tmtoolkit <https://github.com/WZBSocialScienceCenter/tmtoolkit>`_.

Features
--------

Text preprocessing
^^^^^^^^^^^^^^^^^^

tmtoolkit implements several preprocessing methods, including:

* tokenization
* part-of-speech (POS) tagging
* lemmatization
* stemming
* cleaning tokens
* filtering tokens
* filtering documents
* generating n-grams
* generating document-term matrices
* keywords-in-context (KWIC)
* "glueing" of specified subsequent tokens, e.g. ``["Brad", "Pitt"]`` becomes ``["Brad_Pitt"]``

All text preprocessing methods can operate in parallel to speed up computations with large datasets.

Topic modeling
^^^^^^^^^^^^^^

* model computation in parallel for different copora and/or parameter sets
* support for `lda <http://pythonhosted.org/lda/>`_,
  `scikit-learn <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html>`_
  and `gensim <https://radimrehurek.com/gensim/>`_ topic modeling backends
* evaluation of topic models (e.g. in order to an optimal number of topics for a given dataset) using several
  implemented metrics:

   * model coherence (`Mimno et al. 2011 <https://dl.acm.org/citation.cfm?id=2145462>`_) or with
     `metrics implemented in Gensim <https://radimrehurek.com/gensim/models/coherencemodel.html>`_)
   * KL divergence method (`Arun et al. 2010 <http://doi.org/10.1007/978-3-642-13657-3_43>`_)
   * probability of held-out documents (`Wallach et al. 2009 <https://doi.org/10.1145/1553374.1553515>`_)
   * pair-wise cosine distance method (`Cao Juan et al. 2009 <http://doi.org/10.1016/j.neucom.2008.06.011>`_)
   * harmonic mean method (`Griffiths, Steyvers 2004 <http://doi.org/10.1073/pnas.0307752101>`_)
   * the loglikelihood or perplexity methods natively implemented in lda, sklearn or gensim

* plotting of evaluation results
* common statistics for topic models such as word saliency and distinctiveness
  (`Chuang et al. 2012 <https://dl.acm.org/citation.cfm?id=2254572>`_), topic-word relevance
  (`Sievert and Shirley 2014 <https://www.aclweb.org/anthology/W14-3110>`_)
* finding topics according to word patterns
* export estimated document-topic and topic-word distributions to Excel
* visualize topic-word distributions and document-topic distributions as word clouds or heatmaps
* coherence for individual topics
* integrate `PyLDAVis <https://pyldavis.readthedocs.io/en/latest/>`_ to visualize results


Other features
^^^^^^^^^^^^^^

* loading and cleaning of raw text from text files, tabular files (CSV or Excel), ZIP files or folders
* common statistics and transformations for document-term matrices like word cooccurrence and *tf-idf*


Limits
------

* currently only German and English language texts are supported for language-dependent text preprocessing methods
  such as POS tagging or lemmatization
* all data must reside in memory, i.e. no streaming of large data from the hard disk (which for example
  `Gensim <https://radimrehurek.com/gensim/>`_ supports)
* no direct support of word embeddings


Built-in datasets
-----------------

Currently tmtoolkit comes with the following built-in datasets which can be loaded via
`tmtoolkit.corpus.Corpus.from_builtin_corpus()`:

* ``'english-NewsArticles'``: dai, tianru, 2017, "News Articles", https://doi.org/10.7910/DVN/GMFCTR, Harvard Dataverse,
  V1
* ``'german-bt18_speeches_sample'``: Random sample of speeches from the 18th German Bundestag;
  https://github.com/Datenschule/offenesparlament-data


About this documentation
------------------------

This documentation guides you in several chapters from installing tmtoolkit to its specific use cases and shows some
examples with built-in corpora and other datasets. All "hands on" chapters from `Getting started <getting_started.ipynb>`_
to `Topic modeling <topic_modeling.ipynb>`_ are generated from `Jupyter Notebooks <https://jupyter.org/>`_. If you want
to follow along using these notebooks, you can
`download them from the GitHub repository <https://github.com/WZBSocialScienceCenter/tmtoolkit/tree/master/doc/source>`_.

There are also a few other examples as plain Python scripts available in the
`examples folder <https://github.com/WZBSocialScienceCenter/tmtoolkit/tree/master/examples>`_ of the GitHub repository.
.. _install:

Installation
============

The package *tmtoolkit* is available on `PyPI <https://pypi.org/project/tmtoolkit/>`_ and can be installed via
Python package manager *pip*. It is highly recommended to install tmtoolkit and its dependencies in a
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

Now in order to install tmtoolkit, you can choose if you want a minimal installation or install a recommended set of
packages that enable most features. For the recommended installation, you can type **one of the following**, depending on
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

**Note:** For Linux and MacOS users, it's also recommended to install the *datatable* package (see "Optional packages"),
which makes many operations faster and more memory efficient.

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
use cases (see "Optional packages").

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
  Note that *datatable* is currently only available for Linux and MacOS on Python 3.6 and 3.7.
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


License
=======

Code licensed under `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
See `LICENSE <https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/LICENSE>`_ file.

