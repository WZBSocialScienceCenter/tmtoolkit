tmtookit: Text mining and topic modeling toolkit
================================================

*tmtoolkit* is a set of tools for text mining and topic modeling with Python developed especially for the use in the
social sciences. It aims for easy installation, extensive documentation and a clear programming interface while
offering good performance on large datasets by the means of vectorized operations (via NumPy) and parallel computation
(using Python's *multiprocessing* module). It combines several known and well-tested packages such as
`NLTK <http://www.nltk.org/>`_ or `SciPy <https://scipy.org/>`_.

At the moment, tmtoolkit focuses on methods around the *Bag-of-words* model, but word embeddings may be integrated in
the future.

The documentation for tmtoolkit is available on `tmtoolkit.readthedocs.org <https://tmtoolkit.readthedocs.org>`_.

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
  (`Chuang et al. 2012 <https://dl.acm.org/citation.cfm?id=2254572>`_), topic-word relevancy
  (`Sievert and Shirley 2014 <https://www.aclweb.org/anthology/W14-3110>`_)
* export estimated document-topic and topic-word distributions to Excel
* visualize topic-word distributions and document-topic distributions as word clouds or heatmaps
  (see `lda_visualization Jupyter Notebook <https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/examples/lda_visualization.ipynb>`_)
* integrate `PyLDAVis <https://pyldavis.readthedocs.io/en/latest/>`_ to visualize results
* coherence for individual topcis (see
  `model_coherence Jupyter Notebook <https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/examples/model_coherence.ipynb>`_)


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
:meth:`tmtoolkit.corpus.Corpus.from_builtin_corpus`:

* ``'english-NewsArticles'``: dai, tianru, 2017, "News Articles", https://doi.org/10.7910/DVN/GMFCTR, Harvard Dataverse,
  V1
* ``'german-bt18_speeches_sample'``: Random sample of speeches from the 18th German Bundestag;
  https://github.com/Datenschule/offenesparlament-data

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

License
=======

Code licensed under `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
See `LICENSE <https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/LICENSE>`_ file.

