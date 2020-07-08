tmtookit: Text mining and topic modeling toolkit
================================================

|pypi| |pypi_downloads| |rtd| |travis| |coverage| |zenodo|

*tmtoolkit* is a set of tools for text mining and topic modeling with Python developed especially for the use in the
social sciences. It aims for easy installation, extensive documentation and a clear programming interface while
offering good performance on large datasets by the means of vectorized operations (via NumPy) and parallel computation
(using Python's *multiprocessing* module). It combines several known and well-tested packages such as
`SpaCy <https://spacy.io/>`_ and `SciPy <https://scipy.org/>`_.

At the moment, tmtoolkit focuses on methods around the *Bag-of-words* model, but word vectors (word embeddings) can
also be generated.

The documentation for tmtoolkit is available on `tmtoolkit.readthedocs.org <https://tmtoolkit.readthedocs.org>`_ and
the GitHub code repository is on
`github.com/WZBSocialScienceCenter/tmtoolkit <https://github.com/WZBSocialScienceCenter/tmtoolkit>`_.

Features
--------

Text preprocessing
^^^^^^^^^^^^^^^^^^

tmtoolkit implements or provides convenient wrappers for several preprocessing methods, including:

* tokenization and `part-of-speech (POS) tagging <https://tmtoolkit.readthedocs.io/en/latest/preprocessing.html#Part-of-speech-(POS)-tagging>`_ (via SpaCy)
* `lemmatization and term normalization <https://tmtoolkit.readthedocs.io/en/latest/preprocessing.html#Lemmatization-and-term-normalization>`_
* extensive `pattern matching capabilities <https://tmtoolkit.readthedocs.io/en/latest/preprocessing.html#Common-parameters-for-pattern-matching-functions>`_
  (exact matching, regular expressions or "glob" patterns) to be used in many
  methods of the package, e.g. for filtering on token, document or document label level, or for
  `keywords-in-context (KWIC) <#Keywords-in-context-(KWIC)-and-general-filtering-methods>`_
* adding and managing `custom token metadata <https://tmtoolkit.readthedocs.io/en/latest/preprocessing.html#Working-with-token-metadata>`_
* accessing
  `word vectors (word embeddings) <https://tmtoolkit.readthedocs.io/en/latest/preprocessing.html#Accessing-tokens,-vocabulary-and-other-important-properties>`_
* generating `n-grams <https://tmtoolkit.readthedocs.io/en/latest/preprocessing.html#Generating-n-grams>`_
* generating `sparse document-term matrices <https://tmtoolkit.readthedocs.io/en/latest/preprocessing.html#Generating-a-sparse-document-term-matrix-(DTM)>`_
* `expanding compound words and "gluing" of specified subsequent tokens
  <https://tmtoolkit.readthedocs.io/en/latest/preprocessing.html#Expanding-compound-words-and-joining-tokens>`_, e.g. ``["White", "House"]`` becomes
  ``["White_House"]``

All text preprocessing methods can operate in parallel to speed up computations with large datasets.

Topic modeling
^^^^^^^^^^^^^^

* `model computation in parallel <https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Computing-topic-models-in-parallel>`_ for different copora
  and/or parameter sets
* support for `lda <http://pythonhosted.org/lda/>`_,
  `scikit-learn <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html>`_
  and `gensim <https://radimrehurek.com/gensim/>`_ topic modeling backends
* `evaluation of topic models <https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Evaluation-of-topic-models>`_ (e.g. in order to an optimal number
  of topics for a given dataset) using several implemented metrics:

   * model coherence (`Mimno et al. 2011 <https://dl.acm.org/citation.cfm?id=2145462>`_) or with
     `metrics implemented in Gensim <https://radimrehurek.com/gensim/models/coherencemodel.html>`_)
   * KL divergence method (`Arun et al. 2010 <http://doi.org/10.1007/978-3-642-13657-3_43>`_)
   * probability of held-out documents (`Wallach et al. 2009 <https://doi.org/10.1145/1553374.1553515>`_)
   * pair-wise cosine distance method (`Cao Juan et al. 2009 <http://doi.org/10.1016/j.neucom.2008.06.011>`_)
   * harmonic mean method (`Griffiths, Steyvers 2004 <http://doi.org/10.1073/pnas.0307752101>`_)
   * the loglikelihood or perplexity methods natively implemented in lda, sklearn or gensim

* `plotting of evaluation results <https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Evaluation-of-topic-models>`_
* `common statistics for topic models <https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Common-statistics-and-tools-for-topic-models>`_ such as
  word saliency and distinctiveness (`Chuang et al. 2012 <https://dl.acm.org/citation.cfm?id=2254572>`_), topic-word
  relevance (`Sievert and Shirley 2014 <https://www.aclweb.org/anthology/W14-3110>`_)
* `finding / filtering topics with pattern matching <https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Filtering-topics>`_
* `export estimated document-topic and topic-word distributions to Excel
  <https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Displaying-and-exporting-topic-modeling-results>`_
* `visualize topic-word distributions and document-topic distributions <https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Visualizing-topic-models>`_
  as word clouds or heatmaps
* model coherence (`Mimno et al. 2011 <https://dl.acm.org/citation.cfm?id=2145462>`_) for individual topics
* integrate `PyLDAVis <https://pyldavis.readthedocs.io/en/latest/>`_ to visualize results

Other features
^^^^^^^^^^^^^^

* `loading and cleaning of raw text from text files, tabular files (CSV or Excel), ZIP files or folders
  <https://tmtoolkit.readthedocs.io/en/latest/text_corpora.html>`_
* `common statistics and transformations for document-term matrices <https://tmtoolkit.readthedocs.io/en/latest/bow.html>`_ like word cooccurrence and *tf-idf*

Limits
------

* all languages are supported, for which `SpaCy language models <https://spacy.io/models>`_ are available
* all data must reside in memory, i.e. no streaming of large data from the hard disk (which for example
  `Gensim <https://radimrehurek.com/gensim/>`_ supports)


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

The tmtoolkit package is about 7MB big, because it contains some example corpora.

After that, you should initially run tmtoolkit's setup routine. This makes sure that all required data files are
present and downloads them if necessary. You should specify a list of languages for which language models should be
downloaded and installed, e.g.

.. code-block:: text

    python -m tmtoolkit setup en,de

To install *all* available language models, you can run:

.. code-block:: text

    python -m tmtoolkit setup all


Requirements
------------

**tmtoolkit works with Python 3.6, 3.7 or 3.8.**

Requirements are automatically installed via *pip*. Additional packages can also be installed via *pip* for certain
use cases (see "Optional packages").

    **A special note for Windows users**: tmtoolkit has been tested on Windows and works well (I recommend using
    the `Anaconda distribution for Python <https://anaconda.org/)>`_ when using Windows). However, you will need to
    wrap all code that uses multi-processing (i.e. all calls to `tmtoolkit.preprocess.TMPreproc` and the
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

* for faster tabular data creation and access (replaces usage of *pandas* package in most functions): *datatable*;
  note that *datatable* is currently only available for Linux and MacOS on Python 3.6 and 3.7
* for the word cloud functions: *wordcloud* and *Pillow*
* for Excel export: *openpyxl*
* for topic modeling, one of the LDA implementations: *lda*, *scikit-learn* or *gensim*
* for additional topic model coherence metrics: *gensim*
* for stemming: *nltk*

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

.. |pypi| image:: https://badge.fury.io/py/tmtoolkit.svg
    :target: https://badge.fury.io/py/tmtoolkit
    :alt: PyPI Version

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/tmtoolkit
    :target: https://pypi.org/project/tmtoolkit/
    :alt: Downloads from PyPI

.. |travis| image:: https://travis-ci.org/WZBSocialScienceCenter/tmtoolkit.svg?branch=master
    :target: https://travis-ci.org/WZBSocialScienceCenter/tmtoolkit
    :alt: Travis CI Build Status

.. |coverage| image:: https://raw.githubusercontent.com/WZBSocialScienceCenter/tmtoolkit/master/coverage.svg?sanitize=true
    :target: https://github.com/WZBSocialScienceCenter/tmtoolkit/tree/master/tests
    :alt: Coverage status

.. |rtd| image:: https://readthedocs.org/projects/tmtoolkit/badge/?version=latest
    :target: https://tmtoolkit.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |zenodo| image:: https://zenodo.org/badge/109812180.svg
    :target: https://zenodo.org/badge/latestdoi/109812180
    :alt: Citable Zenodo DOI
