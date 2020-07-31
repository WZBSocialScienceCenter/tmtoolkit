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

Requirements and installation
==============================

For requirements and installation procedures, please have a look at the
`installation section in the documentation <https://tmtoolkit.readthedocs.io/en/latest/install.html>`_.

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
