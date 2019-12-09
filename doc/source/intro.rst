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
  (`Chuang et al. 2012 <https://dl.acm.org/citation.cfm?id=2254572>`_), topic-word relevancy
  (`Sievert and Shirley 2014 <https://www.aclweb.org/anthology/W14-3110>`_)
* finding topics according to word patterns
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


About this documentation
------------------------

This documentation guides you in several chapters from installing tmtoolkit to its specific use cases and shows several
examples with the built-in and other datasets. All "hands on" chapters from `Getting started <getting_started.ipynb>`_
to `Topic modeling <topic_modeling.ipynb>`_ are generated from `Jupyter Notebooks <https://jupyter.org/>`_. If you want
to follow along using these notebooks, you can
`download them from the GitHub repository <https://github.com/WZBSocialScienceCenter/tmtoolkit/tree/master/doc/source>`_.

There are also several other examples as plain Python scripts or notebooks available in the
`examples folder <https://github.com/WZBSocialScienceCenter/tmtoolkit/tree/master/examples>`_ of the GitHub repository.
