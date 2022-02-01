.. _api:

API
===

tmtoolkit.bow
-------------

tmtoolkit.bow.bow_stats
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.bow.bow_stats
    :members:

tmtoolkit.bow.dtm
^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.bow.dtm
    :members:


tmtoolkit.corpus
----------------

Corpus class for text processing and mining
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tmtoolkit.corpus.Corpus
    :members:

    .. automethod:: __str__
    .. automethod:: __repr__
    .. automethod:: __len__
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__
    .. automethod:: __iter__
    .. automethod:: __contains__
    .. automethod:: __copy__
    .. automethod:: __deepcopy__

Corpus functions for accessing corpus data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.corpus
    :members:
    :imported-members:

Corpus functions for storing and loading data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.corpus
    :members: tmtoolkit.corpus.corpus_add_files

Corpus functions for managing document and token attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.corpus
    :members: tmtoolkit.corpus.set_document_attr

Corpus functions for transforming tokens
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.corpus
    :members: tmtoolkit.corpus.transform_tokens

Corpus functions for filtering documents and tokens
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.corpus
    :members:

Other corpus functions
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.corpus
    :members:

Functions to visualize corpus summary statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.corpus.visualize
    :members:

Document class for representing a tokenized document
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tmtoolkit.corpus.Document
    :members:

.. autofunction:: tmtoolkit.corpus.document_token_attr
.. autofunction:: tmtoolkit.corpus.document_from_attrs

Other functions and constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autodata:: tmtoolkit.corpus.DEFAULT_LANGUAGE_MODELS
.. autodata:: tmtoolkit.corpus.LANGUAGE_LABELS
.. autofunction:: tmtoolkit.corpus.simplified_pos
.. autofunction:: tmtoolkit.corpus.stem


tmtoolkit.tokenseq
------------------

.. automodule:: tmtoolkit.tokenseq
    :members:


tmtoolkit.topicmod
------------------

.. automodule:: tmtoolkit.topicmod
    :members:

Evaluation metrics for Topic Modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.topicmod.evaluate
    :members:


Printing, importing and exporting topic model results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.topicmod.model_io
    :members:


Statistics for topic models and BoW matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.topicmod.model_stats
    :members:


Parallel model fitting and evaluation with lda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.topicmod.tm_lda
   :members: AVAILABLE_METRICS, DEFAULT_METRICS, compute_models_parallel, evaluate_topic_models


Parallel model fitting and evaluation with scikit-learn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.topicmod.tm_sklearn
   :members: AVAILABLE_METRICS, DEFAULT_METRICS, compute_models_parallel, evaluate_topic_models


Parallel model fitting and evaluation with Gensim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.topicmod.tm_gensim
   :members: AVAILABLE_METRICS, DEFAULT_METRICS, compute_models_parallel, evaluate_topic_models


Visualize topic models and topic model evaluation results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wordclouds from topic models
""""""""""""""""""""""""""""

.. autodata:: tmtoolkit.topicmod.visualize.DEFAULT_WORDCLOUD_KWARGS
.. autofunction:: tmtoolkit.topicmod.visualize.generate_wordclouds_for_topic_words
.. autofunction:: tmtoolkit.topicmod.visualize.generate_wordclouds_for_document_topics
.. autofunction:: tmtoolkit.topicmod.visualize.generate_wordcloud_from_probabilities_and_words
.. autofunction:: tmtoolkit.topicmod.visualize.generate_wordcloud_from_weights
.. autofunction:: tmtoolkit.topicmod.visualize.write_wordclouds_to_folder
.. autofunction:: tmtoolkit.topicmod.visualize.generate_wordclouds_from_distribution

Plot heatmaps for topic models
""""""""""""""""""""""""""""""

.. autofunction:: tmtoolkit.topicmod.visualize.plot_doc_topic_heatmap
.. autofunction:: tmtoolkit.topicmod.visualize.plot_topic_word_heatmap
.. autofunction:: tmtoolkit.topicmod.visualize.plot_heatmap

Plot probability distribution rankings for topic models
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. autofunction:: tmtoolkit.topicmod.visualize.plot_topic_word_ranked_prob
.. autofunction:: tmtoolkit.topicmod.visualize.plot_doc_topic_ranked_prob
.. autofunction:: tmtoolkit.topicmod.visualize.plot_prob_distrib_ranked_prob

Plot topic model evaluation results
"""""""""""""""""""""""""""""""""""

.. autofunction:: tmtoolkit.topicmod.visualize.plot_eval_results

Other functions
"""""""""""""""

.. autofunction:: tmtoolkit.topicmod.visualize.parameters_for_ldavis


Base classes for parallel model fitting and evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.topicmod.parallel
    :members:


tmtoolkit.utils
---------------

.. automodule:: tmtoolkit.utils
    :members:
