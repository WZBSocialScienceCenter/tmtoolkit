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

Corpus class for handling raw text corpora
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tmtoolkit.corpus.Corpus
    :members:

    .. automethod:: __init__
    .. automethod:: __deepcopy__
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__
    .. automethod:: __contains__

Utility functions in :mod:`~tmtoolkit.corpus` module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.corpus
    :members: linebreaks_win2unix, paragraphs_from_lines, path_recursive_split, read_text_file


tmtoolkit.preprocess
--------------------

TMPreproc class for parallel text preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tmtoolkit.preprocess.TMPreproc
    :members:

    .. automethod:: __init__
    .. automethod:: __del__
    .. automethod:: __copy__
    .. automethod:: __deepcopy__


Functional Preprocessing API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.preprocess
    :members: DEFAULT_LANGUAGE_MODELS, LANGUAGE_LABELS, load_stopwords, simplified_pos,
        init_for_language, tokenize, doc_labels, doc_tokens, doc_lengths, doc_frequencies, vocabulary, vocabulary_counts,
        ngrams, sparse_dtm, kwic, kwic_table, glue_tokens, expand_compounds, lemmatize, pos_tag, pos_tags, clean_tokens,
        compact_documents, filter_tokens, filter_tokens_by_mask, filter_tokens_with_kwic, filter_for_pos,
        filter_documents_by_name, filter_documents, remove_tokens, remove_tokens_by_mask, remove_documents,
        remove_documents_by_name, remove_tokens_by_doc_frequency, remove_common_tokens, remove_uncommon_tokens,
        tokendocs2spacydocs, spacydoc_from_tokens, transform, to_lowercase, remove_chars, tokens2ids, ids2tokens,
        token_match, token_match_subsequent, token_glue_subsequent, expand_compound_token,
        str_shape, str_shapesplit, str_multisplit,
        make_index_window_around_matches,
        pos_tag_convert_penn_to_wn, stem


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
