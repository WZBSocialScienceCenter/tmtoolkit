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


tmtoolkit.defaults
------------------

.. automodule:: tmtoolkit.defaults
    :members:

    .. data:: language

       The default language used in the functional preprocessing API (see :mod:`tmtoolkit.preprocess`).


tmtoolkit.preprocess
--------------------

TMPreproc class for parallel text preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tmtoolkit.preprocess.TMPreproc
    :members:

    .. automethod:: __init__
    .. automethod:: __del__
    .. automethod:: __deepcopy__


Functional Preprocessing API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.preprocess
    :members: clean_tokens, doc_frequencies, doc_lengths, expand_compound_token, expand_compounds, filter_documents,
              filter_documents_by_name, filter_for_pos, filter_tokens, glue_tokens, ids2tokens, kwic, kwic_table,
              lemmatize, load_lemmatizer_for_language, load_pos_tagger_for_language, make_index_window_around_matches,
              ngrams, pos_tag, pos_tag_convert_penn_to_wn, remove_chars, remove_chars, remove_common_tokens,
              remove_documents, remove_documents_by_name, remove_tokens, remove_tokens_by_doc_frequency,
              remove_uncommon_tokens, simplified_pos, sparse_dtm, stem, str_multisplit, to_lowercase,
              token_glue_subsequent, token_match, token_match_subsequent, tokenize, tokens2ids, transform, vocabulary,
              vocabulary_counts


tmtoolkit.topicmod
------------------

.. automodule:: tmtoolkit.topicmod
    :members:

Evaluation metrics for Topic Modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tmtoolkit.topicmod.evaluate
    :members:


tmtoolkit.utils
---------------

.. automodule:: tmtoolkit.utils
    :members:
