.. _api:

API
===

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
