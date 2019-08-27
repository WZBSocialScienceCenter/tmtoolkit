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
    :members: tokenize, doc_lengths, vocabulary, vocabulary_counts, doc_frequencies, ngrams,
              sparse_dtm, kwic, kwic_table, glue_tokens, tokens2ids, ids2tokens, str_multisplit, expand_compound_token,
              remove_chars, remove_chars, token_match, token_match_subsequent, token_glue_subsequent,
              make_index_window_around_matches, pos_tag_convert_penn_to_wn, simplified_pos, transform, to_lowercase, stem,
              lemmatize, load_lemmatizer_for_language, pos_tag, load_pos_tagger_for_language, expand_compounds, clean_tokens,
              filter_tokens, remove_tokens, filter_documents, remove_documents,
              filter_documents_by_name, remove_documents_by_name, filter_for_pos, remove_tokens_by_doc_frequency,
              remove_common_tokens, remove_uncommon_tokens


