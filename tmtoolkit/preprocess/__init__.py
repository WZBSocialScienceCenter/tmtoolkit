from spacy.tokens import Doc

Doc.set_extension('label', default='')

# from tmtoolkit.preprocess._common import (init_for_language,
#     tokenize, doc_labels, doc_lengths, vocabulary, vocabulary_counts, doc_frequencies, ngrams, load_stopwords,
#     sparse_dtm, kwic, kwic_table, glue_tokens, tokens2ids, ids2tokens, str_multisplit, str_shape, str_shapesplit,
#     expand_compound_token, remove_chars, token_match, token_match_subsequent, token_glue_subsequent,
#     make_index_window_around_matches, pos_tag_convert_penn_to_wn, simplified_pos, transform, to_lowercase,
#     lemmatize, pos_tag, expand_compounds, clean_tokens, compact_documents,
#     filter_tokens_by_mask, remove_tokens_by_mask, filter_tokens, remove_tokens, filter_documents, remove_documents,
#     filter_documents_by_name, remove_documents_by_name, filter_for_pos, remove_tokens_by_doc_frequency,
#     remove_common_tokens, remove_uncommon_tokens, filter_tokens_with_kwic
# )

from ._tmpreproc import TMPreproc, DEFAULT_LANGUAGE_MODELS
