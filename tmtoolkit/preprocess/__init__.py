from spacy.tokens import Doc

Doc.set_extension('label', default='', force=True)

from ._common import (
    DEFAULT_LANGUAGE_MODELS, LANGUAGE_LABELS, load_stopwords, simplified_pos
)

from ._docfuncs import (
    init_for_language, tokenize, doc_labels, doc_tokens, doc_lengths, doc_frequencies, vocabulary, vocabulary_counts,
    ngrams, sparse_dtm, kwic, kwic_table, glue_tokens, expand_compounds, lemmatize, pos_tag, pos_tags, clean_tokens,
    compact_documents, filter_tokens, filter_tokens_by_mask, filter_tokens_with_kwic, filter_for_pos,
    filter_documents_by_name, filter_documents, remove_tokens, remove_tokens_by_mask, remove_documents,
    remove_documents_by_name, remove_tokens_by_doc_frequency, remove_common_tokens, remove_uncommon_tokens,
    tokendocs2spacydocs, spacydoc_from_tokens, transform, to_lowercase, remove_chars, tokens2ids, ids2tokens
)

from ._tokenfuncs import (
    token_match, token_match_subsequent, token_glue_subsequent, expand_compound_token,
    str_shape, str_shapesplit, str_multisplit,
    make_index_window_around_matches
)

try:
    from ._nltk_extras import pos_tag_convert_penn_to_wn, stem
except ImportError: pass   # when NLTK is not installed

from ._tmpreproc import TMPreproc
