from spacy.tokens import Doc

# Meta data on document level is stored as Doc extension.
# Custom meta data on token level is however *not* stored as Token extension, since this approach proved to be very
# slow. It is instead stored in the `user_data` dict of each Doc instance.
Doc.set_extension('label', default='', force=True)
Doc.set_extension('mask', default=True, force=True)

from ._corpus import Corpus

from ._corpusfuncs import (
    doc_tokens, add_metadata_per_token, vocabulary, dtm, doc_texts, doc_labels, doc_lengths,
    n_tokens, vocabulary_size, tokens_datatable, tokens_dataframe, print_summary, vocabulary_counts,
    doc_frequencies, doc_vectors, token_vectors, ngrams, to_lowercase, to_uppercase,
    serialize_corpus, deserialize_corpus, save_corpus_to_picklefile, load_corpus_from_picklefile,
    load_corpus_from_tokens, load_corpus_from_tokens_datatable,
    lemmatize, remove_punctuation, normalize_unicode, simplify_unicode, doc_token_lengths, filter_clean_tokens,
    compact, ngramify, reset_filter, filter_tokens_by_mask, filter_tokens, filter_documents,
    set_document_attr, filter_documents_by_mask
)

from ._tokenfuncs import token_match, ngrams_from_tokenlist, spacydoc_from_tokens, spacydoc_from_tokens_with_metadata
