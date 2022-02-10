"""
Module for processing text as token sequences in labelled documents. A set of documents is represented as *corpus*
using the :class:`Corpus` class. This sub-package also provides functions that work with a :class:`Corpus` object.

Text parsing and processing relies on the `SpaCy library <https://spacy.io/>`_ which must be installed when using this
sub-package.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

from importlib.util import find_spec

for pkg in ('spacy', 'bidict', 'loky'):
    if find_spec(pkg) is None:
        raise RuntimeError(f'the required package "{pkg}" for text processing is not installed; did you install '
                           f'tmtoolkit with "recommended" or "textproc" option? see '
                           f'https://tmtoolkit.readthedocs.io/en/latest/install.html for further information')

from ..tokenseq import strip_tags, numbertoken_to_magnitude, simplify_unicode_chars

from ._common import DEFAULT_LANGUAGE_MODELS, LANGUAGE_LABELS, simplified_pos
from ._document import Document, document_token_attr, document_from_attrs
from ._corpus import Corpus

from ._corpusfuncs import (
    doc_tokens, set_token_attr, set_document_attr, vocabulary, dtm, doc_texts, doc_labels, doc_lengths,
    corpus_num_tokens, vocabulary_size, tokens_table, print_summary, vocabulary_counts,
    doc_frequencies, doc_vectors, token_vectors, ngrams, to_lowercase, to_uppercase, remove_chars,
    serialize_corpus, deserialize_corpus, save_corpus_to_picklefile, load_corpus_from_picklefile,
    load_corpus_from_tokens, load_corpus_from_tokens_table, spacydocs,
    lemmatize, remove_punctuation, normalize_unicode, simplify_unicode, doc_token_lengths, filter_clean_tokens,
    corpus_ngramify, filter_tokens_by_mask, remove_tokens_by_mask, filter_tokens, remove_tokens,
    filter_documents, remove_documents, filter_documents_by_mask, remove_documents_by_mask,
    filter_documents_by_docattr, remove_documents_by_docattr, kwic, kwic_table, transform_tokens,
    corpus_summary, corpus_num_chars, filter_tokens_with_kwic, filter_documents_by_label,
    remove_documents_by_label, filter_for_pos, filter_tokens_by_doc_frequency, remove_common_tokens,
    remove_uncommon_tokens, filter_documents_by_length, remove_documents_by_length,
    join_collocations_by_patterns, join_collocations_by_statistic, corpus_tokens_flattened, corpus_collocations,
    remove_token_attr, remove_document_attr, builtin_corpora_info, corpus_add_files, corpus_add_folder,
    corpus_add_tabular, corpus_add_zip, corpus_sample, corpus_split_by_token, doc_num_sents, doc_sent_lengths,
    numbers_to_magnitudes, corpus_split_by_paragraph, doc_labels_sample, corpus_retokenize, corpus_unique_chars,
    corpus_join_documents, find_documents
)

if find_spec('nltk') is not None:  # when NLTK is installed
    from ._nltk_extras import stem

from . import visualize
