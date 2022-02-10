"""
A minimal example to showcase a few features of tmtoolkit.

Markus Konrad <markus.konrad@wzb.eu>
Feb. 2022
"""

from tmtoolkit.corpus import Corpus, tokens_table, lemmatize, to_lowercase, dtm
from tmtoolkit.bow.bow_stats import tfidf, sorted_terms_table


# load built-in sample dataset and use 4 worker processes
corp = Corpus.from_builtin_corpus('en-News100', max_workers=4)

# investigate corpus as dataframe
toktbl = tokens_table(corp)
print(toktbl)

# apply some text normalization
lemmatize(corp)
to_lowercase(corp)

# build sparse document-token matrix (DTM)
# document labels identify rows, vocabulary tokens identify columns
mat, doc_labels, vocab = dtm(corp, return_doc_labels=True, return_vocab=True)

# apply tf-idf transformation to DTM
# operation is applied on sparse matrix and uses few memory
tfidf_mat = tfidf(mat)

# show top 5 tokens per document ranked by tf-idf
top_tokens = sorted_terms_table(tfidf_mat, vocab, doc_labels, top_n=5)
print(top_tokens)
