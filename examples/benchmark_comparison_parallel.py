"""
Benchmarking script that loads and processes English language test corpus with TMPreproc in parallel.

This is only to compare parallel spaCy (this script) over single core spaCy (see `benchmark_comparison_singlecore.py`),
hence we only check functions that involve calls to spaCy.

To benchmark whole script with `time` from command line run:

    PYTHONPATH=.. /usr/bin/time -v python benchmark_comparison_parallel.py
"""

import logging

from tmtoolkit.corpus import Corpus, init_corpus_language, tokenize, doc_tokens, vocabulary, dtm

from examples._benchmarktools import add_timing, print_timings

logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True


#%%

docs = Corpus.from_builtin_corpus('en-NewsArticles')
docs.n_max_workers = 4

print('%d documents' % len(docs))

#%%

add_timing('start')

init_corpus_language(docs, 'en')
docs = tokenize(docs)

add_timing('load and tokenize')

toks = doc_tokens(docs)
add_timing('doc_tokens')

vocab = vocabulary(docs)
add_timing('vocabulary')

dtm_ = dtm(docs)
add_timing('sparse_dtm')

# preproc.pos_tag()
# add_timing('pos_tag')
#
# preproc.lemmatize()
# add_timing('lemmatize')

print_timings()
