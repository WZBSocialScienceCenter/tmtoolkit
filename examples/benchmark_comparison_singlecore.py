"""
Benchmarking script that loads and processes English language test corpus with spaCy.

This is only to compare parallel spaCy (`benchmark_comparison_parallel.py`) over single core spaCy (see this script),
hence we only check functions that involve calls to spaCy.

To benchmark whole script with `time` from command line run:

    PYTHONPATH=.. /usr/bin/time -v python benchmark_comparison_singlecore.py
"""

import logging

from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import init_for_language, tokenize, pos_tag, lemmatize

from examples._benchmarktools import add_timing, print_timings

logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True


#%%

corpus = Corpus.from_builtin_corpus('en-NewsArticles')

print('%d documents' % len(corpus))

#%%

add_timing('start')

init_for_language('en')
docs = tokenize(list(corpus.values()))

add_timing('load and tokenize')

pos_tag(docs)
add_timing('pos_tag')

cleaned_tokens = lemmatize(docs)
add_timing('lemmatize')

print_timings()
