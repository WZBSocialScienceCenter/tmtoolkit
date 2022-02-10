"""
Benchmarking script that loads and processes English language test corpus with Corpus in parallel.

This examples requires that you have installed tmtoolkit with the recommended set of packages and have installed an
English language model for spaCy:

    pip install -U "tmtoolkit[recommended]"
    python -m tmtoolkit setup en

For more information, see the installation instructions: https://tmtoolkit.readthedocs.io/en/latest/install.html

To benchmark whole script with `time` from command line run:

    PYTHONPATH=.. /usr/bin/time -v python benchmark_en_newsarticles.py [NUMBER OF WORKERS]
"""

import sys
import logging

from tmtoolkit.corpus import Corpus, doc_tokens, vocabulary, dtm, lemmatize, to_lowercase, filter_clean_tokens

from examples._benchmarktools import add_timing, print_timings

logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True

if len(sys.argv) > 1:
    max_workers = int(sys.argv[1])
else:
    max_workers = 1

print(f'max workers: {max_workers}')

#%%

add_timing('start')

docs = Corpus.from_builtin_corpus('en-NewsArticles', language='en', max_workers=max_workers)
print(str(docs))

#%%

add_timing('load and tokenize')

toks = doc_tokens(docs)
add_timing('doc_tokens')

toks_w_attrs = doc_tokens(docs, with_attr=True)
add_timing('doc_tokens with attributes')

vocab = vocabulary(docs)
add_timing('vocabulary')

lemmatize(docs)
add_timing('lemmatize')

to_lowercase(docs)
add_timing('to_lowercase')

filter_clean_tokens(docs)
add_timing('filter_clean_tokens')

dtm_ = dtm(docs)
add_timing('sparse_dtm')

print_timings()
