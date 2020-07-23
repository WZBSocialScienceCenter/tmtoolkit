"""
Benchmarking script that loads and processes English language test corpus with TMPreproc in parallel.

This is only to compare parallel spaCy (this script) over single core spaCy (see `benchmark_comparison_singlecore.py`),
hence we only check functions that involve calls to spaCy.

To benchmark whole script with `time` from command line run:

    PYTHONPATH=.. /usr/bin/time -v python benchmark_comparison_parallel.py
"""

import logging
from multiprocessing import cpu_count

from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc

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

preproc = TMPreproc(corpus, language='en', n_max_processes=4)
add_timing('load and tokenize')

preproc.pos_tag()
add_timing('pos_tag')

preproc.lemmatize()
add_timing('lemmatize')

# vocab = preproc.vocabulary
# add_timing('get vocab')

# dtm = preproc.get_dtm()
# add_timing('get dtm')

# if isinstance(dtm, tuple):
#     _, _, dtm = dtm
#
# print('final DTM shape:')
# print(dtm.shape)

print_timings()
