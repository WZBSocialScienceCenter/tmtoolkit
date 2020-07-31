"""
Benchmarking script that loads and processes English language test corpus.

To benchmark whole script with `time` from command line run:

    PYTHONPATH=.. /usr/bin/time -v python benchmark_preproc.py
"""

import random
import logging
from tempfile import mkstemp
from multiprocessing import cpu_count

from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc

from examples._benchmarktools import add_timing, print_timings

logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True

random.seed(20200320)

#%%

corpus = Corpus.from_builtin_corpus('en-NewsArticles').sample(1000)

print('%d documents' % len(corpus))


#%%

add_timing('start')

preproc = TMPreproc(corpus, language='en', n_max_processes=cpu_count())
add_timing('load and tokenize')

preproc.expand_compound_tokens()
add_timing('expand_compound_tokens')

preproc.pos_tag()
add_timing('pos_tag')

preproc.lemmatize()
add_timing('lemmatize')

preproc_copy = preproc.copy()
preproc_copy.shutdown_workers()
del preproc_copy
add_timing('copy')

_, statepickle = mkstemp('.pickle')
preproc.save_state(statepickle)
add_timing('save_state')

preproc_copy = TMPreproc.from_state(statepickle)
preproc_copy.shutdown_workers()
del preproc_copy
add_timing('from_state')

preproc_copy = TMPreproc.from_tokens(preproc.tokens_with_metadata, language='en')
preproc_copy.shutdown_workers()
del preproc_copy
add_timing('from_tokens')

preproc_copy = TMPreproc.from_tokens_datatable(preproc.tokens_datatable, language='en')
preproc_copy.shutdown_workers()
del preproc_copy
add_timing('from_tokens_datatable')

preproc.remove_special_chars_in_tokens()
add_timing('remove_special_chars_in_tokens')

preproc.tokens_to_lowercase()
add_timing('tokens_to_lowercase')

preproc.clean_tokens()
add_timing('clean_tokens')

preproc.remove_common_tokens(0.9)
preproc.remove_uncommon_tokens(0.05)
add_timing('remove_common_tokens / remove_uncommon_tokens')

vocab = preproc.vocabulary
add_timing('get vocab')

tokens = preproc.tokens
add_timing('get tokens')

tokens_tagged = preproc.get_tokens(with_metadata=True, as_datatables=False)
add_timing('get tagged tokens')

dtm = preproc.get_dtm()
add_timing('get dtm')


if isinstance(dtm, tuple):
    _, _, dtm = dtm

print('final DTM shape:')
print(dtm.shape)

print_timings()
