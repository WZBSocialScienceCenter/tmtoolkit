"""
Benchmarking script that loads and processes English language documents from `nltk.corpus.gutenberg`.

To benchmark whole script with `time` from command line run:

    PYTHONPATH=.. /usr/bin/time -v python benchmark_preproc.py
"""

import sys
import logging
from datetime import datetime
from multiprocessing import cpu_count

import nltk
from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc

logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True

#%%

use_paragraphs = len(sys.argv) > 1 and sys.argv[1] == 'paragraphs'


#%%

corpus = Corpus({f_id: nltk.corpus.gutenberg.raw(f_id) for f_id in nltk.corpus.gutenberg.fileids()
                 if f_id != 'bible-kjv.txt'})

if use_paragraphs:
    print('using paragraphs as documents')
    corpus.split_by_paragraphs()

print('%d documents' % len(corpus))

#%%

timings = []
timing_labels = []

def add_timing(label):
    timings.append(datetime.today())
    timing_labels.append(label)


#%%

add_timing('start')

preproc = TMPreproc(corpus, n_max_processes=cpu_count())
add_timing('load')

preproc.tokenize()
add_timing('tokenize')

preproc.expand_compound_tokens()
add_timing('expand_compound_tokens')

preproc.pos_tag()
add_timing('pos_tag')

preproc.lemmatize()
add_timing('lemmatize')

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


print('timings:')
t_sum = 0
prev_t = None
for i, (t, label) in enumerate(zip(timings, timing_labels)):
    if i > 0:
        t_delta = (t - prev_t).total_seconds()
        print('%s: %.2f sec' % (label, t_delta))
        t_sum += t_delta

    prev_t = t

print('total: %.2f sec' % t_sum)
