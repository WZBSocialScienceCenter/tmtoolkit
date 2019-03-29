import logging
from datetime import datetime
from multiprocessing import cpu_count

import nltk
from tmtoolkit.preprocess import TMPreproc

logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True


#%%

corpus = {f_id: nltk.corpus.gutenberg.raw(f_id) for f_id in nltk.corpus.gutenberg.fileids() if f_id != 'bible-kjv.txt'}

#%%

t_start = datetime.today()

preproc = TMPreproc(corpus, n_max_processes=cpu_count())

# preproc.tokenize()
# preproc.expand_compound_tokens()
# preproc.pos_tag()
# preproc.lemmatize()
# preproc.remove_special_chars_in_tokens()
# preproc.tokens_to_lowercase()
preproc.clean_tokens()
# preproc.remove_common_tokens(0.9)
# preproc.remove_uncommon_tokens(0.05)

vocab = preproc.vocabulary
tokens = preproc.tokens
tokens_tagged = preproc.tokens_with_pos_tags
dtm = preproc.get_dtm()

if isinstance(dtm, tuple):
    _, _, dtm = dtm

print('final DTM shape:')
print(dtm.shape)

t_delta = datetime.today() - t_start

print(t_delta.total_seconds(), 'sec')
