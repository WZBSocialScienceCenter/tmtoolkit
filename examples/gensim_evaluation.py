# -*- coding: utf-8 -*-
"""
An example for topic modeling evaluation with gensim
"""

from __future__ import division
import logging

import matplotlib.pyplot as plt
import gensim
import pandas as pd

from tmtoolkit.topicmod import tm_gensim
from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.utils import pickle_data
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results


logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True


print('loading data...')
bt18 = pd.read_pickle('data/bt18_sample_1000.pickle')
print('loaded %d documents' % len(bt18))
doc_labels = [u'%s_%s' % info for info in zip(bt18.sitzung, bt18.sequence)]

print('preprocessing data...')
bt18corp = Corpus(dict(zip(doc_labels, bt18.text)))
preproc = TMPreproc(bt18corp, language='german')
preproc.tokenize().stem().clean_tokens()

doc_labels = list(preproc.tokens.keys())
texts = list(preproc.tokens.values())

print('creating gensim corpus...')
gnsm_dict = gensim.corpora.Dictionary.from_documents(texts)
gnsm_corpus = [gnsm_dict.doc2bow(text) for text in texts]

# evaluate topic models with different parameters
const_params = dict(update_every=0, passes=20)
ks = list(range(10, 140, 10)) + list(range(140, 200, 20))
varying_params = [dict(num_topics=k, alpha=1.0 / k) for k in ks]

print('evaluating %d topic models' % len(varying_params))
eval_results = tm_gensim.evaluate_topic_models((gnsm_dict, gnsm_corpus), varying_params, const_params,
                                               coherence_gensim_texts=texts)

# save the results as pickle
print('saving results')
pickle_data(eval_results, 'data/gensim_evaluation_results.pickle')


# plot the results
print('plotting evaluation results')
plt.style.use('ggplot')
results_by_n_topics = results_by_parameter(eval_results, 'num_topics')
plot_eval_results(results_by_n_topics, xaxislabel='num. topics k',
                  title='Evaluation results', figsize=(8, 6))
plt.savefig('data/gensim_evaluation_plot.png')
plt.show()
