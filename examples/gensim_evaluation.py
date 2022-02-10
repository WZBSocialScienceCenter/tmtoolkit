"""
An example for topic modeling evaluation with gensim.

Please note that this is just an example for showing how to perform topic model evaluation with Gensim. The
preprocessing of the data is just done quickly and probably not the best way for the given data.

This examples requires that you have installed tmtoolkit with the recommended set of packages plus Gensim and have
installed a German language model for spaCy:

    pip install -U "tmtoolkit[recommended,gensim]"
    python -m tmtoolkit setup de

For more information, see the installation instructions: https://tmtoolkit.readthedocs.io/en/latest/install.html

"""


import matplotlib.pyplot as plt
import gensim
import pandas as pd

from tmtoolkit import corpus as c
from tmtoolkit.topicmod import tm_gensim
from tmtoolkit.utils import pickle_data, enable_logging
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results

#%%

enable_logging()

#%% loading data

print('loading data...')
bt18 = pd.read_pickle('data/bt18_sample_1000.pickle')
print('loaded %d documents' % len(bt18))
doc_labels = ['%s_%s' % info for info in zip(bt18.sitzung, bt18.sequence)]

#%%

print('loading and tokenizing documents')
# minimal pipeline
bt18corp = c.Corpus(dict(zip(doc_labels, bt18.text)), language='de', load_features=[], max_workers=1.0)
del bt18
c.print_summary(bt18corp)

print('preprocessing data...')

c.stem(bt18corp)
c.filter_clean_tokens(bt18corp)

c.print_summary(bt18corp)

#%%

print('creating gensim corpus...')

texts = list(c.doc_tokens(bt18corp).values())
gnsm_dict = gensim.corpora.Dictionary.from_documents(texts)
gnsm_corpus = [gnsm_dict.doc2bow(text) for text in texts]

del bt18corp

#%%

# evaluate topic models with different parameters
const_params = dict(update_every=0, passes=10)
ks = list(range(10, 140, 10)) + list(range(140, 200, 20))
varying_params = [dict(num_topics=k, alpha=1.0 / k) for k in ks]

print(f'evaluating {len(varying_params)} topic models')
eval_results = tm_gensim.evaluate_topic_models((gnsm_dict, gnsm_corpus), varying_params, const_params,
                                               coherence_gensim_texts=texts)   # necessary for coherence C_V metric

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
