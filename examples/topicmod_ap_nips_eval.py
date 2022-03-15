"""
Topic model evaluation for AP and NIPS datasets (http://archive.ics.uci.edu/ml/datasets/Bag+of+Words).

This examples requires that you have installed tmtoolkit with the "lda" package.

    pip install -U "tmtoolkit[lda]"

For more information, see the installation instructions: https://tmtoolkit.readthedocs.io/en/latest/install.html

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tmtoolkit.utils import unpickle_file, enable_logging
from tmtoolkit.topicmod.tm_lda import evaluate_topic_models, AVAILABLE_METRICS
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results


#%%

if len(sys.argv) != 5:
    print('req. args: dataset, number of workers, eta, alpha numerator')
    exit(1)

dataset = sys.argv[1]
n_workers = int(sys.argv[2])
eta = float(sys.argv[3])
alpha_numerator = float(sys.argv[4])

print(f'dataset: {dataset}, workers: {n_workers}, beta: {eta}, alpha numerator: {alpha_numerator}')

dataset_short = os.path.basename(dataset)[:-7]

#%%

enable_logging()

#%%

print('loading data...')

doc_labels, vocab, dtm = unpickle_file(dataset)
doc_labels = np.asarray(doc_labels)
vocab = np.asarray(vocab)

#%%

print('running evaluations...')

const_params = {
    'n_iter': 1500,
    'eta': eta,
    'random_state': 20220105  # to make results reproducible
}

var_params = [{'n_topics': k, 'alpha': alpha_numerator/k}
              for k in list(range(20, 201, 2))]

metrics = ['arun_2010', 'cao_juan_2009', 'coherence_mimno_2011']

if 'griffiths_2004' in AVAILABLE_METRICS:
    metrics.append('griffiths_2004')

eval_results = evaluate_topic_models(dtm,
                                     varying_parameters=var_params,
                                     constant_parameters=const_params,
                                     return_models=False,
                                     metric=metrics,
                                     n_max_processes=n_workers)

#%%

print('plotting evaluations...')

eval_by_topics = results_by_parameter(eval_results, 'n_topics')
plot_eval_results(eval_by_topics,
                  title=f'Evaluation results for {dataset_short}\nalpha={alpha_numerator}/K, beta={eta:.4}')

#plt.show()
plt.savefig(f'data/topicmod_evaluate_{dataset_short}_{eta:.4}.png')

print('done.')
