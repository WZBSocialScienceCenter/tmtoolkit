# -*- coding: utf-8 -*-
"""
An example for topic modeling evaluation with the [lda package](http://pythonhosted.org/lda/).
"""
import logging

import lda  # for the Reuters dataset

from tmtoolkit.utils import pickle_data
from tmtoolkit.lda_utils import tm_lda
from tmtoolkit.lda_utils.common import results_by_parameter, \
    print_ldamodel_topic_words, print_ldamodel_doc_topics, save_ldamodel_summary_to_excel
from tmtoolkit.lda_utils.visualize import plot_eval_results

import matplotlib.pyplot as plt
plt.style.use('ggplot')


logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True


if __name__ == '__main__':   # this is necessary for multiprocessing on Windows!
    # load the Reuters News dataset provided by lda
    print('loading data')
    doc_labels = lda.datasets.load_reuters_titles()
    vocab = lda.datasets.load_reuters_vocab()
    dtm = lda.datasets.load_reuters()
    print('%d documents with vocab size %d' % (len(doc_labels), len(vocab)))
    assert dtm.shape[0] == len(doc_labels)
    assert dtm.shape[1] == len(vocab)

    # evaluate topic models with different parameters
    const_params = dict(n_iter=1000, random_state=1, refresh=10, eta=0.1)    # beta is called eta in the 'lda' package
    ks = list(range(10, 140, 10)) + list(range(140, 300, 20)) + [300, 325, 350, 375, 400, 450, 500, 600, 800]
    varying_params = [dict(n_topics=k, alpha=1.0/k) for k in ks]

    # this will evaluate all models in parallel
    # still, this will take some time
    print('evaluating %d topic models' % len(varying_params))
    models = tm_lda.evaluate_topic_models(dtm, varying_params, const_params,
                                          return_models=True)  # retain the calculated models

    # save the results as pickle
    print('saving results')
    pickle_data(models, 'data/lda_evaluation_results.pickle')

    # plot the results
    print('plotting evaluation results')
    results_by_n_topics = results_by_parameter(models, 'n_topics')
    plot_eval_results(results_by_n_topics, xaxislabel='num. topics k',
                      title='Evaluation results for alpha=1/k, beta=0.1', figsize=(8, 6))
    plt.savefig('data/lda_evaluation_plot.png')
    plt.show()

    # # the peak seems to be around n_topics == 230
    # # print the distributions of this model
    # n_topics_best_model = 230
    # print('printing best model with n_topics=%d' % n_topics_best_model)
    # best_model = dict(results_by_n_topics)[n_topics_best_model]['model']
    # print_ldamodel_topic_words(best_model.topic_word_, vocab)
    # print_ldamodel_doc_topics(best_model.doc_topic_, doc_labels)
    #
    # # export it as Excel file
    # excel_file = 'data/lda_evaluation_summary.xlsx'
    # print('saving summary as Excel file to `%s`' % excel_file)
    # save_ldamodel_summary_to_excel(excel_file,
    #                                best_model.topic_word_, best_model.doc_topic_,
    #                                doc_labels, vocab, dtm=dtm)
