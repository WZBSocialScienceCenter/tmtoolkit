"""
An example for topic modeling with LDA with focus on the new plotting functions in `tmtoolkit.corpus.visualize` and
in `tmtoolkit.topicmod.visualize`.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import os.path

import matplotlib.pyplot as plt

from tmtoolkit.utils import enable_logging, pickle_data, unpickle_file
from tmtoolkit.corpus import Corpus, lemmatize, to_lowercase, remove_punctuation, remove_common_tokens, \
    remove_uncommon_tokens, filter_clean_tokens, print_summary, remove_documents_by_length, dtm, \
    corpus_retokenize, save_corpus_to_picklefile, load_corpus_from_picklefile
from tmtoolkit.corpus.visualize import plot_doc_lengths_hist, plot_doc_frequencies_hist, plot_vocab_counts_hist, \
    plot_ranked_vocab_counts, plot_num_sents_hist, plot_sent_lengths_hist, plot_num_sents_vs_sent_length, \
    plot_token_lengths_hist
from tmtoolkit.topicmod.tm_lda import evaluate_topic_models    # we're using lda for topic modeling
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.model_io import print_ldamodel_topic_words
from tmtoolkit.topicmod.visualize import plot_eval_results, plot_topic_word_ranked_prob, plot_doc_topic_ranked_prob

#%%

enable_logging()

#%% loading the sample corpus (English news articles)

corp_picklefile = 'data/topicmod_lda_corpus.pickle'

if os.path.exists(corp_picklefile):
    docs = load_corpus_from_picklefile(corp_picklefile)
else:
    docs = Corpus.from_builtin_corpus('en-NewsArticles', max_workers=1.0)
    save_corpus_to_picklefile(docs, corp_picklefile)

print_summary(docs)


#%% plot some corpus summary statistics

# you can copy those and also do the plotting also after corpus transformations in the next cell
# this shows you nicely how the transformations change the distribution of words in the corpus

fig, ax = plt.subplots()
plot_doc_lengths_hist(fig, ax, docs)
plt.show()

fig, ax = plt.subplots()
plot_vocab_counts_hist(fig, ax, docs)
plt.show()

fig, ax = plt.subplots()
plot_ranked_vocab_counts(fig, ax, docs, zipf=True)
plt.show()

fig, ax = plt.subplots()
plot_doc_frequencies_hist(fig, ax, docs)
plt.show()

fig, ax = plt.subplots()
plot_num_sents_hist(fig, ax, docs)
plt.show()

fig, ax = plt.subplots()
plot_sent_lengths_hist(fig, ax, docs)
plt.show()

fig, ax = plt.subplots()
plot_num_sents_vs_sent_length(fig, ax, docs)
plt.show()

fig, ax = plt.subplots()
plot_token_lengths_hist(fig, ax, docs)
plt.show()

#%% apply preprocessing pipeline

corp_preproc_picklefile = 'data/topicmod_lda_corpus_preprocessed.pickle'

if os.path.exists(corp_preproc_picklefile):
    docs = load_corpus_from_picklefile(corp_preproc_picklefile)
else:
    remove_punctuation(docs)
    corpus_retokenize(docs)
    lemmatize(docs)
    to_lowercase(docs)
    filter_clean_tokens(docs, remove_numbers=True)
    remove_common_tokens(docs, df_threshold=0.90)
    remove_uncommon_tokens(docs, df_threshold=0.05)
    remove_documents_by_length(docs, '<', 30)

    save_corpus_to_picklefile(docs, corp_preproc_picklefile)

print_summary(docs)

#%% generating the document-term matrix

dtm_picklefile = 'data/topicmod_lda_dtm.pickle'

if os.path.exists(dtm_picklefile):
    bow_mat, doc_labels, vocab = unpickle_file(dtm_picklefile)
else:
    bow_mat, doc_labels, vocab = dtm(docs, return_doc_labels=True, return_vocab=True)
    pickle_data((bow_mat, doc_labels, vocab), dtm_picklefile)



#%% running the evaluation

eval_res_picklefile = 'data/topicmod_lda_eval_res.pickle'

if os.path.exists(dtm_picklefile):
    eval_results = unpickle_file(eval_res_picklefile)
else:
    const_params = {
        'n_iter': 1500,
        'eta': 0.3,
        'random_state': 20220105  # to make results reproducible
    }

    var_params = [{'n_topics': k, 'alpha': 10.0/k}
                  for k in list(range(20, 101, 20)) + [125, 150, 175, 200, 250, 300]]

    metrics = ['cao_juan_2009', 'arun_2010', 'coherence_mimno_2011', 'griffiths_2004']

    eval_results = evaluate_topic_models(bow_mat,
                                         varying_parameters=var_params,
                                         constant_parameters=const_params,
                                         return_models=True,
                                         metric=metrics)

    pickle_data(eval_results, eval_res_picklefile)

#%% plotting evaluation results

eval_by_topics = results_by_parameter(eval_results, 'n_topics')
plot_eval_results(eval_by_topics)

plt.show()

#%% selecting the model and printing the topics' most likely words

selected_model = dict(eval_by_topics)[200]['model']

print_ldamodel_topic_words(selected_model.topic_word_, vocab=vocab)

#%% investigating, how many "top words" sufficiently describe a topic

fig, ax = plt.subplots()
plot_topic_word_ranked_prob(fig, ax, selected_model.topic_word_, n=40, log_scale=False,
                            highlight=[4, 12, 32], alpha=0.025)

plt.show()

# -> about 5 to 10 words aggregate most of the probability per topic

#%% investigating, how many "top topics" sufficiently describe a document

fig, ax = plt.subplots()
plot_doc_topic_ranked_prob(fig, ax, selected_model.doc_topic_, n=40, log_scale=False, highlight=list(range(4)),
                           alpha=0.003)

plt.show()

# -> about 10 to 15 topics aggregate most of the probability per document
