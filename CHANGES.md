# Changes

## 0.6.0 - 2018-04-25

* **API restructured:**
  * sub-package `lda_utils` is now called `topicmod`
  * no more `common` module in `topicmod` -> divided into `evaluate` (including evaluation metrics from former `eval_metrics`), `model_io`, `model_stats`, and `parallel`
* added coherence metrics ([PR #2](https://github.com/WZBSocialScienceCenter/tmtoolkit/pull/2))
  * implemented modified coherence metric according to Mimno et al. 2011 as `metric_coherence_mimno_2011`
  * added wrapper function for coherence model provided by Gensim as `metric_coherence_gensim`
* added evaluation metric with probability of held-out documents in cross-validation (see `metric_held_out_documents_wallach09`)
* added new example for topic model coherence
* updated examples

## 0.5.0 - 2018-02-13

* add `doc_paths` field to `Corpus`
* change `plot_eval_results` to show individual metrics' results as subplots – **function signature changed!**

## 0.4.2 - 2018-02-06

* made greedy partitioning much more efficient (i.e. faster work distribution)
* added package information variables
* added this CHANGES document :)

## 0.4.1 - 2018-01-24

* fixed bug in `lda_utils.common.ldamodel_full_doc_topics`
* added `topic_labels` for doc-topic heatmap
* minor documentation fixes

## 0.4.0 - 2018-01-18

* improved parameter checks for `TMPreproc.filter_for_pos`
* improved tests for `TMPreproc.filter_for_pos`
* fixed broken test in Python 2.x
* added `generate_topic_labels_from_top_words`
* speed up in `top_n_from_distribution`
* added relevance score calculation (Sievert et al 2014)
* added functions to get most/least distinctive words
* added saliency calculation
* allow to define axis labels and plot title in `plot_eval_results`
