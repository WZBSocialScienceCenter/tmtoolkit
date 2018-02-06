# Changes

## 0.4.2

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
