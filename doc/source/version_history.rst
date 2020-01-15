.. _changes:

Version history
===============

0.9.0 - 2019-12-20
------------------

* added usage and API documentation
* added support for Arun 2010 metric in `tm_gensim` (thx to @mcooper)
* added support for `datatable package <https://github.com/h2oai/datatable/>`_
* added functional API for text preprocessing
* added KWIC in text preprocessing
* added post-installation setup routine to download necessary data files
* added built-in corpora
* added `sorted_terms` and `sorted_terms_data_table` to `bow_stats`
* added `glue_tokens` function
* retain sparse matrices in several `bow_stats` functions such as tfidf
* corpus module: loading of CSV and ZIP files, added several other new methods
* faster `get_dtm` (now works in parallel)
* `filter_tokens` / `filter_documents` accept multiple patterns at once
* lots of (partly **breaking**) changes and speed improvements in `TMPreproc`
* fixed error with `ignore_case` being ignored in `token_match` for regex and glob
* integrate tox
* use Numpy extras for hypothesis tests
* compatibility with Python 3.6, 3.7 and 3.8


0.8.0 - 2019-02-05
------------------

* faster package and sub-module import
* remove support for Python 2.7 (now only Python 3.5 and higher is supported)
* use `germalemma package <https://pypi.org/project/germalemma/>`_
* use importlib instead of deprecated imp
* fix problem with not installing all required packages


0.7.3 - 2018-09-17 (last release to support Python 2.7)
-------------------------------------------------------

* new options in `corpus` module for converting Windows linebreaks to Unix linebreaks

0.7.2 - 2018-07-23
------------------

* new option for `exclude_topics`: `return_new_topic_mapping`
* fixed `issue #7 <https://github.com/WZBSocialScienceCenter/tmtoolkit/issues/7>`_ (results entry about model gets overwritten)

0.7.1 - 2018-06-18
------------------

* fix stupid missing import

0.7.0 - 2018-06-18
------------------

* added sub-package `bow` with functions for DTM creation and statistics
* fixed problems with evaluation and parallel calculation of gensim models (#5)
* added Gensim evaluation example

0.6.3 - 2018-06-01
------------------

* made `get_vocab_and_terms` more memory-efficient
* updated requirements (fixes #6)

0.6.2 - 2018-04-27
------------------

* added new function `exclude_topics` to `model_stats`

0.6.1 - 2018-04-27
------------------

* better figure title placement, grouped subplots and other improvements in `plot_eval_results`
* bugfix in `model_stats` due to missing unicode literals

0.6.0 - 2018-04-25
------------------

* **API restructured: (uninstall package first when upgrading!)**
  * sub-package `lda_utils` is now called `topicmod`
  * no more `common` module in `topicmod` -> divided into `evaluate` (including evaluation metrics from former `eval_metrics`), `model_io`, `model_stats`, and `parallel`
* added coherence metrics `PR #2 <https://github.com/WZBSocialScienceCenter/tmtoolkit/pull/2>`_
  * implemented modified coherence metric according to Mimno et al. 2011 as `metric_coherence_mimno_2011`
  * added wrapper function for coherence model provided by Gensim as `metric_coherence_gensim`
* added evaluation metric with probability of held-out documents in cross-validation (see `metric_held_out_documents_wallach09`)
* added new example for topic model coherence
* updated examples

0.5.0 - 2018-02-13
------------------

* add `doc_paths` field to `Corpus`
* change `plot_eval_results` to show individual metrics' results as subplots – **function signature changed!**

0.4.2 - 2018-02-06
------------------

* made greedy partitioning much more efficient (i.e. faster work distribution)
* added package information variables
* added this CHANGES document :)

0.4.1 - 2018-01-24
------------------

* fixed bug in `lda_utils.common.ldamodel_full_doc_topics`
* added `topic_labels` for doc-topic heatmap
* minor documentation fixes

0.4.0 - 2018-01-18
------------------

* improved parameter checks for `TMPreproc.filter_for_pos`
* improved tests for `TMPreproc.filter_for_pos`
* fixed broken test in Python 2.x
* added `generate_topic_labels_from_top_words`
* speed up in `top_n_from_distribution`
* added relevance score calculation (Sievert et al 2014)
* added functions to get most/least distinctive words
* added saliency calculation
* allow to define axis labels and plot title in `plot_eval_results`

