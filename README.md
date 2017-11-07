# tmtoolkit -- Text Mining and Topic Modeling Toolkit for Python

Markus Konrad <markus.konrad@wzb.eu>, Nov. 2017

`tmtoolkit` is a set of tools for text mining and topic modeling with Python. It contains functions for text preprocessing like lemmatization, stemming or [POS tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging) especially for English and German texts. Preprocessing is done in parallel by using all available processors on your machine. The topic modeling features include topic model evaluation metrics, allowing to calculate models with different parameters in parallel and comparing them (e.g. in order to find the best number of topics for a given set of documents). Topic models can be generated in parallel for different copora and/or parameter sets using the LDA implementations either from [lda](http://pythonhosted.org/lda/), [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) or [gensim](https://radimrehurek.com/gensim/).

## Features

### Text preprocessing

Text preprocessing is built on top of [NLTK](http://www.nltk.org/) and (when run using Python 2.7) [CLiPS pattern](https://www.clips.uantwerpen.be/pattern). Common features include:

* tokenization
* POS tagging (optimized for German and English)
* lemmatization (optimized for German and English)
* stemming
* cleaning tokens
* filtering tokens
* generating n-grams
* generating document-term-matrices

Preprocessing is done in parallel by using all available processors on your machine, greatly improving processing speed as compared to sequential processing on a single processor.

### Topic modeling

Topic models can be generated in parallel for different copora and/or parameter sets using the LDA implementations either from [lda](http://pythonhosted.org/lda/), [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) or [gensim](https://radimrehurek.com/gensim/). They can be evaluated and compared (for example in order to find the optimal number of topics) using several implemented metrics:

* Pair-wise cosine distance method ([Cao Juan et al. 2009](http://doi.org/10.1016/j.neucom.2008.06.011))
* KL divergence method ([Arun et al. 2010](http://doi.org/10.1007/978-3-642-13657-3_43))
* Harmonic mean method ([Griffiths, Steyvers 2004](http://doi.org/10.1073/pnas.0307752101)) -- only when using the *lda* package
* the loglikelihood or perplexity methods natively implemented in lda, sklearn or gensim 

Further features include:

* plot evaluation results
* export estimated document-topic and topic-word distributions to Excel
* integrate [PyLDAVis](https://pyldavis.readthedocs.io/en/latest/) to visualize results

## Installation

The package is available on [PyPI](https://pypi.python.org/pypi/tmtoolkit/) and can be installed via Python package manager *pip*:

```
pip install tmtoolkit
```

## Requirements

`tmtoolkit` works with Python 2.7 and Python 3.5 or above. When using lemmatization for German texts, *Pyphen* and *pattern* should be installed, the latter being only available for Python 2.7.

Requirements are automatically installed via *pip*. Additional packages can also be installed via *pip* for certain use cases (see optional packages).

### Required packages

* six
* NumPy
* SciPy
* NLTK
* Pandas
* Pyphen

### Optional packages

For improved lemmatization of German texts, *Pattern* should be installed. Please note that *Pattern* is only available on Python 2.7.

For Excel export, *openpyxl* should be installed.

For topic modeling, one of the LDA implementations *lda*, *scikit-learn* or *gensim* should be installed.

For LDA evaluation metric `griffiths_2004` it is necessary to install [gmpy2](https://github.com/aleaxit/gmpy) for multiple-precision arithmetic. This in turn requires installing some C header libraries for GMP, MPFR and MPC. On Debian/Ubuntu systems this is done with:  

```
sudo apt install libgmp-dev libmpfr-dev libmpc-dev
```

After that, gmpy2 can be installed via *pip*.

For plotting evaluation results, `matplotlib` is necessary.

## Documentation

Documentation for many methods is still missing at the moment and will be added in the future. For the moment, you should have a look at the examples below and in the `examples` directory.

## Examples

Some examples that can be run directly in an IPython session:

### Preprocessing

We will process as small, self-defined toy corpus with German text. It will be tokenized, cleaned and transformed into a document-term-matrix. 

```python
from tmtoolkit.preprocess import TMPreproc

# a small toy corpus in German, here directly defined as a dict
# to load "real" (text) files use the methods in tmtoolkit.corpus
corpus = {
    u'doc1': u'Ein einfaches Beispiel in einfachem Deutsch.',
    u'doc2': u'Es enthält nur drei sehr einfache Dokumente.',
    u'doc3': u'Die Dokumente sind sehr kurz.',
}

# initialize
preproc = TMPreproc(corpus, language='german')

# run the preprocessing pipeline: tokenize, POS tag, lemmatize, transform to
# lowercase and then clean the tokens (i.e. remove stopwords)
preproc.tokenize().pos_tag().lemmatize().tokens_to_lowercase().clean_tokens()

print(preproc.tokens)
# this will output: 
#  {u'doc1': (u'einfach', u'beispiel', u'einfach', u'deutsch'),
#   u'doc2': (u'enthalten', u'drei', u'einfach', u'dokument'),
#   u'doc3': (u'dokument', u'kurz')}

print(preproc.tokens_with_pos_tags)
# this will output: 
# {u'doc1': [(u'einfach', u'ADJA'),
#   (u'beispiel', u'NN'),
#   (u'einfach', u'ADJA'),
#   (u'deutsch', u'NN')],
# u'doc2': [(u'enthalten', u'VVFIN'),
#   (u'drei', u'CARD'),
#   (u'einfach', u'ADJA'),
#   (u'dokument', u'NN')],
#  u'doc3': [(u'dokument', u'NN'), (u'kurz', u'ADJD')]}

# generate sparse DTM and print it as a data table
doc_labels, vocab, dtm = preproc.get_dtm()

import pandas as pd
print(pd.DataFrame(dtm.todense(), columns=vocab, index=doc_labels))
```

### Topic modeling

We will use the [lda package](http://pythonhosted.org/lda/) for topic modeling. Several models for different numbers of topics and alpha values are generated and compared. The best is chosen and the results are printed.

```python
from tmtoolkit.lda_utils import tm_lda
import lda  # for the Reuters dataset

import matplotlib.pyplot as plt
plt.style.use('ggplot')

doc_labels = lda.datasets.load_reuters_titles()
vocab = lda.datasets.load_reuters_vocab()
dtm = lda.datasets.load_reuters()

# evaluate topic models with different parameters
const_params = dict(n_iter=100, random_state=1)  # low number of iter. just for showing how it works here
varying_params = [dict(n_topics=k, alpha=1.0/k) for k in range(10, 251, 10)]

# this will evaluate 25 models (with n_topics = 10, 20, .. 250) in parallel
models = tm_lda.evaluate_topic_models(dtm, varying_params, const_params,
                                      return_models=True)

# plot the results
# note that since we used a low number of iterations, the plot looks quite "unstable"
# for the given metrics.
from tmtoolkit.lda_utils.common import results_by_parameter, plot_eval_results

results_by_n_topics = results_by_parameter(models, 'n_topics')
plot_eval_results(plt, results_by_n_topics)
plt.show()

# the peak seems to be around n_topics == 140
from tmtoolkit.lda_utils.common import print_ldamodel_topic_words, print_ldamodel_doc_topics

best_model = dict(results_by_n_topics)[140]['model']
print_ldamodel_topic_words(best_model.topic_word_, vocab)
print_ldamodel_doc_topics(best_model.doc_topic_, doc_labels)
```

More examples can be found in the `examples` directory.

## License

Licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See `LICENSE` file. 
