"""
Example script that loads and processes the proceedings of the 18th German Bundestag and generates a tf-idf matrix.
The data is quite large, consisting of 15,733 documents with 14,355,341 tokens in total. This script shows how to
handle large data efficiently by using the parallel processing power of tmtoolkit and sparse matrix calculations
that use few memory.

Note that it is highly advisable to run this script section by section (denoted with "#%%" or even line by line in an
interactive Python interpreter in order to see the effects of each code block.

The data for the debates comes from offenesparlament.de, see https://github.com/Datenschule/offenesparlament-data.

This examples requires that you have installed tmtoolkit with the recommended set of packages and have installed a
German language model for spaCy:

    pip install -U "tmtoolkit[recommended]"
    python -m tmtoolkit setup de

For more information, see the installation instructions: https://tmtoolkit.readthedocs.io/en/latest/install.html

Markus Konrad <markus.konrad@wzb.eu>
June 2019 / Feb. 2022
"""

import re
import pickle
import string
import random
from pprint import pprint
from zipfile import ZipFile

from tmtoolkit import corpus as c
from tmtoolkit.corpus import visualize as cvis
from tmtoolkit.tokenseq import unique_chars
from tmtoolkit.bow.bow_stats import tfidf, sorted_terms_table
from tmtoolkit.utils import enable_logging, pickle_data, unpickle_file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 100)

#%% Optional: set up output log for tmtoolkit

enable_logging()

#%% Load the data inside the zip file

print('loading data from zip file')

with ZipFile('data/bt18_full.zip') as bt18zip:
    # there is a pickled pandas data frame inside the zipfile
    # extract it and load it
    bt18pickle = bt18zip.read('bt18_speeches_merged.pickle')
    bt18_data = pickle.loads(bt18pickle)

# we don't need this anymore, remove it to free memory
del bt18pickle, bt18zip

#%% Generate document labels

# format of the document labels: <session_number>_<speech_number>
bt18_data['doc_label'] = ['%s_%s' % (str(sitzung).zfill(3), str(seq).zfill(5))
                          for sitzung, seq in zip(bt18_data.sitzung, bt18_data.sequence)]

print('loaded data frame with %d rows:' % bt18_data.shape[0])
print(bt18_data.head())

bt18_texts = dict(zip(bt18_data.doc_label, bt18_data.text))
del bt18_data


#%% Prepare raw text data preprocessing

# remove some special characters

corpus_chars = unique_chars(bt18_texts.values())
print('special characters in text data:')
pprint(sorted(corpus_chars - set(string.printable)))

keepchars = set('óéıàŁŽńôíśžê̆č€şćÖÇ₂ãüÀÄłÜšěŠźēûçÉöáåúèäßëğîǧҫČœřïñ§°')
delchars = corpus_chars - set(string.printable) - keepchars
print(f'will remove characters: {delchars}')

delchars_table = str.maketrans('', '', ''.join(delchars))

# we will pass this function as "raw_preproc" function
def del_special_chars(t):
    return t.translate(delchars_table)

# some contractions have a stray space in between, like "EU -Hilfen" where it should be "EU-Hilfen"
# correct this by applying a custom function with a regular expression (RE) to each document in the corpus
pttrn_contraction_ws = re.compile(r'(\w+)(\s+)(-\w+)')

# in each document text `t`, remove the RE group 2 (the stray white space "(\s+)") for each match `m`
# we will pass this function as "raw_preproc" function
def correct_contractions(t):
    return pttrn_contraction_ws.sub(lambda m: m.group(1) + m.group(3), t)


# correct hyphenation issues in the documents like "groß-zügig"
# we will pass this function as "raw_preproc" function
pttrn_hyphenation = re.compile(r'([a-zäöüß])-([a-zäöüß])')
def correct_hyphenation(t):
    return pttrn_hyphenation.sub(lambda m: m.group(1) + m.group(2), t)


#%% Generate a Corpus object


# we use the column "doc_label" as document labels and "text" as raw text
print('creating corpus object')
corpus = c.Corpus(bt18_texts, language='de', max_workers=1.0,
                  raw_preproc=[del_special_chars, correct_contractions, correct_hyphenation])

# we don't need this anymore, remove it to free memory
del bt18_texts

c.print_summary(corpus)

#%% storing a Corpus object

# at any time, we may store a Corpus object to disk via `save_corpus_to_picklefile` and later load it
# via `load_corpus_from_picklefile`; this helps you to prevent long running computations again

# c.save_corpus_to_picklefile(corpus, 'data/bt18_corpus.pickle')
# corpus = load_corpus_from_picklefile('data/bt18_corpus.pickle')

#%% Have a look at the vocabulary of the whole corpus
print('vocabulary:')
pprint(c.vocabulary(corpus))

print(f'\nvocabulary contains {c.vocabulary_size(corpus)} tokens')

#%% Display a keywords-in-context (KWIC) table

print('keywords-in-context (KWIC) table for keyword "Merkel":')
print(c.kwic_table(corpus, 'Merkel'))

#%% Text normalization

# lemmatization
c.lemmatize(corpus)

# convert all tokens to lowercase and apply several "cleaning" methods
print('applying further token normalization')
c.to_lowercase(corpus)
c.filter_clean_tokens(corpus)
c.remove_tokens(corpus, r'^-.+', match_type='regex')

print('vocabulary:')
pprint(c.vocabulary(corpus))

print(f'\nvocabulary contains {c.vocabulary_size(corpus)} tokens')

# there are still some stray tokens which should be removed:
c.remove_tokens(corpus, ['+40', '+', '.plädieren'])

#%% Let's have a look at the most frequent tokens

print('retrieving document frequencies for all tokens in the vocabulary')
c.vocabulary_counts(corpus, proportions=1, as_table='-freq').head(50)

# the rank - count plot shows quite a deviation from Zipf's law, because we already applied some token normalization
fig, ax = plt.subplots()
cvis.plot_ranked_vocab_counts(fig, ax, corpus, zipf=True)
plt.show()

#%% Further token cleanup

# we can remove tokens above a certain threshold of (relative or absolute) document frequency
c.remove_common_tokens(corpus, df_threshold=0.8)

# since we'll later use tf-idf, removing very common or very uncommon tokens may not even be necessary; however
# it reduces the computation time and memory consumption of all downstream tasks

#%% Document lengths (number of tokens per document)

fig, ax = plt.subplots()
cvis.plot_doc_lengths_hist(fig, ax, corpus)
plt.show()


#%% Let's have a look at very short documents

docsizes = c.doc_lengths(corpus, as_table='length')

# document labels of documents with lesser or equal 30 tokens
doc_labels_short = docsizes.doc[docsizes.length <= 30]
doc_labels_short_texts = c.doc_texts(corpus, select=doc_labels_short, collapse=' ')

print(f'{len(doc_labels_short)} documents with lesser or equal 30 tokens:')
for lbl, txt in doc_labels_short_texts.items():
    print(lbl)
    pprint(txt)
    print('---')


#%% Remove very short documents

print('removing documents with lesser or equal 30 tokens')
c.remove_documents_by_label(corpus, doc_labels_short.to_list())


#%% Another keywords-in-context (KWIC) table

print('keywords-in-context (KWIC) table for keyword "merkel" with normalized tokens:')
print(c.kwic_table(corpus, 'merkel'))

#%% Create a document-term-matrix (DTM)

# this creates a sparse DTM where the matrix rows correspond to the current document labels and the
# matrix columns correspond to the current vocabulary
# the calculations take several minutes, even when they're performed in parallel

print('creating document-term-matrix (DTM)')
dtm = c.dtm(corpus)

print('matrix created:')
print(repr(dtm))

doc_labels = np.array(c.doc_labels(corpus))
vocab = np.array(c.vocabulary(corpus))


#%% Saving / loading a DTM

# again, you may store the DTM along with the document labels and vocabulary to disk to later load it again:

# pickle_data((dtm, doc_labels, vocab), 'data/bt18_dtm.pickle')
# dtm, doc_labels, vocab = unpickle_file('data/bt18_dtm.pickle')


#%% Computing a tf-idf matrix

# we can apply tf-idf to the DTM
# the result will remain a sparse matrix, hence it doesn't allocate much memory

print('computing a tf-idf matrix from the DTM')
tfidf_mat = tfidf(dtm)
print('matrix created:')
print(repr(tfidf_mat))

#%% Investigating the top tokens of the tf-idf transformed matrix

# this will create a data frame of the 20 most "informative" (tf-idf-wise) tokens per document
top_tokens = sorted_terms_table(tfidf_mat, vocab, doc_labels, top_n=20)

random_doc = random.choice(doc_labels)
print(f'20 most "informative" (tf-idf high ranked) tokens in randomly chosen document "{random_doc}":')

print(top_tokens[top_tokens.index.get_level_values(0) == random_doc])
