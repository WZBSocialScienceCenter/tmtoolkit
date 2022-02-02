"""
Example script that loads and processes the proceedings of the 18th German Bundestag and generates a tf-idf matrix.
The data is quite large, consisting of 15,733 documents with 14,355,341 tokens in total. This script shows how to
handle large data efficiently by using the parallel processing power of tmtoolkit and sparse matrix calculations
that use few memory.

Note that it is highly advisable to run this script section by section (denoted with "#%%" or even line by line in an
interactive Python interpreter in order to see the effects of each code block.

The data for the debates comes from offenesparlament.de, see https://github.com/Datenschule/offenesparlament-data.

Markus Konrad <markus.konrad@wzb.eu>
June 2019 / Feb. 2022
"""

import re
import pickle
import string
import random
from pprint import pprint
from zipfile import ZipFile

from tmtoolkit.corpus import Corpus, print_summary, corpus_unique_chars, save_corpus_to_picklefile,\
    load_corpus_from_picklefile, normalize_unicode, simplify_unicode, remove_chars, tokens_table, vocabulary
from tmtoolkit.bow.bow_stats import tfidf, sorted_terms_table
from tmtoolkit.utils import enable_logging, unpickle_file, pickle_data
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

#%% Correct contractions

# some contractions have a stray space in between, like "EU -Hilfen" where it should be "EU-Hilfen"
# correct this by applying a custom function with a regular expression (RE) to each document in the corpus
pttrn_contraction_ws = re.compile(r'(\w+)(\s+)(-\w+)')

# in each document text `t`, remove the RE group 2 (the stray white space "(\s+)") for each match `m`
# we will pass this function as "raw_preproc" function
def correct_contractions(t):
    return pttrn_contraction_ws.sub(lambda m: m.group(1) + m.group(3), t)


#%% Generate a Corpus object


# we use the column "doc_label" as document labels and "text" as raw text
print('creating corpus object')
corpus = Corpus(bt18_texts, language='de', max_workers=1.0, raw_preproc=correct_contractions)

# we don't need this anymore, remove it to free memory
del bt18_texts

save_corpus_to_picklefile(corpus, 'data/bt18_corpus.pickle')
#corpus = load_corpus_from_picklefile('data/bt18_corpus.pickle')

print_summary(corpus)

#%% Investigate the set of characters used in the whole corpus

# we can see that there are several "strange" characters and unprintable unicode characters which may later cause
# trouble
print('used set of characters used in the whole corpus:')
corpus_unique_chars(corpus)

# lets see which of these are not in Pythons standard set of printable characters
print('used set of characters not in Pythons standard set of printable characters:')
pprint(corpus_unique_chars(corpus) - set(string.printable))

#%% Replace some characters in each of document of the corpus

normalize_unicode(corpus)
simplify_unicode(corpus)

nonprintable = corpus_unique_chars(corpus) - set(string.printable)
remove_chars(corpus, nonprintable)

sorted(corpus_unique_chars(corpus))

#%%

save_corpus_to_picklefile(corpus, 'data/bt18_corpus_2.pickle')

#%% Have a glimpse at the tokens

tokens_table(corpus)


#%% Have a look at the vocabulary of the whole corpus
print('vocabulary:')
pprint(preproc.vocabulary)

print('\nvocabulary contains %d tokens' % len(preproc.vocabulary))

#%% Fix hyphenation problems

# we can see in the above vocabulary that there are several hyphenation problems (e.g. "wiederho-len"), because of
# words being hyphenated on line breaks
# we use a quite "brutal" way to fix this by simply removing all hyphens in the tokens

preproc.remove_chars_in_tokens(['-'])

print('vocabulary:')
pprint(preproc.vocabulary)

print('\nvocabulary contains %d tokens' % len(preproc.vocabulary))


#%% Display a keywords-in-context (KWIC) table

# the result is returned as *datatable* (because it is much faster to construct)
print('keywords-in-context (KWIC) table for keyword "Merkel":')
print(preproc.get_kwic_table('Merkel'))

#%% Apply Part-of-Speech tagging (POS tagging) and lemmatization to normalize the vocabulary

# this is very computationally extensive and hence takes a long time, even when computed in parallel
# consider storing / loading the processing state as shown below
preproc.pos_tag().lemmatize()

#%% Saving / loading state

# at any time you can save the current processing state to disk via `save_state(<path to file>)` and later
# restore it via `from_state(<path to file>)`
# this is extremely useful when you have computations that take a long time and after which you want to create
# "save points" in order to load the state and continue experimenting with the data without having to run the
# whole processing pipeline again

# preproc.save_state('data/bt18_tagged_lemmatized_state.pickle')
# preproc = TMPreproc.from_state('data/bt18_tagged_lemmatized_state.pickle')

#%% Further token normalization

# convert all tokens to lowercase and apply several "cleaning" methods (see `clean_tokens` for details)
print('applying further token normalization')
preproc.tokens_to_lowercase().clean_tokens().remove_tokens(r'^-.+', match_type='regex')

print('vocabulary:')
pprint(preproc.vocabulary)

print('\nvocabulary contains %d tokens' % len(preproc.vocabulary))

# there are still some stray tokens which should be removed:
preproc.remove_tokens(['#en', "''", "'s", '+++', '+40', ',50', '...', '.plädieren'])

#%% Let's have a look at the most frequent tokens

print('retrieving document frequencies for all tokens in the vocabulary')
vocab_doc_freq = preproc.vocabulary_rel_doc_frequency
vocab_doc_freq_df = pd.DataFrame({'token': list(vocab_doc_freq.keys()),
                                  'freq': list(vocab_doc_freq.values())})

print('top 50 tokens by relative document frequency:')
vocab_top = vocab_doc_freq_df.sort_values('freq', ascending=False).head(50)
print(vocab_top)

# plot this
plt.figure()
vocab_top.plot(x='token', y='freq', kind='bar')
plt.show()

#%% Further token cleanup

# we can remove tokens above a certain threshold of (relative or absolute) document frequency
preproc.remove_common_tokens(0.8)   # this will only remove "müssen"

# since we'll later use tf-idf, common words don't have much influence on the result and can remain

#%% Document lengths (number of tokens per document)

doc_labels = np.array(list(preproc.doc_lengths.keys()))
doc_lengths = np.array(list(preproc.doc_lengths.values()))

print('range of document lengths: %d tokens minimum, %d tokens maximum' % (np.min(doc_lengths), np.max(doc_lengths)))
print('mean document length:', np.mean(doc_lengths))
print('mean document length:', np.median(doc_lengths))

plt.figure()
plt.hist(doc_lengths, bins=100)
plt.title('Histogram of document lengths')
plt.xlabel('Number of tokens')
plt.show()


#%% Let's have a look at very short document

# document labels of documents with lesser or equal 30 tokens
doc_labels_short = doc_labels[doc_lengths <= 30]

print('%d documents with lesser or equal 30 tokens:' % len(doc_labels_short))
for dl in doc_labels_short:
    print(dl)
    pprint(' '.join(preproc.tokens[dl]))
    print('---')


#%% Remove very short documents

print('removing documents with lesser or equal 30 tokens')
preproc.remove_documents_by_name(doc_labels_short)


#%% Another keywords-in-context (KWIC) table

print('keywords-in-context (KWIC) table for keyword "merkel" with normalized tokens:')
print(preproc.get_kwic_table('merkel'))

#%% Create a document-term-matrix (DTM)

# this creates a sparse DTM where the matrix rows correspond to the current document labels and the
# matrix columns correspond to the current vocabulary
# the calculations take several minutes, even when they're performed in parallel

print('creating document-term-matrix (DTM)')
dtm = preproc.dtm

print('matrix created:')
print(repr(dtm))

doc_labels = preproc.doc_labels
vocab = np.array(preproc.vocabulary)

print('number of rows match number of documents (%d)' % len(doc_labels))
print('number of columns match vocabulary size (%d)' % len(vocab))


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
print('20 most "informative" (tf-idf high ranked) tokens in randomly chosen document "%s":' % random_doc)


if has_datatable:
    print(top_tokens[dt.f.doc == random_doc, :])
else:
    print(top_tokens[top_tokens.doc == random_doc])
