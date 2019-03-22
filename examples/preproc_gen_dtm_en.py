"""
An example for preprocessing documents in English language and generating a document-term-matrix (DTM).

Markus Konrad <markus.konrad@wzb.eu>
"""

from pprint import pprint

import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.bow.dtm import dtm_to_dataframe


# IMPORTANT NOTE FOR WINDOWS USERS:
# You must put everything below inside the following "if" clause. This is necessary for multiprocessing on Windows.
#
# if __name__ == '__main__':

#%% Define a simple example corpus and pass it to "TMPreproc" for preprocessing

corpus = {
    'doc1': 'A simple example in simple English.',
    'doc2': 'It contains only three (in numbers: 3) very simple documents.',
    'doc3': 'Simply written documents are very brief.',
}

preproc = TMPreproc(corpus, language='english')

print('input corpus:')
pprint(corpus)

#%% run a typical preprocessing pipeline

print('running preprocessing pipeline...')
preproc.pos_tag().lemmatize().tokens_to_lowercase().clean_tokens(remove_numbers=True)

#%% print tokens

print('final tokens:')
pprint(preproc.tokens)

#%% Generate document-term-matrix (DTM)

print('DTM as data frame:')

dtm_df = dtm_to_dataframe(preproc.dtm, preproc.doc_labels, preproc.vocabulary)
print(dtm_df)

print('done.')
