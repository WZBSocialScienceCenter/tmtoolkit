"""
An example for preprocessing documents in German language and generating a document-term-matrix (DTM).

Markus Konrad <markus.konrad@wzb.eu>
"""

from pprint import pprint

from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.bow.dtm import dtm_to_dataframe

# IMPORTANT NOTE FOR WINDOWS USERS:
# You must put everything below inside the following "if" clause. This is necessary for multiprocessing on Windows.
#
# if __name__ == '__main__':

#%% Define a simple example corpus and pass it to "TMPreproc" for preprocessing

corpus = {
    'doc1': 'Ein einfaches Beispiel in einfachem Deutsch.',
    'doc2': 'Es enthält nur drei sehr einfache Dokumente.',
    'doc3': 'Die Dokumente sind sehr kurz.',
}

# this will directly tokenize the documents
preproc = TMPreproc(corpus, language='german')

#%% show tokenized documents

pprint(preproc.tokens)

#%% show tokenized documents as data frame

print(preproc.tokens_dataframe)

#%% POS tagging

preproc.pos_tag()

print('POS tagged:')
print(preproc.tokens_dataframe)

#%% Lemmatization

print('lemmatized:')
preproc.lemmatize()
print(preproc.tokens_dataframe)

#%% Lower-case transformation

print('lowercase:')
preproc.tokens_to_lowercase()
print(preproc.tokens_dataframe)

#%% Clean tokens (remove stopwords and punctuation)

print('cleaned:')
preproc.clean_tokens()
print(preproc.tokens_dataframe)

#%% Generate document-term-matrix (DTM)

print('DTM as data frame:')

dtm_df = dtm_to_dataframe(preproc.dtm, preproc.doc_labels, preproc.vocabulary)
print(dtm_df)

print('done.')
