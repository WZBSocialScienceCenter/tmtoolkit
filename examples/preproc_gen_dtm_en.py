# -*- coding: utf-8 -*-
"""
An example for preprocessing documents in German language and generating a document-term-matrix (DTM).
"""
from pprint import pprint
from tmtoolkit.preprocess import TMPreproc

corpus = {
    'doc1': u'A simple example in simple English.',
    'doc2': u'It contains only three very simple documents.',
    'doc3': u'Simply written documents are very brief.',
}

preproc = TMPreproc(corpus, language='english')

print('input corpus:')
pprint(corpus)

print('running preprocessing pipeline...')
preproc.tokenize().pos_tag().lemmatize().tokens_to_lowercase().clean_tokens()

print('final tokens:')
pprint(preproc.tokens)

print('DTM:')
doc_labels, vocab, dtm = preproc.get_dtm()

import pandas as pd
print(pd.DataFrame(dtm.todense(), columns=vocab, index=doc_labels))
