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

print('tokenized:')
pprint(preproc.tokenize())

print('POS tagged:')
pprint(preproc.pos_tag())

print('lemmatized:')
pprint(preproc.lemmatize())

print('lowercase:')
pprint(preproc.tokens_to_lowercase())

print('cleaned:')
pprint(preproc.clean_tokens())

print('DTM:')
doc_labels, vocab, dtm = preproc.get_dtm()

import pandas as pd
print(pd.DataFrame(dtm.todense(), columns=vocab, index=doc_labels))
