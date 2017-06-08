# -*- coding: utf-8 -*-
"""
An example for preprocessing documents in German language and generating a document-term-matrix (DTM).
"""
from pprint import pprint

from tmtoolkit.preprocess import TMPreproc


corpus = {
    u'doc1': u'Ein einfaches Beispiel in einfachem Deutsch.',
    u'doc2': u'Es enthält nur drei sehr einfache Dokumente.',
    u'doc3': u'Die Dokumente sind sehr kurz.',
}

preproc = TMPreproc(corpus, language='german')

print('tokenized:')
pprint(preproc.tokenize())

print('POS tagged:')
pprint(preproc.pos_tag())

#print(preproc.stem())

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
