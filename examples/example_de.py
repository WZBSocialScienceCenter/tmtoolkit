# -*- coding: utf-8 -*-

from tmtoolkit.preprocess import TMPreproc
#from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger


corpus = {
    u'doc1': u'Ein einfaches Beispiel in einfachem Deutsch.',
    u'doc2': u'Es enthält nur drei sehr einfache Dokumente.',
    u'doc3': u'Die Dokumente sind sehr kurz.',
}

preproc = TMPreproc(corpus, language=u'german')

print(u'tokenized:')
print(preproc.tokenize())

print(u'POS tagged:')
print(preproc.pos_tag())

#print(preproc.stem())

print(u'lemmatized:')
preproc.load_lemmata_dict()
print(preproc.lemmatize())

print(u'lowercase:')
print(preproc.tokens_to_lowercase())

print(u'cleaned:')
print(preproc.clean_tokens())

print(u'DTM:')
doc_labels, vocab, dtm = preproc.get_dtm()

import pandas as pd
print(pd.DataFrame(dtm.todense(), columns=vocab, index=doc_labels))
#print(dtm.todense())