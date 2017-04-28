# -*- coding: utf-8 -*-

from tm_prep.preprocess import TMPreproc
#from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger


corpus = {
    u'doc1': u'Ein einfaches Beispiel auf Deutsch.',
    u'doc2': u'Es enthält nur zwei Dokumente.',
}

preproc = TMPreproc(corpus, language=u'german')

print(preproc.tokenize())

print(preproc.pos_tag())

#print(preproc.stem())

preproc.load_lemmata_dict()
print(preproc.lemmatize(use_dict=True, use_patternlib=True))

print(preproc.tokens_to_lowercase())