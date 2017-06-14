# -*- coding: utf-8 -*-
"""
An example for constructing a corpus of texts from files and passing them to the preprocessing step.
"""
from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc

#corpus = Corpus().from_files(['data/gutenberg/kafka_verwandlung.txt'])
#print(corpus.docs.keys())

corpus = Corpus().from_folder('data/gutenberg')
print("all loaded documents:")
print(corpus.docs.keys())
print("-----")

corpus.split_by_paragraphs()
print("documents split into paragraphs")
print(corpus.docs.keys())
print("-----")

print("first 5 paragraphs of Werther:")
for par_num in range(1, 6):
    doclabel = u'werther-goethe_werther1-%d' % par_num
    print(u"par%d (document label '%s'):" % (par_num, doclabel))
    print(corpus.docs[doclabel])
print("-----")

preproc = TMPreproc(corpus.docs, language=u'german')
preproc.tokenize()

print("tokenized first 5 paragraphs of Werther:")
for par_num in range(1, 6):
    doclabel = u'werther-goethe_werther1-%d' % par_num
    print(u"par%d (document label '%s'):" % (par_num, doclabel))
    print(preproc.tokens[doclabel])
