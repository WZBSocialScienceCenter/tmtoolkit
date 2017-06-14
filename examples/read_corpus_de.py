# -*- coding: utf-8 -*-
"""
An example for constructing a corpus of texts from files and passing them to the preprocessing step.
"""
from pprint import pprint

from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc

#corpus = Corpus().from_files(['data/gutenberg/kafka_verwandlung.txt'])
#print(corpus.docs.keys())

corpus = Corpus().from_folder('data/gutenberg')
print(corpus.docs.keys())

corpus.split_by_paragraphs()
print(corpus.docs.keys())

for par_num in range(1, 6):
    doclabel = u'werther-goethe_werther1-%d' % par_num
    print(u"par%d (document label '%s'):" % (par_num, doclabel))
    print(corpus.docs[doclabel])
