# -*- coding: utf-8 -*-
"""
An example for constructing a corpus of texts from files and passing them to the proprocessing step.
"""
from pprint import pprint

from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc

#corpus = Corpus().from_files(['data/gutenberg/kafka_verwandlung.txt'])
#print(corpus.docs.keys())

corpus = Corpus().from_folder('data/gutenberg')
print(corpus.docs.keys())
