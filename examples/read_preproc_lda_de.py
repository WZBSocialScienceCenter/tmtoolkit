# -*- coding: utf-8 -*-
"""
requires "europarl_raw" corpus to be downloaded via `nltk.download()`
"""
import os
import time
import logging
from random import sample

import nltk
import lda


from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.dtm import save_dtm_to_pickle, load_dtm_from_pickle
from tmtoolkit.lda_utils.common import print_ldamodel_topic_words, print_ldamodel_doc_topics, save_ldamodel_to_pickle


FILES = """ep-00-01-17.de
ep-00-01-18.de
ep-00-01-19.de
ep-00-01-20.de
ep-00-01-21.de
ep-00-02-02.de
ep-00-02-03.de
ep-00-02-14.de
ep-00-02-15.de
ep-00-02-16.de""".split('\n')
FILEIDS = ['german/' + f for f in FILES]

DTM_PICKLE = 'data/read_preproc_lda_de_dtm.pickle'
LDA_PICKLE = 'data/read_preproc_lda_de_lda.pickle'

logging.basicConfig(level=logging.DEBUG)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.DEBUG)
tmtoolkit_log.propagate = True


if __name__ == '__main__':   # this is necessary for multiprocessing on Windows!
    if os.path.exists(DTM_PICKLE):
        print("loading DTM data from pickle file '%s'..." % DTM_PICKLE)

        dtm, vocab, doc_labels = load_dtm_from_pickle(DTM_PICKLE)
    else:
        europarl = nltk.corpus.util.LazyCorpusLoader('europarl_raw',
                                                     nltk.corpus.EuroparlCorpusReader,
                                                     fileids=FILEIDS)

        corpus = Corpus({f: europarl.raw(f_id) for f, f_id in zip(FILES, FILEIDS)})

        print("all loaded documents:")
        for dl, text in corpus.docs.items():
            print("%s: %d chars" % (dl, len(text)))
        print("-----")

        start_time = time.time()
        preproc = TMPreproc(corpus.docs, language=u'german')
        print('tokenizing...')
        preproc.tokenize()
        print('POS tagging...')
        preproc.pos_tag()
        print('lemmatization...')
        preproc.lemmatize()
        print('lowercase transform...')
        preproc.tokens_to_lowercase()
        print('cleaning...')
        preproc.clean_tokens()

        proc_time = time.time() - start_time
        print('-- processing took %f sec. so far' % proc_time)

        preproc.save_state('data/read_preproc_lda_de_state.pickle')

        print('token samples:')
        for dl, tokens in preproc.tokens_with_pos_tags.items():
            print("> %s:" % dl)
            print(">>", sample(tokens, 10))

        print('generating DTM...')
        doc_labels, vocab, dtm = preproc.get_dtm()

        print("saving DTM data to pickle file '%s'..." % DTM_PICKLE)
        save_dtm_to_pickle(dtm, vocab, doc_labels, DTM_PICKLE)

    print("running LDA...")
    # note: this won't result in a good topic model. it's only here for demonstration purposes.
    # we should increase the number of iterations and also do some evaluation to get the "correct" number of topics.
    model = lda.LDA(n_topics=30, n_iter=500)
    model.fit(dtm)

    # print topic-word distributions with respective probabilities
    print_ldamodel_topic_words(model.topic_word_, vocab)

    # print document-topic distributions with respective probabilities
    print_ldamodel_doc_topics(model.doc_topic_, doc_labels)

    print("saving LDA model to pickle file '%s'..." % LDA_PICKLE)
    save_ldamodel_to_pickle(LDA_PICKLE, model, vocab, doc_labels)

    print("done.")
