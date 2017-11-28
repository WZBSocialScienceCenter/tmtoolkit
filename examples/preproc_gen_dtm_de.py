# -*- coding: utf-8 -*-
"""
An example for preprocessing documents in German language and generating a document-term-matrix (DTM).
"""
from pprint import pprint

import pandas as pd
from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.utils import pickle_data


if __name__ == '__main__':   # this is necessary for multiprocessing on Windows!
    corpus = {
        u'doc1': u'Ein einfaches Beispiel in einfachem Deutsch.',
        u'doc2': u'Es enthält nur drei sehr einfache Dokumente.',
        u'doc3': u'Die Dokumente sind sehr kurz.',
    }

    preproc = TMPreproc(corpus, language='german')

    print('tokenized:')
    preproc.tokenize()
    pprint(preproc.tokens)

    # preproc.stem()
    # pprint(preproc.tokens)

    print('POS tagged:')
    preproc.pos_tag()
    pprint(preproc.tokens_with_pos_tags)

    print('lemmatized:')
    preproc.lemmatize()
    pprint(preproc.tokens_with_pos_tags)

    print('lowercase:')
    preproc.tokens_to_lowercase()
    pprint(preproc.tokens)

    print('cleaned:')
    preproc.clean_tokens()
    pprint(preproc.tokens_with_pos_tags)
    pprint(preproc.tokens)

    print('filtered:')
    preproc.filter_for_token(u'einfach', remove_found_token=True)
    preproc.filter_for_pos('N')
    pprint(preproc.tokens_with_pos_tags)

    print('saving tokens as pickle...')
    pickle_data(preproc.tokens, 'data/preproc_gen_dtm_de_tokens.pickle')

    print('DTM:')
    doc_labels, vocab, dtm = preproc.get_dtm()

    print(pd.DataFrame(dtm.todense(), columns=vocab, index=doc_labels))
