import sys
import os
import pickle

import nltk
from tmtoolkit.preprocess._tmpreproc import LANGUAGE_LABELS

DATADIR = '../tmtoolkit/data'
FILENAME = 'stopwords.pickle'

for lang_code, lang_label in LANGUAGE_LABELS.items():
    try:
        stopwords = nltk.corpus.stopwords.words(lang_label)
    except OSError:
        print('could not load stopwords for language "%s"' % lang_label, file=sys.stderr)
        continue

    lang_dir = os.path.join(DATADIR, lang_code)

    if not os.path.exists(lang_dir):
        os.mkdir(lang_dir, 0o755)

    stop_file = os.path.join(lang_dir, FILENAME)

    with open(stop_file, 'wb') as f:
        print('saving stopwords for "%s" to file "%s"' % (lang_label, stop_file))
        pickle.dump(stopwords, f)
