import os
import string
from importlib import import_module
from collections import defaultdict

import nltk

from germalemma import GermaLemma
from dtm import create_sparse_dtm, get_vocab_and_terms
from utils import require_listlike, require_dictlike, unpickle_file, remove_tokens_from_list


class TMPreproc(object):
    DATAPATH = os.path.join('tmtoolkit', 'data')
    PATTERN_SUBMODULES = {
        'english': 'en',
        'german': 'de',
        'spanish': 'es',
        'french': 'fr',
        'italian': 'it',
        'dutch': 'nl',
    }

    POS_TAGGER_PICKLE = u'pos_tagger.pickle'
    LEMMATA_PICKLE = u'lemmata.pickle'

    def __init__(self, docs, language=u'english', stopwords=None, punctuation=None):
        require_dictlike(docs)

        self.docs = docs
        self.language = language

        if stopwords is None:
            self.stopwords = nltk.corpus.stopwords.words(language)
        else:
            self.stopwords = stopwords

        if punctuation is None:
            self.punctuation = list(string.punctuation)

        self.load_tokenizer(lambda x: nltk.tokenize.word_tokenize(x, self.language))

        self.stemmer = None
        self.lemmata_dict = None
        self.pos_tagger = None

        self.tokens = None
        self.tokens_pos_tags = None

        self.pattern_module = None
        self.germalemma = None   # GermaLemma instance

    def add_stopwords(self, stopwords):
        require_listlike(stopwords)
        self.stopwords += stopwords

    def add_punctuation(self, punctuation):
        require_listlike(punctuation)
        self.punctuation += punctuation

    def load_tokenizer(self, custom_tokenizer):
        self.tokenizer = custom_tokenizer

    def load_stemmer(self, custom_stemmer=None):
        if custom_stemmer:
            self.stemmer = custom_stemmer
        else:
            self.stemmer = nltk.stem.SnowballStemmer(self.language)

    def load_lemmata_dict(self, custom_lemmata_dict=None):
        if custom_lemmata_dict:
            self.lemmata_dict = custom_lemmata_dict
        else:
            picklefile = os.path.join(self.DATAPATH, self.language, self.LEMMATA_PICKLE)
            unpickled_obj = unpickle_file(picklefile)
            if type(unpickled_obj) is dict:
                self.lemmata_dict = unpickled_obj
            elif type(unpickled_obj) is tuple and len(unpickled_obj) == 2:
                self.lemmata_dict = unpickled_obj[0]
            else:
                raise ValueError("object of invalid data type in lemmata pickle file")

    def load_pos_tagger(self, custom_pos_tagger=None):
        if custom_pos_tagger:
            self.pos_tagger = custom_pos_tagger
        else:
            picklefile = os.path.join(self.DATAPATH, self.language, self.POS_TAGGER_PICKLE)
            self.pos_tagger = unpickle_file(picklefile)

    def tokenize(self):
        if not callable(self.tokenizer):
            raise ValueError('tokenizer must be callable')

        self.tokens = {dl: self.tokenizer(txt) for dl, txt in self.docs.items()}

        return self.tokens

    def token_transform(self, transform_fn):
        if not callable(transform_fn):
            raise ValueError('transform_fn must be callable')

        self._require_tokens()

        self.tokens = {dl: map(transform_fn, dt) for dl, dt in self.tokens.items()}

        return self.tokens

    def tokens_to_lowercase(self):
        return self.token_transform(string.lower)

    def stem(self):
        self._require_tokens()

        if not self.stemmer:
            self.load_stemmer()

        stemmer = self.stemmer
        if not hasattr(stemmer, 'stem') or not callable(stemmer.stem):
            raise ValueError("stemmer must have a callable attribute `stem`")

        self.tokens = {dl: [stemmer.stem(t) for t in dt]
                       for dl, dt in self.tokens.items()}

        return self.tokens

    def lemmatize(self, use_dict=True, use_patternlib=False, use_germalemma=None):
        self._require_tokens()
        self._require_pos_tags()

        tmp_lemmata = defaultdict(list)

        if use_germalemma is None and self.language == 'german':
            use_germalemma = True

        if use_germalemma:
            if not self.germalemma:
                lemmata_pickle = os.path.join(self.DATAPATH, self.language, self.LEMMATA_PICKLE)
                lemmata_dict, lemmata_lower_dict = unpickle_file(lemmata_pickle)
                self.germalemma = GermaLemma(lemmata=lemmata_dict, lemmata_lower=lemmata_lower_dict)

            for dl, dt in self.tokens.items():
                tok_tags = self.tokens_pos_tags[dl]
                for t, pos in zip(dt, tok_tags):
                    try:
                        l = self.germalemma.find_lemma(t, pos)
                    except ValueError:
                        l = t
                    tmp_lemmata[dl].append(l)
        else:
            if use_dict:
                if not self.lemmata_dict:
                    self.load_lemmata_dict()

                for dl, dt in self.tokens.items():
                    tok_tags = self.tokens_pos_tags[dl]
                    for t, pos in zip(dt, tok_tags):
                        if pos.startswith('N') or pos.startswith('V'):
                            pos = pos[0]
                        elif pos.startswith('ADJ') or pos.startswith('ADV'):
                            pos = pos[:3]
                        else:
                            pos = None

                        if pos:
                            l = self.lemmata_dict.get(pos, {}).get(t, None)
                            if l == '-' or l == '':
                                l = None
                        else:
                            l = None
                        tmp_lemmata[dl].append(l)

            if use_patternlib:
                self._require_pos_tags()

                if not self.pattern_module:
                    if self.language not in self.PATTERN_SUBMODULES:
                        raise ValueError("no CLiPS pattern module for this language:", self.language)

                    modname = 'pattern.%s' % self.PATTERN_SUBMODULES[self.language]
                    self.pattern_module = import_module(modname)

                for dl, dt in self.tokens.items():
                    tok_tags = self.tokens_pos_tags[dl]
                    tok_lemmata = tmp_lemmata.get(dl, [None] * len(dt))

                    lemmata_final = []
                    for t, t_found, pos in zip(dt, tok_lemmata, tok_tags):
                        l = t_found

                        if l is None:
                            if pos.startswith('NP'):     # singularize noun
                                l = self.pattern_module.singularize(t)
                            elif pos.startswith('V'):   # get infinitive of verb
                                l = self.pattern_module.conjugate(t, self.pattern_module.INFINITIVE)
                            elif pos.startswith('ADJ') or pos.startswith('ADV'):  # get baseform of adjective or adverb
                                l = self.pattern_module.predicative(t)

                        lemmata_final.append(l)

                    tmp_lemmata[dl] = lemmata_final

        # merge
        lemmatized_tokens = {}
        for dl in self.tokens.keys():
            new_dt = [l or t for t, l in zip(self.tokens[dl], tmp_lemmata.get(dl, []))]
            assert len(new_dt) == len(self.tokens[dl])
            lemmatized_tokens[dl] = new_dt

        assert len(lemmatized_tokens) == len(self.docs)
        self.tokens = lemmatized_tokens

        return self.tokens

    def clean_tokens(self, remove_punct=True, remove_stopwords=True):
        tokens_to_remove = []

        if remove_punct:
            tokens_to_remove.extend(self.punctuation)
        if remove_stopwords:
            tokens_to_remove.extend(self.stopwords)

        if tokens_to_remove:
            self.tokens = {dl: remove_tokens_from_list(dt, tokens_to_remove)
                           for dl, dt in self.tokens.items()}

        return self.tokens

    def pos_tag(self):
        self._require_tokens()

        if not self.pos_tagger:
            self.load_pos_tagger()

        tagger = self.pos_tagger
        if not hasattr(tagger, 'tag') or not callable(tagger.tag):
            raise ValueError("pos_tagger must have a callable attribute `tag`")

        self.tokens_pos_tags = {dl: list(zip(*tagger.tag(dt)))[1] for dl, dt in self.tokens.items()}

        return self.tokens_pos_tags

    def get_dtm(self):
        vocab, doc_labels, docs_terms, dtm_alloc_size = get_vocab_and_terms(self.tokens)
        dtm = create_sparse_dtm(vocab, doc_labels, docs_terms, dtm_alloc_size)
        return doc_labels, vocab, dtm

    def _require_tokens(self):
        if not self.tokens:
            raise ValueError("documents must be tokenized before this operation")

        if len(self.docs) != len(self.tokens):
            raise ValueError("number of input documents and number of documents in tokens is not equal")

    def _require_pos_tags(self):
        if not self.tokens_pos_tags:
            raise ValueError("tokens must be POS-tagged before this operation")

        if len(self.tokens) != len(self.tokens_pos_tags):
            raise ValueError("number of documents in tokens and number of documents in POS tags is not equal")

        for dl, dt in self.tokens.items():
            tags = self.tokens_pos_tags[dl]
            if len(dt) != len(tags):
                raise ValueError("number of tokens is not equal to number of POS tags in the same document ('%s')" % dl)
