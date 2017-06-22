# -*- coding: utf-8 -*-
import sys
import os
import string
from importlib import import_module
from collections import defaultdict

import nltk

from .germalemma import GermaLemma
from .filter_tokens import filter_for_tokenpattern, filter_for_pos
from .dtm import create_sparse_dtm, get_vocab_and_terms
from .utils import require_listlike, require_dictlike, unpickle_file, \
    pos_tag_convert_penn_to_wn, simplified_pos, filter_elements_in_dict


class GenericPOSTagger(object):
    @staticmethod
    def tag(tokens):
        return nltk.pos_tag(tokens)


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

        self.docs = docs           # input documents as dict with document label -> document text
        self.language = language   # document language

        if stopwords is None:      # load default stopword list for this language
            self.stopwords = nltk.corpus.stopwords.words(language)
        else:                      # set passed stopword list
            self.stopwords = stopwords

        if punctuation is None:    # load default punctuation list
            self.punctuation = list(string.punctuation)

        # set a tokenizer for this language
        self.tokenizer = None      # self.tokenizer is a function with a document text as argument
        self.load_tokenizer(lambda x: nltk.tokenize.word_tokenize(x, self.language))

        self.stemmer = None                # stemmer instance (must have a callable attribute `stem`)
        self.lemmata_dict = None           # lemmata dictionary with POS -> word -> lemma mapping
        self.pos_tagger = None             # POS tagger instance (must have a callable attribute `tag`)
        self.pos_tagset = None             # tagset used for POS tagging

        self.tokens = None             # tokens at the current processing stage. dict with document label -> tokens list
        self.tokens_pos_tags = None    # POS tags for self.tokens. dict with document label -> POS tag list

        self.pattern_module = None          # dynamically loaded CLiPS pattern library module
        self.germalemma = None              # GermaLemma instance
        self.wordnet_lemmatizer = None      # nltk.stem.WordNetLemmatizer instance

    @property
    def tokens_with_pos_tags(self):
        self._require_tokens()
        self._require_pos_tags()

        return {dl: list(zip(self.tokens[dl], self.tokens_pos_tags[dl])) for dl in self.tokens.keys()}

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

    def load_pos_tagger(self, custom_pos_tagger=None, custom_pos_tagset=None):
        if custom_pos_tagger:
            self.pos_tagger = custom_pos_tagger
            self.pos_tagset = custom_pos_tagset
        else:
            picklefile = os.path.join(self.DATAPATH, self.language, self.POS_TAGGER_PICKLE)
            self.pos_tagger = unpickle_file(picklefile)
            if self.language == 'german':
                self.pos_tagset = 'stts'

    def tokenize(self):
        if not callable(self.tokenizer):
            raise ValueError('tokenizer must be callable')

        self.tokens = {dl: self.tokenizer(txt) for dl, txt in self.docs.items()}

        return self.tokens

    def token_transform(self, transform_fn):
        if not callable(transform_fn):
            raise ValueError('transform_fn must be callable')

        self._require_tokens()

        self.tokens = {dl: list(map(transform_fn, dt)) for dl, dt in self.tokens.items()}

        return self.tokens

    def tokens_to_lowercase(self):
        return self.token_transform(string.lower if sys.version_info[0] < 3 else str.lower)

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

    def lemmatize(self, use_dict=False, use_patternlib=False, use_germalemma=None):
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
                        pos = simplified_pos(pos, tagset=self.pos_tagset)

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

        if len(tmp_lemmata) == 0:
            if not self.wordnet_lemmatizer:
                self.wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

            for dl, dt in self.tokens.items():
                tok_tags = self.tokens_pos_tags[dl]
                for t, pos in zip(dt, tok_tags):
                    wn_pos = pos_tag_convert_penn_to_wn(pos)
                    if wn_pos:
                        l = self.wordnet_lemmatizer.lemmatize(t, wn_pos)
                    else:
                        l = t
                    tmp_lemmata[dl].append(l)

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
            if type(tokens_to_remove) is not set:
                tokens_to_remove = set(tokens_to_remove)

            matches = {}
            for dl, dt in self.tokens.items():
                matches[dl] = [t not in tokens_to_remove for t in dt]
            self.tokens = filter_elements_in_dict(self.tokens, matches)
            if self.tokens_pos_tags:
                self.tokens_pos_tags = filter_elements_in_dict(self.tokens_pos_tags, matches)

        return self.tokens

    def pos_tag(self):
        """
        Apply Part-of-Speech (POS) tagging on each token. Save the results in `self.tokens_pos_tags` and also return
        them.
        Uses the default NLTK tagger if no language-specific tagger could be loaded (English is assumed then as
        language). The default NLTK tagger uses Penn Treebank tagset
        (https://ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).
        The default German tagger based on TIGER corpus uses the STTS tagset
        (http://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/TagSets/stts-table.html).
        """
        self._require_tokens()

        if not self.pos_tagger:
            try:
                self.load_pos_tagger()
            except IOError:
                self.pos_tagger = GenericPOSTagger
                self.pos_tagset = 'penn'

        tagger = self.pos_tagger
        if not hasattr(tagger, 'tag') or not callable(tagger.tag):
            raise ValueError("pos_tagger must have a callable attribute `tag`")

        self.tokens_pos_tags = {dl: list(zip(*tagger.tag(dt)))[1]
                                for dl, dt in self.tokens.items()}

        return self.tokens_pos_tags

    def filter_for_token(self, search_token, ignore_case=False, remove_found_token=False):
        return self.filter_for_tokenpattern(search_token, fixed=True, ignore_case=ignore_case,
                                            remove_found_token=remove_found_token)

    def filter_for_tokenpattern(self, tokpattern, fixed=False, ignore_case=False, remove_found_token=False):
        self._require_tokens()

        res = filter_for_tokenpattern(self.tokens, tokpattern, fixed=fixed, ignore_case=ignore_case,
                                      remove_found_token=remove_found_token, return_matches=remove_found_token)
        if type(res) is tuple:
            self.tokens, matches = res
        else:
            self.tokens = res
            matches = None

        if self.tokens_pos_tags:
            del_docs = set(self.tokens_pos_tags.keys()) - set(self.tokens.keys())
            for dl in del_docs:
                del self.tokens_pos_tags[dl]

            if matches:
                self.tokens_pos_tags = filter_elements_in_dict(self.tokens_pos_tags, matches, negate_matches=True)

        return self.tokens

    def filter_for_pos(self, required_pos, simplify_pos=True):
        self._require_tokens()
        self._require_pos_tags()

        self.tokens, matches = filter_for_pos(self.tokens, self.tokens_pos_tags, required_pos,
                                              simplify_pos=simplify_pos, simplify_pos_tagset=self.pos_tagset,
                                              return_matches=True)
        self.tokens_pos_tags = filter_elements_in_dict(self.tokens_pos_tags, matches)

        return self.tokens

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
