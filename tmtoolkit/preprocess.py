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
    apply_to_mat_column, pos_tag_convert_penn_to_wn, simplified_pos, flatten_list, tuplize


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

    def __init__(self, docs, language='english', stopwords=None, punctuation=None, special_chars=None):
        require_dictlike(docs)

        self.docs = docs           # input documents as dict with document label -> document text
        self.language = language   # document language

        if stopwords is None:      # load default stopword list for this language
            self.stopwords = nltk.corpus.stopwords.words(language)
        else:                      # set passed stopword list
            self.stopwords = stopwords

        if punctuation is None:    # load default punctuation list
            self.punctuation = list(string.punctuation)
        else:
            self.punctuation = punctuation

        if special_chars is None:
            self.special_chars = list(string.punctuation)
        else:
            self.special_chars = special_chars

        # set a tokenizer for this language
        self.tokenizer = None      # self.tokenizer is a function with a document text as argument
        self.load_tokenizer(lambda x: nltk.tokenize.word_tokenize(x, self.language))

        self.stemmer = None                # stemmer instance (must have a callable attribute `stem`)
        self.lemmata_dict = None           # lemmata dictionary with POS -> word -> lemma mapping
        self.pos_tagged = False
        self.pos_tagger = None             # POS tagger instance (must have a callable attribute `tag`)
        self.pos_tagset = None             # tagset used for POS tagging

        self._tokens = {}             # tokens at the current processing stage. dict with document label -> tokens list
        self._ngrams = {}             # generated ngrams

        self.ngrams_as_tokens = False

        self.pattern_module = None          # dynamically loaded CLiPS pattern library module
        self.germalemma = None              # GermaLemma instance
        self.wordnet_lemmatizer = None      # nltk.stem.WordNetLemmatizer instance

    @property
    def tokens(self):
        self._require_tokens()

        return {dl: list(zip(*dt))[0] for dl, dt in self._tokens.items()}

    @property
    def tokens_with_pos_tags(self):
        self._require_pos_tags()
        self._require_no_ngrams_as_tokens()

        return self._tokens

    @property
    def ngrams(self):
        return self._ngrams

    def add_stopwords(self, stopwords):
        require_listlike(stopwords)
        self.stopwords += stopwords

        return self

    def add_punctuation(self, punctuation):
        require_listlike(punctuation)
        self.punctuation += punctuation

        return self

    def add_special_chars(self, special_chars):
        require_listlike(special_chars)
        self.special_chars += special_chars

        return self

    def load_tokenizer(self, custom_tokenizer):
        self.tokenizer = custom_tokenizer

        return self

    def load_stemmer(self, custom_stemmer=None):
        if custom_stemmer:
            self.stemmer = custom_stemmer
        else:
            self.stemmer = nltk.stem.SnowballStemmer(self.language)

        return self

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

        return self

    def load_pos_tagger(self, custom_pos_tagger=None, custom_pos_tagset=None):
        if custom_pos_tagger:
            self.pos_tagger = custom_pos_tagger
            self.pos_tagset = custom_pos_tagset
        else:
            picklefile = os.path.join(self.DATAPATH, self.language, self.POS_TAGGER_PICKLE)
            self.pos_tagger = unpickle_file(picklefile)
            if self.language == 'german':
                self.pos_tagset = 'stts'

        return self

    def tokenize(self):
        if not callable(self.tokenizer):
            raise ValueError('tokenizer must be callable')

        self._tokens = {dl: tuplize(self.tokenizer(txt)) for dl, txt in self.docs.items()}

        return self

    def generate_ngrams(self, n, join=True, join_str=' ', reassign_tokens=False):
        self._ngrams = {dl: create_ngrams(list(zip(*dt))[0], n=n, join=join, join_str=join_str)
                        for dl, dt in self._tokens.items()}

        if reassign_tokens:
            self.use_ngrams_as_tokens(join=False)

        return self

    def use_ngrams_as_tokens(self, join=False, join_str=' '):
        self._require_ngrams()

        if join:
            new_tok = {dl: tuplize([join_str.join(g_tuple) for g_tuple in dg])
                       for dl, dg in self._ngrams.items()}
        else:
            new_tok = {dl: tuplize(dg) for dl, dg in self._ngrams.items()}

        self._tokens = new_tok
        self.pos_tagged = False
        self.ngrams_as_tokens = True

        return self

    def transform_tokens(self, transform_fn):
        if not callable(transform_fn):
            raise ValueError('transform_fn must be callable')

        self._require_tokens()

        self._tokens = {dl: apply_to_mat_column(dt, 0, transform_fn)
                        for dl, dt in self._tokens.items()}

        return self

    def tokens_to_lowercase(self):
        self.transform_tokens(string.lower if sys.version_info[0] < 3 else str.lower)

        return self

    def stem(self):
        self._require_tokens()
        self._require_no_ngrams_as_tokens()

        if not self.stemmer:
            self.load_stemmer()

        stemmer = self.stemmer
        if not hasattr(stemmer, 'stem') or not callable(stemmer.stem):
            raise ValueError("stemmer must have a callable attribute `stem`")

        self._tokens = {dl: apply_to_mat_column(dt, 0, lambda t: stemmer.stem(t))
                        for dl, dt in self._tokens.items()}

        return self

    def lemmatize(self, use_dict=False, use_patternlib=False, use_germalemma=None):
        self._require_pos_tags()
        self._require_no_ngrams_as_tokens()

        tmp_lemmata = defaultdict(list)

        if use_germalemma is None and self.language == 'german':
            use_germalemma = True

        if use_germalemma:
            if not self.germalemma:
                lemmata_pickle = os.path.join(self.DATAPATH, self.language, self.LEMMATA_PICKLE)
                lemmata_dict, lemmata_lower_dict = unpickle_file(lemmata_pickle)
                self.germalemma = GermaLemma(lemmata=lemmata_dict, lemmata_lower=lemmata_lower_dict)

            for dl, tok_tags in self._tokens.items():
                for t, pos in tok_tags:
                    try:
                        l = self.germalemma.find_lemma(t, pos)
                    except ValueError:
                        l = t
                    tmp_lemmata[dl].append(l)
        else:
            if use_dict:
                if not self.lemmata_dict:
                    self.load_lemmata_dict()

                for dl, tok_tags in self._tokens.items():
                    for t, pos in tok_tags:
                        pos = simplified_pos(pos, tagset=self.pos_tagset)

                        if pos:
                            l = self.lemmata_dict.get(pos, {}).get(t, None)
                            if l == '-' or l == '':
                                l = None
                        else:
                            l = None
                        tmp_lemmata[dl].append(l)

            if use_patternlib:
                if not self.pattern_module:
                    if self.language not in self.PATTERN_SUBMODULES:
                        raise ValueError("no CLiPS pattern module for this language:", self.language)

                    modname = 'pattern.%s' % self.PATTERN_SUBMODULES[self.language]
                    self.pattern_module = import_module(modname)

                for dl, tok_tags in self._tokens.items():
                    tok_lemmata = tmp_lemmata.get(dl, [None] * len(tok_tags))

                    lemmata_final = []
                    for (t, pos), t_found in zip(tok_tags, tok_lemmata):
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

            for dl, tok_tags in self._tokens.items():
                for t, pos in tok_tags:
                    wn_pos = pos_tag_convert_penn_to_wn(pos)
                    if wn_pos:
                        l = self.wordnet_lemmatizer.lemmatize(t, wn_pos)
                    else:
                        l = t
                    tmp_lemmata[dl].append(l)

        # merge
        lemmatized_tokens = {}
        for dl, tok_tags in self._tokens.items():
            tok_lemmata = tmp_lemmata.get(dl, [None] * len(tok_tags))
            new_tok_tags = [(l or t, pos) for (t, pos), l in zip(tok_tags, tok_lemmata)]
            assert len(new_tok_tags) == len(tok_tags)
            lemmatized_tokens[dl] = new_tok_tags

        assert len(lemmatized_tokens) == len(self._tokens) == len(self.docs)
        self._tokens = lemmatized_tokens

        return self

    def expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        self._require_no_pos_tags()
        self._require_no_ngrams_as_tokens()

        tmp_tokens = {}
        for dl, dt in self._tokens.items():
            nested = [expand_compound_token(tup[0], split_chars, split_on_len, split_on_casechange) for tup in dt]
            tmp_tokens[dl] = tuplize(flatten_list(nested))

        self._tokens = tmp_tokens

        return self

    def remove_special_chars_in_tokens(self):
        self._require_tokens()

        self._tokens = {dl: apply_to_mat_column(dt, 0, lambda x:
                                                       remove_special_chars_in_tokens(x, self.special_chars),
                                                map_func=False)
                        for dl, dt in self._tokens.items()}

        return self

    def clean_tokens(self, remove_punct=True, remove_stopwords=True, remove_empty=True):
        self._require_tokens()

        tokens_to_remove = [''] if remove_empty else []

        if remove_punct:
            tokens_to_remove.extend(self.punctuation)
        if remove_stopwords:
            tokens_to_remove.extend(self.stopwords)

        if tokens_to_remove:
            if type(tokens_to_remove) is not set:
                tokens_to_remove = set(tokens_to_remove)

            self._tokens = {dl: [t for t in dt if t[0] not in tokens_to_remove]
                            for dl, dt in self._tokens.items()}

        return self

    def pos_tag(self):
        """
        Apply Part-of-Speech (POS) tagging on each token.
        Uses the default NLTK tagger if no language-specific tagger could be loaded (English is assumed then as
        language). The default NLTK tagger uses Penn Treebank tagset
        (https://ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).
        The default German tagger based on TIGER corpus uses the STTS tagset
        (http://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/TagSets/stts-table.html).
        """
        self._require_tokens()
        self._require_no_ngrams_as_tokens()

        if not self.pos_tagger:
            try:
                self.load_pos_tagger()
            except IOError:
                self.pos_tagger = GenericPOSTagger
                self.pos_tagset = 'penn'

        tagger = self.pos_tagger
        if not hasattr(tagger, 'tag') or not callable(tagger.tag):
            raise ValueError("pos_tagger must have a callable attribute `tag`")

        self._tokens = {dl: apply_to_mat_column(dt, 0, tagger.tag, map_func=False, expand=True)
                        for dl, dt in self._tokens.items()}

        self.pos_tagged = True

        return self

    def filter_for_token(self, search_token, ignore_case=False, remove_found_token=False):
        self.filter_for_tokenpattern(search_token, fixed=True, ignore_case=ignore_case,
                                     remove_found_token=remove_found_token)

        return self

    def filter_for_tokenpattern(self, tokpattern, fixed=False, ignore_case=False, remove_found_token=False):
        self._require_tokens()

        self._tokens = filter_for_tokenpattern(self._tokens, tokpattern, fixed=fixed, ignore_case=ignore_case,
                                               remove_found_token=remove_found_token)

        return self

    def filter_for_pos(self, required_pos, simplify_pos=True):
        self._require_pos_tags()

        self._tokens = filter_for_pos(self._tokens, required_pos, simplify_pos=simplify_pos,
                                      simplify_pos_tagset=self.pos_tagset)

        return self

    def get_dtm(self, from_ngrams=False):
        self._require_tokens()

        if from_ngrams:
            self._require_ngrams()
            tok = self.ngrams
        else:
            tok = self.tokens

        vocab, doc_labels, docs_terms, dtm_alloc_size = get_vocab_and_terms(tok)
        dtm = create_sparse_dtm(vocab, doc_labels, docs_terms, dtm_alloc_size)

        return doc_labels, vocab, dtm

    def _require_tokens(self):
        if not self._tokens:
            raise ValueError("documents must be tokenized before this operation")

    def _require_pos_tags(self):
        self._require_tokens()

        if not self.pos_tagged:
            raise ValueError("tokens must be POS-tagged before this operation")

    def _require_ngrams(self):
        if not self._ngrams:
            raise ValueError("ngrams must be created before this operation")

    def _require_no_ngrams_as_tokens(self):
        if self.ngrams_as_tokens:
            raise ValueError("ngrams are used as tokens -- this is not possible for this operation")


    def _require_no_pos_tags(self):
        self._require_tokens()

        if self.pos_tagged:
            raise ValueError("tokens shall not be POS-tagged for this operation")


def str_multisplit(s, split_chars):
    parts = [s]
    for c in split_chars:
        parts_ = []
        for p in parts:
            parts_.extend(p.split(c))
        parts = parts_

    return parts


def expand_compound_token(t, split_chars=('-',), split_on_len=2, split_on_casechange=False):
    if not split_on_len and not split_on_casechange:
        raise ValueError('At least one of the arguments `split_on_len` and `split_on_casechange` must evaluate to True')

    if not any(isinstance(split_chars, type_) for type_ in (list, set, tuple)):
        split_chars = [split_chars]

    parts = []
    add = False   # signals if current part should be appended to previous part

    t_parts = [t]
    for c in split_chars:
        t_parts_ = []
        for t in t_parts:
            t_parts_.extend(t.split(c))
        t_parts = t_parts_

    for p in str_multisplit(t, split_chars):  # for each part p in compound token t
        if not p: continue  # skip empty part
        if add and parts:   # append current part p to previous part
            parts[-1] += p
        else:               # add p as separate token
            parts.append(p)

        if split_on_len:
            add = len(p) < split_on_len   # if p only consists of `split_on_len` characters -> append the next p to it

        if split_on_casechange:
            # alt. strategy: if p is all uppercase ("US", "E", etc.) -> append the next p to it
            add = add and p.isupper() if split_on_len else p.isupper()

    if add and len(parts) >= 2:
        parts = parts[:-2] + [parts[-2] + parts[-1]]

    return parts


def remove_special_chars_in_tokens(tokens, special_chars):
    if not special_chars:
        raise ValueError('`special_chars` must be a non-empty sequence')

    special_chars_str = u''.join(special_chars)

    if 'maketrans' in dir(string):  # python 2
        del_chars = {ord(c): None for c in special_chars}
        return [t.translate(del_chars) for t in tokens]
    elif 'maketrans' in dir(str):  # python 3
        del_chars = str.maketrans('', '', special_chars_str)
        return [t.translate(del_chars) for t in tokens]
    else:
        raise RuntimeError('no maketrans() function found')


def create_ngrams(tokens, n, join=True, join_str=' '):
    if n < 2:
        raise ValueError('`n` must be at least 2')

    if len(tokens) < n:
        # raise ValueError('`len(tokens)` should not be smaller than `n`')
        ngrams = [tokens]
    else:
        ngrams = [[tokens[i+j] for j in range(n)]
                  for i in range(len(tokens)-n+1)]
    if join:
        return list(map(lambda x: join_str.join(x), ngrams))
    else:
        return ngrams
