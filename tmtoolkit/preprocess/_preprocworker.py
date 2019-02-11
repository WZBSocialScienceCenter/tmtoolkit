"""
Preprocessing worker class for parallel text processing.
"""

import multiprocessing as mp
from importlib import import_module
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import nltk
from germalemma import GermaLemma

from .. import logger
from ..filter_tokens import filter_for_token, filter_for_pos
from ..utils import apply_to_mat_column, pos_tag_convert_penn_to_wn, simplified_pos, \
    flatten_list, tuplize, ith_column
from .utils import expand_compound_token, remove_special_chars_in_tokens, create_ngrams, tokens2ids, ids2tokens
from ._common import PATTERN_SUBMODULES


class PreprocWorker(mp.Process):
    def __init__(self, worker_id, docs, language, tasks_queue, results_queue, tokenizer, stemmer, lemmata_dict, pos_tagger,
                 group=None, target=None, name=None, args=(), kwargs=None):
        super().__init__(group, target, name, args, kwargs or {})
        logger.debug('worker `%s`: init with worker ID %d' % (name, worker_id))
        logger.debug('worker `%s`: docs = %s' % (name, str(set(docs.keys()))))
        self.worker_id = worker_id
        self.docs = docs
        self.language = language
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue

        # set a tokenizer
        self.tokenizer = tokenizer      # tokenizer instance (must have a callable attribute `tokenize` with a document
                                        # text as argument)

        # set a stemmer
        self.stemmer = stemmer                # stemmer instance (must have a callable attribute `stem`)

        # set a POS tagger
        self.pos_tagger = pos_tagger             # POS tagger instance (must have a callable attribute `tag`)

        self.lemmata_dict = lemmata_dict
        self.pattern_module = None          # dynamically loaded CLiPS pattern library module
        self.germalemma = None              # GermaLemma instance
        self.wordnet_lemmatizer = None      # nltk.stem.WordNetLemmatizer instance

        self._vocab = None
        self._vocab_counts = None
        self._tokens = {}             # tokens for this worker at the current processing stage.
                                      # dict with document label -> data frame
        self._ngrams = {}             # generated ngrams

        #self._filtered = False
        self._orig_tokens = None      # original (unfiltered) tokens, when filtering is currently applied

    def run(self):
        logger.debug('worker `%s`: run' % self.name)

        for next_task, task_kwargs in iter(self.tasks_queue.get, None):
            logger.debug('worker `%s`: received task `%s`' % (self.name, next_task))

            exec_task_fn = getattr(self, '_task_' + next_task)
            if exec_task_fn:
                exec_task_fn(**task_kwargs)
            else:
                raise NotImplementedError("Task not implemented: `%s`" % next_task)

            self.tasks_queue.task_done()

        logger.debug('worker `%s`: shutting down' % self.name)
        self.tasks_queue.task_done()

    def _put_items_in_results_queue(self, container):
        if container:
            logger.debug('worker `%s`: putting %d results in queue' % (self.name, len(container)))
            for pair in container.items():
                self.results_queue.put(pair)
        else:
            # we *have* to put something in the result queue -> signal that we return "nothing"
            logger.debug('worker `%s`: putting None in results queue' % self.name)
            self.results_queue.put(None)

    def _task_get_tokens(self):
        tokids = [tokendf.token for tokendf in self._tokens.values()]
        tokens = ids2tokens(self._vocab, tokids)

        result = {}
        for i, (dl, df) in enumerate(self._tokens.items()):
            result[dl] = pd.concat((pd.DataFrame({'token': tokens[i]}),
                                    df.loc[:, df.columns.difference(['token'])]),
                                   axis=1)

        self._put_items_in_results_queue(result)

    def _task_get_vocab(self):
        self.results_queue.put(self._vocab)

    def _task_get_vocab_doc_freq(self):
        assert len(self._vocab) == len(self._vocab_counts)
        self.results_queue.put(dict(zip(self._vocab, self._vocab_counts)))

    # def _task_get_tokens_with_worker_id(self):
    #     self.results_queue.put((self.worker_id, self._tokens))

    def _task_get_ngrams(self):
        ngrams = _ids2tokens_for_ngrams(self._ngrams, self._vocab)
        self._put_items_in_results_queue(ngrams)

    # def _task_get_ngrams_with_worker_id(self):
    #    self.results_queue.put((self.worker_id, self._ngrams))

    def _task_get_state(self):
        logger.debug('worker `%s`: getting state' % self.name)

        state_attrs = (
            'docs',
            'language',
            '_tokens',
            '_ngrams',
            '_orig_tokens'
        )

        state = {attr: getattr(self, attr) for attr in state_attrs}
        logger.debug('worker `%s`: got state with %d items' % (self.name, len(state)))
        self.results_queue.put(state)

    def _task_set_tokens(self, tokens):
        logger.debug('worker `%s`: setting tokens' % self.name)
        self._tokens = tokens

    def _task_set_ngrams(self, ngrams):
        logger.debug('worker `%s`: setting ngrams' % self.name)
        self._ngrams = ngrams

    def _task_set_state(self, **state):
        logger.debug('worker `%s`: setting state' % self.name)

        for attr, val in state.items():
            setattr(self, attr, val)

    def _task_tokenize(self):
        tok = [self.tokenizer.tokenize(txt) for txt in self.docs.values()]
        self._vocab, tokids, self._vocab_counts = tokens2ids(tok, return_counts=True)
        self._tokens = dict(zip(self.docs.keys(), list(map(lambda x: pd.DataFrame({'token': x}), tokids))))

    def _task_generate_ngrams(self, n):
        self._ngrams = {dl: create_ngrams(dt.token, n=n, join=False)
                         for dl, dt in self._tokens.items()}

    def _task_use_joined_ngrams_as_tokens(self, join_str):
        ngrams_docs = _ids2tokens_for_ngrams(self._ngrams, self._vocab)

        ngrams_joined = [np.apply_along_axis(lambda ngram: join_str.join(ngram), 1, ngrams)
                         for ngrams in ngrams_docs.values()]

        self._vocab, tokids, self._vocab_counts = tokens2ids(ngrams_joined, return_counts=True)
        self._tokens = dict(zip(self.docs.keys(), list(map(lambda x: pd.DataFrame({'token': x}), tokids))))

    def _task_transform_tokens(self, transform_fn):
        self._tokens = {dl: apply_to_mat_column(dt, 0, transform_fn) if dt else []
                        for dl, dt in self._tokens.items()}

    def _task_stem(self):
        self._tokens = {dl: apply_to_mat_column(dt, 0, lambda t: self.stemmer.stem(t)) if dt else []
                        for dl, dt in self._tokens.items()}

    def _task_pos_tag(self):
        for df in self._tokens.values():
            df['meta_pos'] = pd.Series(self.pos_tagger.tag(ids2tokens(self._vocab, df.token)), dtype='category')

    def _task_lemmatize(self, pos_tagset, use_dict=False, use_patternlib=False, use_germalemma=None):
        tmp_lemmata = defaultdict(list)

        if use_germalemma is None and self.language == 'german':
            use_germalemma = True

        if use_germalemma:
            if not self.germalemma:
                self.germalemma = GermaLemma()

            for dl, tok_tags in self._tokens.items():
                for t, pos in tok_tags:
                    try:
                        l = self.germalemma.find_lemma(t, pos)
                    except ValueError:
                        l = t
                    tmp_lemmata[dl].append(l)
        else:
            if use_dict and self.lemmata_dict:
                for dl, tok_tags in self._tokens.items():
                    for t, pos in tok_tags:
                        pos = simplified_pos(pos, tagset=pos_tagset)

                        if pos:
                            l = self.lemmata_dict.get(pos, {}).get(t, None)
                            if l == '-' or l == '':
                                l = None
                        else:
                            l = None
                        tmp_lemmata[dl].append(l)

            if use_patternlib:
                if not self.pattern_module:
                    if self.language not in PATTERN_SUBMODULES:
                        raise ValueError("no CLiPS pattern module for this language:", self.language)

                    modname = 'pattern.%s' % PATTERN_SUBMODULES[self.language]
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

        assert len(lemmatized_tokens) == len(self._tokens)
        self._tokens = lemmatized_tokens

    def _task_expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        tmp_tokens = {}
        for dl, dt in self._tokens.items():
            nested = [expand_compound_token(tup[0], split_chars, split_on_len, split_on_casechange) for tup in dt]
            tmp_tokens[dl] = tuplize(flatten_list(nested))

        self._tokens = tmp_tokens

    def _task_remove_special_chars_in_tokens(self, special_chars):
        self._tokens = {dl: apply_to_mat_column(dt, 0, lambda x: remove_special_chars_in_tokens(x, special_chars),
                                                map_func=False) if dt else []
                        for dl, dt in self._tokens.items()}

    def _task_clean_tokens(self, tokens_to_remove, save_orig_tokens=False, remove_shorter_than=None,
                           remove_longer_than=None, remove_numbers=False):
        if save_orig_tokens:
            self._save_orig_tokens()

        if remove_shorter_than is not None:
            self._tokens = {dl: [t for t in dt if len(t[0]) >= remove_shorter_than]
                            for dl, dt in self._tokens.items()}

        if remove_longer_than is not None:
            self._tokens = {dl: [t for t in dt if len(t[0]) <= remove_longer_than]
                            for dl, dt in self._tokens.items()}

        if remove_numbers:
            self._tokens = {dl: [t for t in dt if not t[0].isnumeric()]
                            for dl, dt in self._tokens.items()}

        if type(tokens_to_remove) is not set:   # using a set is much faster than other sequence types for "in" tests
            tokens_to_remove = set(tokens_to_remove)

        self._tokens = {dl: [t for t in dt if t[0] not in tokens_to_remove]
                        for dl, dt in self._tokens.items()}

    def _task_filter_for_token(self, search_token, match_type='exact', ignore_case=False, glob_method='match',
                               remove_found_token=False):
        self._save_orig_tokens()

        self._tokens = filter_for_token(self._tokens, search_token, match_type=match_type, ignore_case=ignore_case,
                                        glob_method=glob_method, remove_found_token=remove_found_token,
                                        remove_empty_docs=False)

    def _task_filter_for_pos(self, required_pos, pos_tagset, simplify_pos=True):
        self._save_orig_tokens()
        self._tokens = filter_for_pos(self._tokens, required_pos,
                                      simplify_pos=simplify_pos,
                                      simplify_pos_tagset=pos_tagset)

    def _task_reset_filter(self):
        self._tokens = self._orig_tokens
        self._orig_tokens = None

    def _save_orig_tokens(self):
        if self._orig_tokens is None:   # initial filtering -> safe a copy of the original tokens
            self._orig_tokens = deepcopy(self._tokens)


def _ids2tokens_for_ngrams(docs_ngrams, vocab):
    return {dl: np.array(list(map(lambda ngram: ids2tokens(vocab, ngram), ngrams)))
            for dl, ngrams in docs_ngrams.items()}
