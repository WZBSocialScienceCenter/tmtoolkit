"""
Preprocessing worker class for parallel text processing.
"""

import multiprocessing as mp
from importlib import import_module
from copy import deepcopy
from functools import partial
import re

import numpy as np
import pandas as pd
import nltk
from germalemma import GermaLemma

from .. import logger
from ..filter_tokens import filter_for_token, filter_for_pos
from ..utils import pos_tag_convert_penn_to_wn, flatten_list
from .utils import expand_compound_token, remove_chars_in_tokens, create_ngrams, tokens2ids, ids2tokens
from ._common import PATTERN_SUBMODULES


pttrn_metadata_key = re.compile(r'^meta_(.+)$')


class PreprocWorker(mp.Process):
    def __init__(self, worker_id, docs, language, tasks_queue, results_queue, tokenizer, stemmer, pos_tagger,
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

    def _get_available_metadata_keys(self):
        cols = np.unique(np.concatenate([df.columns for df in self._tokens.values()]))
        keys = []
        for c in cols:
            m = pttrn_metadata_key.search(c)
            if m:
                keys.append(m.group(1))

        return keys

    def _task_get_tokens(self):
        tokids = [tokendf.token for tokendf in self._tokens.values()]
        tokens = self._ids2tokens(tokids)

        result = {}
        for i, (dl, df) in enumerate(self._tokens.items()):
            result[dl] = pd.concat((pd.DataFrame({'token': tokens[i]}),
                                    df.loc[:, df.columns.difference(['token'])]),
                                   axis=1)

        self._put_items_in_results_queue(result)

    def _task_get_available_metadata_keys(self):
        self.results_queue.put(self._get_available_metadata_keys())

    def _task_get_vocab(self):
        self.results_queue.put(self._vocab)

    def _task_replace_vocab(self, vocab):
        if len(self._vocab) != len(vocab):
            raise ValueError('cannot replace vocabulary with differing length')

        self._task_set_vocab(vocab)

    def _task_set_vocab(self, vocab):
        if not isinstance(vocab, np.ndarray):
            vocab = np.array(vocab)

        self._vocab = vocab

    def _task_get_vocab_doc_freq(self):
        assert len(self._vocab) == len(self._vocab_counts)
        self.results_queue.put(dict(zip(self._vocab, self._vocab_counts)))

    # def _task_get_tokens_with_worker_id(self):
    #     self.results_queue.put((self.worker_id, self._tokens))

    def _task_get_ngrams(self):
        ngrams = self._ids2tokens_for_ngrams(self._ngrams)
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

    def _task_add_metadata_per_token(self, key, data, default, dtype):
        logger.debug('worker `%s`: adding metadata per token' % self.name)

        searchtokids = self._searchtokens2ids(list(data.keys()))
        mapper = dict(zip(searchtokids, data.values()))

        def replace(val):
            return mapper.get(val, default)
        replace = np.vectorize(replace)

        col = 'meta_' + key
        for df in self._tokens.values():
            df[col] = pd.Series(replace(df.token), dtype=dtype)

    def _task_add_metadata_per_doc(self, key, data, dtype):
        logger.debug('worker `%s`: adding metadata per document' % self.name)

        col = 'meta_' + key
        for dl, meta in data.items():
            if isinstance(meta, pd.Series):
                meta_ser = meta
            else:
                meta_ser = pd.Series(meta, dtype=dtype)

            self._tokens[dl][col] = meta_ser

    def _task_remove_metadata(self, key):
        logger.debug('worker `%s`: removing metadata column' % self.name)

        col = 'meta_' + key

        for df in self._tokens.values():
            if col in df.columns:
                del df[col]

    def _task_tokenize(self):
        self._create_token_ids_and_vocab([self.tokenizer.tokenize(txt) for txt in self.docs.values()])

    def _task_generate_ngrams(self, n):
        self._ngrams = {dl: create_ngrams(dt.token, n=n, join=False)
                         for dl, dt in self._tokens.items()}

    def _task_use_joined_ngrams_as_tokens(self, join_str):
        ngrams_docs = self._ids2tokens_for_ngrams(self._ngrams)

        ngrams_joined = [np.apply_along_axis(lambda ngram: join_str.join(ngram), 1, ngrams)
                         for ngrams in ngrams_docs.values()]

        self._create_token_ids_and_vocab(ngrams_joined)

    def _task_transform_tokens(self, transform_fn, vectorize):
        if vectorize:
            transform_fn = np.vectorize(transform_fn)

        self._vocab = transform_fn(self._vocab)

    def _task_tokens_to_lowercase(self):
        self._vocab = np.char.lower(self._vocab)

    def _task_stem(self):
        self._task_transform_tokens(self.stemmer.stem, vectorize=True)

    def _task_pos_tag(self):
        for df in self._tokens.values():
            tokens_and_tags = self.pos_tagger.tag(self._ids2tokens(df.token))
            pos_tags = list(zip(*tokens_and_tags))[1]

            df['meta_pos'] = pd.Series(pos_tags, dtype='category')

    def _task_lemmatize(self, pos_tagset):
        if self.language == 'english':
            if not self.wordnet_lemmatizer:
                self.wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

            lemmatize_fn = self.wordnet_lemmatizer.lemmatize

            def lemmatize_wrapper(row, lemmatize_fn):
                _, pos, tok = row
                wn_pos = pos_tag_convert_penn_to_wn(pos)
                if wn_pos:
                    return lemmatize_fn(tok, wn_pos)
                else:
                    return tok
        elif self.language == 'german':
            if not self.germalemma:
                self.germalemma = GermaLemma()

            lemmatize_fn = self.germalemma.find_lemma

            def lemmatize_wrapper(row, lemmatize_fn):
                _, pos, tok = row
                try:
                    return lemmatize_fn(tok, pos)
                except ValueError:
                    return tok
        else:
            if not self.pattern_module:
                if self.language not in PATTERN_SUBMODULES:
                    raise ValueError("no CLiPS pattern module for this language:", self.language)

                modname = 'pattern.%s' % PATTERN_SUBMODULES[self.language]
                self.pattern_module = import_module(modname)

            lemmatize_fn = self.pattern_module

            def lemmatize_wrapper(row, lemmatize_fn):
                _, pos, tok = row
                if pos.startswith('NP'):  # singularize noun
                    return lemmatize_fn.singularize(tok)
                elif pos.startswith('V'):  # get infinitive of verb
                    return lemmatize_fn.conjugate(tok, lemmatize_fn.INFINITIVE)
                elif pos.startswith('ADJ') or pos.startswith('ADV'):  # get baseform of adjective or adverb
                    return lemmatize_fn.predicative(tok)
                return tok

        tokens_pos = pd.concat(list(self._tokens.values()), ignore_index=True)[['token', 'meta_pos']].drop_duplicates()
        tokens_pos['token_word'] = self._ids2tokens([tokens_pos.token])[0]
        tokens_pos['lemma'] = tokens_pos.apply(lemmatize_wrapper, axis=1, lemmatize_fn=lemmatize_fn)

        new_tokens_dfs = []
        for doc_tok in self._tokens.values():
            new_doc_tok = pd.merge(doc_tok, tokens_pos[['token', 'meta_pos', 'lemma']],
                                   how='left', on=('token', 'meta_pos'))
            new_doc_tok['token'] = new_doc_tok['lemma']
            del new_doc_tok['lemma']

            new_tokens_dfs.append(new_doc_tok)

        self._vocab, tokids, self._vocab_counts = tokens2ids([df.token for df in new_tokens_dfs], return_counts=True)
        for df, tok in zip(new_tokens_dfs, tokids):
            df['token'] = tok
        self._tokens = dict(zip(self._tokens.keys(), new_tokens_dfs))

    def _task_expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        """
        Note: This function will reset the token dataframe `self._tokens` to the newly created tokens. This means
        all token metadata will be gone.
        """
        tokens = self._ids2tokens([df.token for df in self._tokens.values()])

        new_tokens = []
        for tok in tokens:
            nested = [expand_compound_token(t, split_chars, split_on_len, split_on_casechange) for t in tok]
            new_tokens.append(flatten_list(nested))

        self._create_token_ids_and_vocab(new_tokens)

    def _task_remove_chars_in_tokens(self, chars):
        self._vocab = np.array(remove_chars_in_tokens(self._vocab, chars=chars))

    def _task_clean_tokens(self, tokens_to_remove, remove_shorter_than=None, remove_longer_than=None, remove_numbers=False):
        remove_mask = np.repeat(False, len(self._vocab))

        if remove_shorter_than is not None:
            remove_mask |= np.char.str_len(self._vocab) < remove_shorter_than

        if remove_longer_than is not None:
            remove_mask |= np.char.str_len(self._vocab) > remove_longer_than

        if remove_numbers:
            remove_mask |= np.char.isnumeric(self._vocab)

        remove_tok_indices = np.where(self._vocab[:, None] == list(tokens_to_remove))[0]
        remove_mask[remove_tok_indices] = True

        vocab_ids = np.arange(len(self._vocab))
        self._vocab = self._vocab[~remove_mask]
        vocab_ids_masked = vocab_ids[~remove_mask]

        mapper = dict(zip(vocab_ids_masked, range(len(vocab_ids_masked))))

        def replace(val):
            return mapper[val]
        replace = np.vectorize(replace)

        tmp_tokens = {}
        for dl, df in self._tokens.items():
            filter_indices = np.where(df.token[:, None] == vocab_ids_masked)[0]
            df = df.iloc[filter_indices, :].copy()
            df['token'] = replace(df.token)

            tmp_tokens[dl] = df.reset_index(drop=True)
            print(tmp_tokens[dl])

        self._tokens = tmp_tokens

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

    def _create_token_ids_and_vocab(self, tokens):
        self._vocab, tokids, self._vocab_counts = tokens2ids(tokens, return_counts=True)
        self._tokens = dict(zip(self.docs.keys(), list(map(lambda x: pd.DataFrame({'token': x}), tokids))))

    def _ids2tokens(self, tokids):
        return ids2tokens(self._vocab, tokids)

    def _searchtokens2ids(self, searchtokens):
        if not isinstance(searchtokens, np.ndarray):
            searchtokens = np.array(searchtokens)

        return np.where(searchtokens[:, None] == self._vocab)[1]

    def _ids2tokens_for_ngrams(self, docs_ngrams):
        return {dl: np.array(list(map(lambda ngram: ids2tokens(self._vocab, ngram), ngrams)))
                for dl, ngrams in docs_ngrams.items()}
