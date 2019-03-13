"""
Preprocessing worker class for parallel text processing.
"""

import multiprocessing as mp
from importlib import import_module
import re

import numpy as np
import pandas as pd
import nltk
from germalemma import GermaLemma

from .. import logger
from ..utils import pos_tag_convert_penn_to_wn, flatten_list, simplified_pos, token_match, \
    expand_compound_token, remove_chars_in_tokens, create_ngrams, ids2tokens, empty_chararray, tokens2ids, \
    make_vocab_unique_and_update_token_ids
from ._common import PATTERN_SUBMODULES


pttrn_metadata_key = re.compile(r'^meta_(.+)$')


class PreprocWorker(mp.Process):
    def __init__(self, worker_id, docs, language, tasks_queue, results_queue, tokenizer, stemmer, pos_tagger,
                 docs_are_tokenized=False, group=None, target=None, name=None, args=(), kwargs=None):
        super().__init__(group, target, name, args, kwargs or {})
        logger.debug('worker `%s`: init with worker ID %d' % (name, worker_id))
        if docs is not None:
            logger.debug('worker `%s`: docs = %s' % (name, str(set(docs.keys()))))
        self.worker_id = worker_id
        self.language = language
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue

        # set a tokenizer
        self.tokenizer = tokenizer      # tokenizer instance (must have a callable attribute `tokenize` with a document
                                        # text as argument)

        # set a stemmer
        self.stemmer = stemmer              # stemmer instance (must have a callable attribute `stem`)

        # set a POS tagger
        self.pos_tagger = pos_tagger        # POS tagger instance (must have a callable attribute `tag`)

        self.pattern_module = None          # dynamically loaded CLiPS pattern library module
        self.germalemma = None              # GermaLemma instance
        self.wordnet_lemmatizer = None      # nltk.stem.WordNetLemmatizer instance

        self._vocab = empty_chararray()
        self._vocab_counts = np.array([], dtype=np.int)
        self._tokens = {}             # tokens for this worker at the current processing stage.
                                      # dict with document label -> data frame
        self._ngrams = {}             # generated ngrams

        if docs is not None:
            if docs_are_tokenized:
                # docs are already tokenized, but there token IDs must be created
                has_meta = isinstance(next(iter(docs.values())), pd.DataFrame)
                if has_meta:
                    tokenslist = [df.token for df in docs.values()]
                else:
                    tokenslist = list(docs.values())

                self._create_token_ids_and_vocab(tokenslist, doc_labels=docs.keys())
                if has_meta:
                    self._add_metadata_to_tokens(docs)
            else:
                # directly tokenize documents
                logger.debug('tokenizing %d documents' % len(docs))
                self._create_token_ids_and_vocab([self.tokenizer.tokenize(txt) for txt in docs.values()],
                                                 doc_labels=docs.keys())
        else:  # if not documents are passed, tokens, vocab and vocab_counts must be set via set_state directly
               # after instantiation
            logger.debug('no documents passed')

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

    def _get_available_metadata_keys(self):
        all_cols = [df.columns for df in self._tokens.values()]
        if all_cols:
            cols = np.unique(np.concatenate(all_cols))
        else:
            cols = []
        keys = []
        for c in cols:
            m = pttrn_metadata_key.search(c)
            if m:
                keys.append(m.group(1))

        return keys

    def _task_get_doc_labels(self):
        self.results_queue.put(list(self._tokens.keys()))

    def _task_get_tokens(self):
        tokids = [tokendf.token for tokendf in self._tokens.values()]
        tokens = self._ids2tokens(tokids)

        result = {}
        for i, (dl, df) in enumerate(self._tokens.items()):
            result[dl] = pd.concat((pd.DataFrame({'token': tokens[i]}),
                                    df.loc[:, df.columns.difference(['token'])]),
                                   axis=1)

        self.results_queue.put(result)

    def _task_get_available_metadata_keys(self):
        self.results_queue.put(self._get_available_metadata_keys())

    def _task_get_vocab(self, with_worker_id=False):
        if with_worker_id:
            self.results_queue.put((self.worker_id, self._vocab))
        else:
            self.results_queue.put(self._vocab)

    def _task_set_vocab(self, vocab):
        if not isinstance(vocab, np.ndarray):
            vocab = np.array(vocab)

        self._update_vocab(vocab)

    def _task_get_vocab_counts(self):
        assert len(self._vocab) == len(self._vocab_counts)
        self.results_queue.put(dict(zip(self._vocab, self._vocab_counts)))

    def _task_get_num_unique_tokens_per_doc(self):
        self.results_queue.put({dl: len(dt.token.unique()) for dl, dt in self._tokens.items()})

    def _task_get_ngrams(self):
        self.results_queue.put(self._ids2tokens_for_ngrams(self._ngrams))

    def _task_get_state(self):
        logger.debug('worker `%s`: getting state' % self.name)

        state_attrs = (
            'language',
            '_vocab',
            '_vocab_counts',
            '_tokens',
            '_ngrams',
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

    def _task_generate_ngrams(self, n):
        self._ngrams = {dl: create_ngrams(dt.token, n=n, join=False)
                         for dl, dt in self._tokens.items()}

    def _task_use_joined_ngrams_as_tokens(self, join_str):
        ngrams_docs = self._ids2tokens_for_ngrams(self._ngrams)

        ngrams_joined = [np.array(list(map(lambda g: join_str.join(g), ngrams)))
                         for ngrams in ngrams_docs.values()]

        self._create_token_ids_and_vocab(ngrams_joined)

    def _task_transform_tokens(self, transform_fn, vectorize):
        if vectorize:
            transform_fn = np.vectorize(transform_fn)

        self._update_vocab(transform_fn(self._vocab))

    def _task_tokens_to_lowercase(self):
        self._update_vocab(np.char.lower(self._vocab))

    def _task_stem(self):
        self._task_transform_tokens(self.stemmer.stem, vectorize=True)

    def _task_pos_tag(self):
        for df in self._tokens.values():
            if len(df) > 0:
                tokens_and_tags = self.pos_tagger.tag(self._ids2tokens(df.token))
                pos_tags = list(zip(*tokens_and_tags))[1]
            else:
                pos_tags = []

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

        self._vocab, tokids, self._vocab_counts = tokens2ids([df.token.values for df in new_tokens_dfs], return_counts=True)
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
        self._update_vocab(np.array(remove_chars_in_tokens(self._vocab, chars=chars)))

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

        self._remove_tokens_from_vocab(remove_mask)

    def _task_filter_tokens(self, search_token, match_type, ignore_case, glob_method, inverse):
        matches = token_match(search_token, self._vocab, match_type=match_type, ignore_case=ignore_case,
                              glob_method=glob_method)

        if inverse:
            matches = ~matches

        self._remove_tokens_from_vocab(~matches)

    def _task_filter_documents(self, search_token, match_type, ignore_case, glob_method, inverse):
        matches = token_match(search_token, self._vocab, match_type=match_type, ignore_case=ignore_case,
                              glob_method=glob_method)

        if inverse:
            matches = ~matches

        matches_ids = np.where(matches)[0]

        filtered = {dl: self._ids2tokens(df.token) for dl, df in self._tokens.items()
                    if len(np.where(df.token[:, None] == matches_ids)[0]) > 0}

        self._create_token_ids_and_vocab(list(filtered.values()), filtered.keys())

    def _task_filter_for_pos(self, required_pos, pos_tagset, simplify_pos):
        if required_pos is None or isinstance(required_pos, str):
            required_pos = {required_pos}  # turn it into a set

        if simplify_pos:
            simplify_fn = lambda x: simplified_pos(x, tagset=pos_tagset)
        else:
            simplify_fn = lambda x: x

        filtered_meta = {}
        filtered_toks = []
        for dl, df in self._tokens.items():
            pos_simplified = df.meta_pos.apply(simplify_fn)
            df = df.loc[pos_simplified.isin(required_pos), :].copy().reset_index(drop=True)
            filtered_meta[dl] = df
            filtered_toks.append(self._ids2tokens(df.token))

        self._create_token_ids_and_vocab(filtered_toks)

        # re-add metadata
        self._add_metadata_to_tokens(filtered_meta)

    def _add_metadata_to_tokens(self, meta_dfs):
        """Add all meta data columns from `meta_dfs` to `self._tokens`."""
        tmp_tokens = {}
        for dl, df in self._tokens.items():
            meta_df = meta_dfs[dl]
            meta_cols = list(meta_df.columns.difference(['token']))
            assert len(df) == len(meta_df)
            tmp_tokens[dl] = pd.concat((df, meta_df[meta_cols]), ignore_index=True, axis=1).\
                rename(columns=dict(zip(range(len(meta_cols) + 1), ['token'] + meta_cols)))

        self._tokens = tmp_tokens

    def _update_vocab(self, vocab):
        self._vocab, tokids = make_vocab_unique_and_update_token_ids(vocab, [df.token.values for df in self._tokens.values()])
        token_dfs = list(map(lambda x: pd.DataFrame({'token': x}), tokids))
        old_tokens = self._tokens

        self._tokens = dict(zip(self._tokens.keys(), token_dfs))
        self._add_metadata_to_tokens(old_tokens)

    def _create_token_ids_and_vocab(self, tokens, doc_labels=None):
        if doc_labels is None:
            doc_labels = self._tokens.keys()

        self._vocab, tokids, self._vocab_counts = tokens2ids(tokens, return_counts=True)
        token_dfs = list(map(lambda x: pd.DataFrame({'token': x}), tokids))

        if len(doc_labels) != len(token_dfs):
            raise RuntimeError('length of document labels must match length of token data frames')

        self._tokens = dict(zip(doc_labels, token_dfs))

    def _ids2tokens(self, tokids):
        return ids2tokens(self._vocab, tokids)

    def _searchtokens2ids(self, searchtokens):
        if not isinstance(searchtokens, np.ndarray):
            searchtokens = np.array(searchtokens)

        return np.where(searchtokens[:, None] == self._vocab)[1]

    def _ids2tokens_for_ngrams(self, docs_ngrams):
        res = {}
        for dl, ngrams in docs_ngrams.items():
            tok = list(map(lambda ngram: ids2tokens(self._vocab, ngram), ngrams))

            if not tok:
                tok = empty_chararray()
            else:
                tok = np.array(tok)

            res[dl] = tok

        return res

    def _remove_tokens_from_vocab(self, remove_mask):
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
            df['token'] = replace(df.token) if len(df.token) > 0 else empty_chararray()

            tmp_tokens[dl] = df.reset_index(drop=True)

        self._tokens = tmp_tokens
