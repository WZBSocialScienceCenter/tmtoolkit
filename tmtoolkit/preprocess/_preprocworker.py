"""
Preprocessing worker class for parallel text processing.
"""

import multiprocessing as mp
from importlib import import_module
import re

import numpy as np
import nltk
from germalemma import GermaLemma

from .. import logger
from ..utils import pos_tag_convert_penn_to_wn, simplified_pos, token_match, \
    expand_compound_token, remove_chars_in_tokens, create_ngrams, ids2tokens, empty_chararray, tokens2ids, \
    make_vocab_unique_and_update_token_ids
from ._common import PATTERN_SUBMODULES


pttrn_metadata_key = re.compile(r'^meta_(.+)$')


class PreprocWorker(mp.Process):
    def __init__(self, worker_id, language, tasks_queue, results_queue, tokenizer, stemmer, pos_tagger,
                 group=None, target=None, name=None, args=(), kwargs=None):
        super().__init__(group, target, name, args, kwargs or {})
        logger.debug('worker `%s`: init with worker ID %d' % (name, worker_id))
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
                                      # dict with document label -> dict of {tokens -> np.array, meta_... -> np.array}
        self._ngrams = {}             # generated ngrams
        self._metadata_keys = []
        self._pos_tag_labels = []

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

    def _task_init(self, docs, docs_are_tokenized):
        logger.debug('worker `%s`: docs = %s' % (self.name, str(set(docs.keys()))))

        if docs_are_tokenized:
            logger.info('got %d already tokenized documents' % len(docs))
            # docs are already tokenized, but their token IDs must be created
            tokenslist = [doc.pop('token') for doc in docs.values()]
            self._create_token_ids_and_vocab(tokenslist, doc_labels=docs.keys())

            meta_keys = None
            for dl, doc in docs.items():   # re-add meta data
                if meta_keys is None:
                    meta_keys = list(doc.keys())
                    for k in meta_keys:
                        if not k.startswith('meta_'):
                            raise ValueError('all meta data keys should start with "meta_"; document `%s` contains '
                                             'meta data key `%s`' % (dl, k))
                elif set(meta_keys) != set(doc.keys()):
                    raise ValueError('all documents must have the same meta data keys `%s`, but document `%s` has '
                                     'different keys: `%s`' % (str(meta_keys), dl, str(list(doc.keys()))))

                self._tokens[dl].update(doc)

            self._metadata_keys = [k[5:] for k in meta_keys]  # strip "meta_"
        else:
            # directly tokenize documents
            logger.info('tokenizing %d documents' % len(docs))
            self._create_token_ids_and_vocab([self.tokenizer.tokenize(txt) for txt in docs.values()],
                                             doc_labels=docs.keys())

    def _task_get_doc_labels(self):
        self.results_queue.put(list(self._tokens.keys()))

    def _task_get_tokens(self):
        tokids = [doc['token'] for doc in self._tokens.values()]
        tokens = self._ids2tokens(tokids)

        result = {}
        for i, (dl, doc) in enumerate(self._tokens.items()):
            # set token strings
            result[dl] = {'token': tokens[i]}

            # convert POS IDs back to POS tags
            if 'pos' in self._metadata_keys:
                result[dl]['meta_pos'] = np.array(ids2tokens(self._pos_tag_labels, doc['meta_pos'])) \
                    if len(doc['meta_pos']) > 0 else empty_chararray()

            # other meta data
            result[dl].update({k: v for k, v in doc.items() if k not in {'token', 'meta_pos'}})

        self.results_queue.put(result)

    def _task_get_available_metadata_keys(self):
        self.results_queue.put(self._metadata_keys)

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

    def _task_get_vocab_doc_frequencies(self):
        doc_freqs = np.zeros(len(self._vocab), dtype=np.uint)

        for doc in self._tokens.values():
            doc_freqs[np.unique(doc['token'])] += 1

        self.results_queue.put(dict(zip(self._vocab, doc_freqs)))

    def _task_get_num_unique_tokens_per_doc(self):
        self.results_queue.put({dl: len(np.unique(doc['token'])) for dl, doc in self._tokens.items()})

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
            '_pos_tag_labels',
            '_metadata_keys'
        )

        state = {attr: getattr(self, attr) for attr in state_attrs}
        logger.debug('worker `%s`: got state with %d items' % (self.name, len(state)))
        self.results_queue.put(state)

    def _task_set_state(self, **state):
        logger.debug('worker `%s`: setting state' % self.name)

        for attr, val in state.items():
            setattr(self, attr, val)

    def _task_add_metadata_per_token(self, key, data, default, dtype):
        logger.debug('worker `%s`: adding metadata per token' % self.name)

        searchtokids = self._searchtokens2ids(list(data.keys()))
        mapper = {searchtoken_id: data[searchtoken] for searchtoken, searchtoken_id in searchtokids.items()}

        def replace(val):
            return mapper.get(val, default)
        replace = np.vectorize(replace)

        col = 'meta_' + key
        for doc in self._tokens.values():
            doc[col] = np.array(replace(doc['token']) if len(doc['token']) > 0 else [], dtype=dtype)

        if key not in self._metadata_keys:
            self._metadata_keys.append(key)

    def _task_add_metadata_per_doc(self, key, data, dtype):
        logger.debug('worker `%s`: adding metadata per document' % self.name)

        col = 'meta_' + key
        for dl, meta in data.items():
            if isinstance(meta, np.ndarray):
                meta_arr = meta
            else:
                meta_arr = np.array(meta, dtype=dtype)

            self._tokens[dl][col] = meta_arr

        if key not in self._metadata_keys:
            self._metadata_keys.append(key)

    def _task_remove_metadata(self, key):
        logger.debug('worker `%s`: removing metadata column' % self.name)

        if key in self._metadata_keys:
            col = 'meta_' + key
            for doc in self._tokens.values():
                if col in doc.keys():
                    del doc[col]

            self._metadata_keys.pop(self._metadata_keys.index(key))

    def _task_generate_ngrams(self, n):
        self._ngrams = {dl: create_ngrams(doc['token'], n=n, join=False)
                        for dl, doc in self._tokens.items()}

    def _task_use_joined_ngrams_as_tokens(self, join_str):
        ngrams_docs = self._ids2tokens_for_ngrams(self._ngrams)

        ngrams_joined = [np.array(list(map(lambda g: join_str.join(g), ngrams)))
                         for ngrams in ngrams_docs.values()]

        self._create_token_ids_and_vocab(ngrams_joined)

        # we have to reset ngrams because they wouldn't map properly to the new vocab any more
        self._ngrams = {}

    def _task_transform_tokens(self, transform_fn, vectorize):
        if vectorize:
            transform_fn = np.vectorize(transform_fn)

        self._update_vocab(transform_fn(self._vocab))

    def _task_tokens_to_lowercase(self):
        self._update_vocab(np.char.lower(self._vocab))

    def _task_stem(self):
        self._task_transform_tokens(self.stemmer.stem, vectorize=True)

    def _task_pos_tag(self):
        all_tags = []

        for doc in self._tokens.values():
            if len(doc['token']) > 0:
                tokens_and_tags = self.pos_tagger.tag(self._ids2tokens(doc['token']))
                all_tags.append(np.array(list(zip(*tokens_and_tags))[1]))
            else:
                all_tags.append(empty_chararray())

        self._pos_tag_labels, tag_ids = tokens2ids(all_tags)

        for doc, pos in zip(self._tokens.values(), tag_ids):
            doc['meta_pos'] = pos

        if 'pos' not in self._metadata_keys:
            self._metadata_keys.append('pos')

    def _task_lemmatize(self, pos_tagset):
        if self.language == 'english':
            if not self.wordnet_lemmatizer:
                self.wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

            lemmatize_fn = self.wordnet_lemmatizer.lemmatize

            def lemmatize_wrapper(row):
                tok, pos = row
                wn_pos = pos_tag_convert_penn_to_wn(pos)
                if wn_pos:
                    return lemmatize_fn(tok, wn_pos)
                else:
                    return tok
        elif self.language == 'german':
            if not self.germalemma:
                self.germalemma = GermaLemma()

            lemmatize_fn = self.germalemma.find_lemma

            def lemmatize_wrapper(row):
                tok, pos = row
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

            def lemmatize_wrapper(row):
                tok, pos = row
                if pos.startswith('NP'):  # singularize noun
                    return lemmatize_fn.singularize(tok)
                elif pos.startswith('V'):  # get infinitive of verb
                    return lemmatize_fn.conjugate(tok, lemmatize_fn.INFINITIVE)
                elif pos.startswith('ADJ') or pos.startswith('ADV'):  # get baseform of adjective or adverb
                    return lemmatize_fn.predicative(tok)
                return tok

        # both are numeric ids:
        all_tokens = np.concatenate([doc['token'] for doc in self._tokens.values()])
        all_tags = np.concatenate([doc['meta_pos'] for doc in self._tokens.values()])
        assert len(all_tokens) == len(all_tags)

        tokens_pos = np.unique(np.column_stack((all_tokens, all_tags)), axis=0)
        assert tokens_pos.ndim == 2
        assert tokens_pos.shape[0] <= len(all_tokens)
        assert tokens_pos.shape[1] == 2

        unique_tokens_tags_strs = zip(ids2tokens(self._vocab, [tokens_pos[:, 0]])[0],           # token strings
                                      ids2tokens(self._pos_tag_labels, [tokens_pos[:, 1]])[0])  # tag strings

        lemmata = map(lemmatize_wrapper, unique_tokens_tags_strs)

        mapper = dict(zip(map(hash, tuple(zip(tokens_pos[:, 0], tokens_pos[:, 1]))), lemmata))  # orig token ID and POS id -> lemma string
        def replace(val):
            return mapper[val]
        replace = np.vectorize(replace)

        new_tokens = []
        for doc in self._tokens.values():
            if len(doc['token']) > 0:
                new_tokens.append(replace(list(map(hash, tuple(zip(doc['token'], doc['meta_pos']))))))
            else:
                new_tokens.append(empty_chararray())

        self._vocab, new_tok_ids, self._vocab_counts = tokens2ids(new_tokens, return_counts=True)

        for doc, new_ids in zip(self._tokens.values(), new_tok_ids):
            doc['token'] = new_ids

    def _task_expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        """
        Note: This function will reset the token dataframe `self._tokens` to the newly created tokens. This means
        all token metadata will be gone.
        """
        tokens = self._ids2tokens([doc['token'] for doc in self._tokens.values()])

        new_tokens = []
        for tok in tokens:
            new = expand_compound_token(tok, split_chars, split_on_len, split_on_casechange)
            if new:
                new_tokens.append(np.concatenate(new))
            else:
                new_tokens.append(empty_chararray())

        self._create_token_ids_and_vocab(new_tokens)
        self._metadata_keys = []  # clear this because after expanding we have new words

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

        remove_mask |= np.isin(self._vocab, list(tokens_to_remove), assume_unique=True)

        if np.sum(remove_mask) > 0:
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

        filtered = {dl: self._ids2tokens(doc['token']) for dl, doc in self._tokens.items()
                    if sum(np.isin(doc['token'], matches_ids)) > 0}

        if self._metadata_keys:
            docs_meta = {dl: {k: v for k, v in self._tokens[dl].items() if k != 'token'} for dl in filtered.keys()}
        else:
            docs_meta = {}

        self._create_token_ids_and_vocab(list(filtered.values()), filtered.keys())

        for dl, doc in docs_meta.items():   # re-add meta data
            self._tokens[dl].update(doc)

    def _task_filter_for_pos(self, required_pos, pos_tagset, simplify_pos):
        if required_pos is None or isinstance(required_pos, str):
            required_pos = {required_pos}  # turn it into a set

        if simplify_pos:
            simplify_fn = np.vectorize(lambda x: simplified_pos(x, tagset=pos_tagset))
        else:
            simplify_fn = None

        filtered_toks = []
        filtered_docs = {}
        for dl, doc in self._tokens.items():
            pos = ids2tokens(self._pos_tag_labels, doc['meta_pos'])
            if simplify_fn:
                pos_simplified = simplify_fn(pos) if len(pos) > 0 else empty_chararray()
            else:
                pos_simplified = pos

            match_mask = np.repeat(False, len(pos_simplified))
            for req_pos in required_pos:
                match_mask |= (pos_simplified == req_pos)

            filtered_docs[dl] = {k: v[match_mask] for k, v in doc.items() if k != 'token'}
            filtered_toks.append(self._ids2tokens(doc['token'][match_mask]))

        self._create_token_ids_and_vocab(filtered_toks)

        for dl, doc in filtered_docs.items():   # re-add meta data
            self._tokens[dl].update(doc)

    def _update_vocab(self, vocab):
        self._vocab, tokids, changed = make_vocab_unique_and_update_token_ids(vocab,
            [doc['token'] for doc in self._tokens.values()], signal_change=True)

        if changed:  # recalculate the vocab counts
            self._update_vocab_counts_from_token_ids(tokids)

        for doc, new_ids in zip(self._tokens.values(), tokids):
            doc['token'] = new_ids
            n_tok = len(new_ids)
            n_others = map(len, (v for k, v in doc.items() if k != 'token'))
            assert all(n == n_tok for n in n_others)

    def _update_vocab_counts_from_token_ids(self, tokids):
        if len(tokids) > 0:
            vocab_ids, counts = np.unique(np.concatenate(tokids), return_counts=True)
            # vocab_ids may not be sorted, but self._vocab_counts maps to sorted vocab
            self._vocab_counts = counts[np.argsort(vocab_ids)]
        else:
            self._vocab_counts = np.array([], dtype=np.int)

    def _create_token_ids_and_vocab(self, tokens, doc_labels=None):
        if doc_labels is None:
            doc_labels = self._tokens.keys()

        self._vocab, tokids, self._vocab_counts = tokens2ids(tokens, return_counts=True)
        token_dfs = list(map(lambda x: {'token': x}, tokids))

        if len(doc_labels) != len(token_dfs):
            raise RuntimeError('length of document labels must match length of token data frames')

        self._tokens = dict(zip(doc_labels, token_dfs))

    def _ids2tokens(self, tokids):
        return ids2tokens(self._vocab, tokids)

    def _searchtokens2ids(self, searchtokens):
        if not isinstance(searchtokens, np.ndarray):
            searchtokens = np.array(searchtokens)

        # TODO: can this be made more (memory) efficient? (broadcasting creates large intermediate object)
        ind_searchtokens, ind_vocab = np.where(searchtokens[:, None] == self._vocab)
        assert len(ind_searchtokens) == len(ind_vocab)

        return dict(zip(searchtokens[ind_searchtokens], ind_vocab))

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
        tmp_tokids = []
        for dl, doc in self._tokens.items():
            filter_indices = np.isin(doc['token'], vocab_ids_masked)

            new_doc = {}
            for k, v in doc.items():
                filt_v = v[filter_indices]
                if k == 'token':
                    new_doc[k] = replace(filt_v) if len(filt_v) > 0 else np.array([], dtype=np.int)
                else:
                    new_doc[k] = filt_v

            tmp_tokids.append(new_doc['token'])
            tmp_tokens[dl] = new_doc

        self._update_vocab_counts_from_token_ids(tmp_tokids)
        self._tokens = tmp_tokens
