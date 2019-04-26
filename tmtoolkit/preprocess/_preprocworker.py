"""
Preprocessing worker class for parallel text processing.
"""

import multiprocessing as mp
from importlib import import_module
from collections import Counter
import re

import numpy as np
import nltk
from germalemma import GermaLemma

from .. import logger
from ..utils import flatten_list, pos_tag_convert_penn_to_wn, simplified_pos, token_match, \
    expand_compound_token, remove_chars_in_tokens, create_ngrams
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

        self._doc_labels = []         # list of document labels for self._tokens
        self._tokens = []             # tokens for this worker at the current processing stage.
                                      # list of token strings
        self._tokens_meta = []        # list of dicts of with metadata for each token in each document {meta_... -> list}
        self._metadata_keys = []
        self._ngrams = []             # generated ngrams as list of token strings

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

        self._doc_labels = list(docs.keys())
        self._ngrams = []

        if docs_are_tokenized:
            logger.info('got %d already tokenized documents' % len(docs))

            self._tokens = [doc['token'] for doc in docs.values()]

            meta_keys = None
            self._tokens_meta = []
            for dl, doc in docs.items():
                doc_meta = {k: metadata for k, metadata in doc.items() if k.startswith('meta_')}
                self._tokens_meta.append(doc_meta)
                if not all(k.startswith('meta_') for k in doc_meta.keys()):
                    raise ValueError('all meta data keys must start with "meta_"'
                                     ' but this is not the case in document `%s`' % dl)

                if meta_keys is None:
                    meta_keys = set(doc_meta.keys())
                else:
                    if meta_keys != set(doc_meta.keys()):
                        raise ValueError('all documents must contain the same meta data keys')

            self._metadata_keys = [k[5:] for k in meta_keys]  # strip "meta_"
        else:
            # directly tokenize documents
            logger.info('tokenizing %d documents' % len(docs))

            self._tokens = [self.tokenizer.tokenize(txt) for txt in docs.values()]
            self._tokens_meta = [{} for _ in range(len(docs))]

    def _task_get_doc_labels(self):
        self.results_queue.put(self._doc_labels)

    def _task_get_tokens(self):
        # tokens with metadata
        self.results_queue.put(dict(zip(self._doc_labels,
                                        (dict(meta, token=t) for t, meta in zip(self._tokens, self._tokens_meta)))))

    def _task_replace_tokens(self, tokens):
        assert set(tokens.keys()) == set(self._doc_labels)
        for dl, dt in tokens.items():
            self._tokens[self._doc_labels.index(dl)] = dt

    def _task_get_available_metadata_keys(self):
        self.results_queue.put(self._metadata_keys)

    def _task_get_vocab_counts(self):
        self.results_queue.put(Counter(flatten_list(self._tokens)))

    def _task_get_vocab_doc_frequencies(self):
        doc_freqs = Counter()

        for dt in self._tokens:
            for t in set(dt):
                doc_freqs[t] += 1

        self.results_queue.put(doc_freqs)

    def _task_get_num_unique_tokens_per_doc(self):
        self.results_queue.put({dl: len(set(dt)) for dl, dt in zip(self._doc_labels, self._tokens)})

    def _task_get_ngrams(self):
        self.results_queue.put(dict(zip(self._doc_labels, self._ngrams)))

    def _task_get_state(self):
        logger.debug('worker `%s`: getting state' % self.name)

        state_attrs = (
            'language',
            '_doc_labels',
            '_tokens',
            '_tokens_meta',
            '_ngrams',
            '_metadata_keys'
        )

        state = {attr: getattr(self, attr) for attr in state_attrs}
        logger.debug('worker `%s`: got state with %d items' % (self.name, len(state)))
        self.results_queue.put(state)

    def _task_set_state(self, **state):
        logger.debug('worker `%s`: setting state' % self.name)

        for attr, val in state.items():
            setattr(self, attr, val)

    def _task_add_metadata_per_token(self, key, data, default):
        logger.debug('worker `%s`: adding metadata per token' % self.name)

        col = 'meta_' + key
        for dt, dmeta in zip(self._tokens, self._tokens_meta):
            dmeta[col] = []
            for t in dt:
                dmeta[col].append(data.get(t, default))

        if key not in self._metadata_keys:
            self._metadata_keys.append(key)

    def _task_add_metadata_per_doc(self, key, data):
        logger.debug('worker `%s`: adding metadata per document' % self.name)

        col = 'meta_' + key
        for dl, tmeta in zip(self._doc_labels, self._tokens_meta):
            tmeta[col] = data[dl]

        if key not in self._metadata_keys:
            self._metadata_keys.append(key)

    def _task_remove_metadata(self, key):
        logger.debug('worker `%s`: removing metadata column' % self.name)

        if key in self._metadata_keys:
            col = 'meta_' + key
            for tmeta in self._tokens_meta:
                del tmeta[col]

            self._metadata_keys.pop(self._metadata_keys.index(key))

    def _task_generate_ngrams(self, n):
        self._ngrams = [create_ngrams(dt, n=n, join=False) for dt in self._tokens]

    def _task_use_joined_ngrams_as_tokens(self, join_str):
        self._tokens = [list(map(lambda g: join_str.join(g), dngrams)) for dngrams in self._ngrams]

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

        # reset ngrams as they're used as normal tokens now
        self._ngrams = {}

    def _task_transform_tokens(self, transform_fn, **kwargs):
        if kwargs:
            transform_fn_wrapper = lambda x: transform_fn(x, **kwargs)
        else:
            transform_fn_wrapper = transform_fn

        self._tokens = [list(map(transform_fn_wrapper, dt)) for dt in self._tokens]

    def _task_tokens_to_lowercase(self):
        self._task_transform_tokens(str.lower)

    def _task_stem(self):
        self._task_transform_tokens(self.stemmer.stem)

    def _task_remove_chars_in_tokens(self, chars):
        self._tokens = [remove_chars_in_tokens(dt, chars=chars) for dt in self._tokens]

    def _task_pos_tag(self):
        for dt, dmeta in zip(self._tokens, self._tokens_meta):
            if len(dt) > 0:
                tokens_and_tags = self.pos_tagger.tag(dt)
                tags = list(zip(*tokens_and_tags))[1]
            else:
                tags = []

            dmeta['meta_pos'] = tags

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

        new_tokens = []
        for dt, dmeta in zip(self._tokens, self._tokens_meta):
            new_tokens.append(list(map(lemmatize_wrapper, zip(dt, dmeta['meta_pos']))))

        self._tokens = new_tokens

    def _task_expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        """
        Note: This function will reset the token dataframe `self._tokens` to the newly created tokens. This means
        all token metadata will be gone.
        """
        self._tokens = [flatten_list(expand_compound_token(dt, split_chars, split_on_len, split_on_casechange))
                        for dt in self._tokens]

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

    def _task_clean_tokens(self, tokens_to_remove, remove_shorter_than=None, remove_longer_than=None,
                           remove_numbers=False):
        remove_masks = [np.repeat(False, len(dt)) for dt in self._tokens]

        if remove_shorter_than is not None or remove_longer_than is not None:
            token_lengths = [np.fromiter(map(len, dt), np.int, len(dt)) for dt in self._tokens]
        else:
            token_lengths = None

        if remove_shorter_than is not None:
            remove_masks = [mask | (n < remove_shorter_than) for mask, n in zip(remove_masks, token_lengths)]

        if remove_longer_than is not None:
            remove_masks = [mask | (n > remove_longer_than) for mask, n in zip(remove_masks, token_lengths)]

        if remove_numbers:
            remove_masks = [mask | np.char.isnumeric(np.array(dt, dtype=str))
                            for mask, dt in zip(remove_masks, self._tokens)]

        if tokens_to_remove:
            tokens_to_remove = set(tokens_to_remove)
            # this is actually much faster than using np.isin:
            remove_masks = [mask | np.array([t in tokens_to_remove for t in dt], dtype=bool)
                            for mask, dt in zip(remove_masks, self._tokens)]

        self._apply_matches_array(remove_masks, invert=True)

    def _task_filter_tokens(self, search_token, match_type, ignore_case, glob_method, inverse):
        matches = [token_match(search_token, dt,
                               match_type=match_type,
                               ignore_case=ignore_case,
                               glob_method=glob_method) for dt in self._tokens]

        self._apply_matches_array(matches, invert=inverse)

    def _task_filter_documents(self, search_token, match_type, ignore_case, glob_method, inverse):
        matches = [token_match(search_token, dt,
                               match_type=match_type,
                               ignore_case=ignore_case,
                               glob_method=glob_method) for dt in self._tokens]

        if inverse:
            matches = [~m for m in matches]

        new_doc_labels = []
        new_tokens = []
        new_meta = []
        for dl, dt, dmeta, n in zip(self._doc_labels, self._tokens, self._tokens_meta, map(np.sum, matches)):
            if n > 0:
                new_doc_labels.append(dl)
                new_tokens.append(dt)
                new_meta.append(dmeta)

        self._doc_labels = new_doc_labels
        self._tokens = new_tokens
        self._tokens_meta = new_meta

    def _task_filter_for_pos(self, required_pos, pos_tagset, simplify_pos, inverse):
        if required_pos is None or isinstance(required_pos, str):
            required_pos = [required_pos]

        if simplify_pos:
            simplify_fn = np.vectorize(lambda x: simplified_pos(x, tagset=pos_tagset))
        else:
            simplify_fn = np.vectorize(lambda x: x)  # identity function

        matches = [np.isin(simplify_fn(dmeta['meta_pos']), required_pos) if len(dmeta['meta_pos']) > 0
                   else np.array([], dtype=bool)
                   for dt, dmeta in zip(self._tokens, self._tokens_meta)]

        self._apply_matches_array(matches, invert=inverse)

    def _apply_matches_array(self, matches, invert=False):
        if invert:
            matches = [~m for m in matches]

        self._tokens = [np.array(dt)[mask].tolist() for mask, dt in zip(matches, self._tokens)]

        new_meta = []
        for mask, dmeta in zip(matches, self._tokens_meta):
            new_dmeta = {}
            for meta_key, meta_vals in dmeta.items():
                new_dmeta[meta_key] = np.array(meta_vals)[mask].tolist()
            new_meta.append(new_dmeta)

        self._tokens_meta = new_meta

    def _clear_metadata(self):
        self._tokens_meta = [{} for _ in range(len(self._tokens))]
        self._metadata_keys = []
