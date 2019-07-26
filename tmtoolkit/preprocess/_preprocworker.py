"""
Preprocessing worker class for parallel text processing.
"""

import multiprocessing as mp
import re
import logging

import numpy as np

from ..utils import merge_dict_sequences_inplace
from ._common import ngrams, vocabulary, vocabulary_counts, doc_frequencies, sparse_dtm, \
    glue_tokens, remove_chars, token_match, \
    simplified_pos, transform, _build_kwic, expand_compounds, clean_tokens, filter_tokens, \
    filter_documents, filter_documents_by_name



logger = logging.getLogger('tmtoolkit')
logger.addHandler(logging.NullHandler())


pttrn_metadata_key = re.compile(r'^meta_(.+)$')


class PreprocWorker(mp.Process):
    def __init__(self, worker_id, language, tasks_queue, results_queue, tokenizer, stemmer, lemmatizer, pos_tagger,
                 group=None, target=None, name=None, args=(), kwargs=None):
        super().__init__(group, target, name, args, kwargs or {})
        logger.debug('worker `%s`: init with worker ID %d' % (name, worker_id))
        self.worker_id = worker_id
        self.language = language
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue

        # set a tokenizer
        self.tokenizer = tokenizer      # tokenizer function

        # set a stemmer
        self.stemmer = stemmer           # stemmer function

        # set a lemmatizer
        self.lemmatizer = lemmatizer     # lemmatizer function

        # set a POS tagger
        self.pos_tagger = pos_tagger        # POS tagger instance (must have a callable attribute `tag`)

        self.pattern_module = None          # dynamically loaded CLiPS pattern library module
        self.germalemma = None              # GermaLemma instance
        self.wordnet_lemmatizer = None      # nltk.stem.WordNetLemmatizer instance

        self._doc_labels = []         # list of document labels for self._tokens
        self._tokens = []             # tokens for this worker at the current processing stage.
                                      # list of token strings
        self._tokens_meta = []        # dict of lists with metadata for each token in each document {meta_... -> list}
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

            self._tokens = self.tokenizer(list(docs.values()), language=self.language)
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

    def _task_get_vocab(self):
        """Put this worker's vocabulary in the result queue."""
        self.results_queue.put(vocabulary(self._tokens))

    def _task_get_vocab_counts(self):
        self.results_queue.put(vocabulary_counts(self._tokens))

    def _task_get_vocab_doc_frequencies(self):
        self.results_queue.put(doc_frequencies(self._tokens))

    def _task_get_ngrams(self):
        self.results_queue.put(dict(zip(self._doc_labels, self._ngrams)))

    def _task_get_dtm(self):
        """
        Put this worker's document-term-matrix (DTM), the document labels and sorted vocabulary in the result queue.
        """

        # create a sparse DTM in COO format
        logger.info('creating sparse DTM for %d documents' % len(self._doc_labels))
        dtm, vocab = sparse_dtm(self._tokens)

        # put tuple in queue with:
        # DTM, document labels that correspond to DTM rows and vocab that corresponds to DTM columns
        self.results_queue.put((dtm, self._doc_labels, vocab))

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
        self._ngrams = ngrams(self._tokens, n, join=False)

    def _task_use_joined_ngrams_as_tokens(self, join_str):
        self._tokens = [list(map(lambda g: join_str.join(g), dngrams)) for dngrams in self._ngrams]

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

        # reset ngrams as they're used as normal tokens now
        self._ngrams = {}

    def _task_transform_tokens(self, transform_fn, **kwargs):
        self._tokens = transform(self._tokens,  transform_fn, **kwargs)

    def _task_tokens_to_lowercase(self):
        self._tokens = transform(self._tokens, str.lower)

    def _task_stem(self):
        self._tokens = self.stemmer(self._tokens)

    def _task_remove_chars(self, chars):
        self._tokens = remove_chars(self._tokens, chars=chars)

    def _task_pos_tag(self):
        pos_tags = self.pos_tagger(self._tokens)
        merge_dict_sequences_inplace(self._tokens_meta, pos_tags)

        if 'pos' not in self._metadata_keys:
            self._metadata_keys.append('pos')

    def _task_lemmatize(self):
        self._tokens = self.lemmatizer(self._tokens, self._tokens_meta)

    def _task_expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        """
        Note: This function will reset the token dataframe `self._tokens` to the newly created tokens. This means
        all token metadata will be gone.
        """
        self._tokens = expand_compounds(self._tokens, split_chars=split_chars, split_on_len=split_on_len,
                                        split_on_casechange=split_on_casechange)

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

    def _task_clean_tokens(self, tokens_to_remove, remove_shorter_than, remove_longer_than, remove_numbers):
        # punctuation, empty token and stopwords may already be included in `tokens_to_remove`
        self._tokens, self._tokens_meta = clean_tokens(self._tokens, self._tokens_meta, remove_punct=False,
                                                       remove_stopwords=tokens_to_remove, remove_empty=False,
                                                       remove_shorter_than=remove_shorter_than,
                                                       remove_longer_than=remove_longer_than,
                                                       remove_numbers=remove_numbers)

    def _task_get_kwic(self, search_token, highlight_keyword, with_metadata, with_window_indices, context_size,
                       match_type, ignore_case, glob_method, inverse):

        docs = list(zip(self._tokens, self._tokens_meta)) if self._metadata_keys else self._tokens

        kwic = _build_kwic(docs, search_token,
                           highlight_keyword=highlight_keyword,
                           with_metadata=with_metadata,
                           with_window_indices=with_window_indices,
                           context_size=context_size,
                           match_type=match_type,
                           ignore_case=ignore_case,
                           glob_method=glob_method,
                           inverse=inverse)

        # result is a dict with doc label -> list of kwic windows, where each kwic window is dict with
        # token -> token list and optionally meta_* -> meta data list
        self.results_queue.put(dict(zip(self._doc_labels, kwic)))

    def _task_glue_tokens(self, patterns, glue, match_type, ignore_case, glob_method, inverse):
        new_tokens_and_meta, glued_tokens = glue_tokens(list(zip(self._tokens, self._tokens_meta)), patterns,
                                                        glue=glue, match_type=match_type, ignore_case=ignore_case,
                                                        glob_method=glob_method, inverse=inverse,
                                                        return_glued_tokens=True)
        if new_tokens_and_meta:
            self._tokens, self._tokens_meta = zip(*new_tokens_and_meta)

        # result is a set of glued tokens
        self.results_queue.put(glued_tokens)

    def _task_filter_tokens(self, search_tokens, match_type, ignore_case, glob_method, inverse):
        self._tokens, self._tokens_meta = filter_tokens(self._tokens, search_tokens, self._tokens_meta,
                                                        match_type=match_type, ignore_case=ignore_case,
                                                        glob_method=glob_method, inverse=inverse)

    def _task_filter_documents(self, search_tokens, matches_threshold, match_type, ignore_case, glob_method, inverse):
        self._tokens, self._tokens_meta, self._doc_labels = filter_documents(
            self._tokens, search_tokens, docs_meta=self._tokens_meta, doc_labels=self._doc_labels,
            matches_threshold=matches_threshold, match_type=match_type, ignore_case=ignore_case,
            glob_method=glob_method, inverse=inverse
        )

    def _task_filter_documents_by_name(self, name_patterns, match_type, ignore_case, glob_method, inverse):
        self._tokens, self._doc_labels, self._tokens_meta = filter_documents_by_name(self._tokens, self._doc_labels,
                                                                                     name_patterns, self._tokens_meta,
                                                                                     match_type=match_type,
                                                                                     ignore_case=ignore_case,
                                                                                     glob_method=glob_method,
                                                                                     inverse=inverse)

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
