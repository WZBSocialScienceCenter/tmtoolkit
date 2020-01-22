"""
Preprocessing worker class for parallel text processing.
"""

import multiprocessing as mp
import re
import logging

from spacy.tokens import Doc, Token

from ._common import ngrams, vocabulary, vocabulary_counts, doc_frequencies, sparse_dtm, \
    glue_tokens, remove_chars, transform, _build_kwic, expand_compounds, clean_tokens, filter_tokens, \
    filter_documents, filter_documents_by_name, filter_for_pos, filter_tokens_by_mask, filter_tokens_with_kwic


logger = logging.getLogger('tmtoolkit')
logger.addHandler(logging.NullHandler())


pttrn_metadata_key = re.compile(r'^meta_(.+)$')

Doc.set_extension('label', default='')      # TODO
Token.set_extension('text', default='')
Token.set_extension('filt', default=True)


class PreprocWorker(mp.Process):
    def __init__(self, worker_id, nlp, tasks_queue, results_queue, shutdown_event,
                 group=None, target=None, name=None, args=(), kwargs=None):
        super().__init__(group, target, name, args, kwargs or {}, daemon=True)
        logger.debug('worker `%s`: init with worker ID %d' % (name, worker_id))
        self.worker_id = worker_id
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue
        self.shutdown_event = shutdown_event

        pipeline_components = dict(nlp.pipeline)
        self.nlp = nlp
        self.tagger = pipeline_components['tagger']

        self._doc_labels = []         # list of document labels for self._tokens   # TODO make as doc extensions
        self._docs = []               # SpaCy documents

        self._std_attrs = ['lemma']
        self._metadata_keys = []
        self._ngrams = []             # generated ngrams as list of token strings

    def run(self):
        logger.debug('worker `%s`: run' % self.name)

        while not self.shutdown_event.is_set():
            q_item = self.tasks_queue.get()

            if q_item is None:
                break

            next_task, task_kwargs = q_item
            logger.debug('worker `%s`: received task `%s`' % (self.name, next_task))

            exec_task_fn = getattr(self, '_task_' + next_task)
            if exec_task_fn:
                exec_task_fn(**task_kwargs)
                self.tasks_queue.task_done()
            else:
                raise NotImplementedError("Task not implemented: `%s`" % next_task)

        logger.debug('worker `%s`: shutting down' % self.name)
        self.tasks_queue.task_done()

    def _task_init(self, docs, docs_are_tokenized):
        logger.debug('worker `%s`: docs = %s' % (self.name, str(set(docs.keys()))))

        self._doc_labels = list(docs.keys())
        self._ngrams = []

        if docs_are_tokenized:  # TODO: what to do here? docs should be spacy docs!
            pass
            # logger.info('got %d already tokenized documents' % len(docs))
            #
            # self._tokens = [doc['token'] for doc in docs.values()]
            #
            # meta_keys = None
            # for dl, doc in docs.items():
            #     doc_meta = {k: metadata for k, metadata in doc.items() if k.startswith('meta_')}
            #     if not all(k.startswith('meta_') for k in doc_meta.keys()):
            #         raise ValueError('all meta data keys must start with "meta_"'
            #                          ' but this is not the case in document `%s`' % dl)
            #
            #     if meta_keys is None:
            #         meta_keys = set(doc_meta.keys())
            #     else:
            #         if meta_keys != set(doc_meta.keys()):
            #             raise ValueError('all documents must contain the same meta data keys')
            #
            # self._metadata_keys = [k[5:] for k in meta_keys]  # strip "meta_"
        else:
            # directly tokenize documents
            logger.info('tokenizing %d documents' % len(docs))

            self._docs = [self.nlp.make_doc(d) for d in docs.values()]
            self._update_docs_attr('text', [[t.text for t in doc] for doc in self._docs])

    @property
    def _tokens(self):
        return self._get_docs_attr('text')

    @property
    def _tokens_meta(self):
        res = []

        for doc in self._docs:
            resdoc = {}

            for meta_key in self._std_attrs:
                k = 'meta_' + meta_key
                assert k not in resdoc
                resdoc[k] = [getattr(t, meta_key + '_') for t in doc]

            for meta_key in self._metadata_keys:
                k = 'meta_' + meta_key
                assert k not in resdoc
                resdoc[k] = [getattr(t._, k) for t in doc]

            res.append(resdoc)

        return res

    def _update_docs_attr(self, attr_name, token_attrs):
        assert len(self._docs) == len(token_attrs)
        for doc, new_attr_vals in zip(self._docs, token_attrs):
            assert len(doc) == len(new_attr_vals)
            for t, v in zip(doc, new_attr_vals):
                setattr(t._, attr_name, v)

    def _get_docs_attr(self, attr_name, custom_attr=True):
        return [[getattr(t._, attr_name) if custom_attr else getattr(t, attr_name) for t in doc] for doc in self._docs]

    def _task_get_doc_labels(self):
        self.results_queue.put(self._doc_labels)

    def _task_get_tokens(self):
        # tokens with metadata
        res = {}
        for dl, doc in zip(self._doc_labels, self._docs):
            resdoc = {'token': [t._.text for t in doc]}

            for meta_key in self._std_attrs:
                k = 'meta_' + meta_key
                assert k not in resdoc
                resdoc[k] = [getattr(t, meta_key + '_') for t in doc]

            for meta_key in self._metadata_keys:
                k = 'meta_' + meta_key
                assert k not in resdoc
                resdoc[k] = [getattr(t._, k) for t in doc]

            res[dl] = resdoc

        self.results_queue.put(res)

    def _task_replace_tokens(self, tokens):   # TODO: update _docs? how?
        assert set(tokens.keys()) == set(self._doc_labels)
        # for dl, dt in tokens.items():
        #     self._tokens[self._doc_labels.index(dl)] = dt

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
            '_doc_labels',
            '_docs',
            '_ngrams',
            '_std_attrs',
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

        attr_name = 'meta_' + key
        Token.set_extension(attr_name, default=default)

        for doc, tok in zip(self._docs, self._tokens):
            assert len(doc) == len(tok)
            for t, tok_text in zip(doc, tok):
                setattr(t._, attr_name, data.get(tok_text, default))

        if key not in self._metadata_keys:
            self._metadata_keys.append(key)

    def _task_add_metadata_per_doc(self, key, data, default):   # TODO caller must provide default
        logger.debug('worker `%s`: adding metadata per document' % self.name)

        attr_name = 'meta_' + key
        Token.set_extension(attr_name, default=default)

        for dl, doc in zip(self._doc_labels, self._docs):
            meta_vals = data.get(dl, [default] * len(doc))
            assert len(doc) == len(meta_vals)
            for t, v in zip(doc, meta_vals):
                setattr(t._, attr_name, v)

        if key not in self._metadata_keys:
            self._metadata_keys.append(key)

    def _task_remove_metadata(self, key):   # TODO: use spacy token attrib.
        logger.debug('worker `%s`: removing metadata column' % self.name)

        if key in self._metadata_keys:
            Token.remove_extension('meta_' + key)
            self._metadata_keys.pop(self._metadata_keys.index(key))

    def _task_generate_ngrams(self, n):
        self._ngrams = ngrams(self._tokens, n, join=False)

    def _task_use_joined_ngrams_as_tokens(self, join_str):   # TODO: update _docs? how?
        #self._tokens = [list(map(lambda g: join_str.join(g), dngrams)) for dngrams in self._ngrams]

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

        # reset ngrams as they're used as normal tokens now
        self._ngrams = {}

    def _task_transform_tokens(self, transform_fn, **kwargs):
        self._update_docs_attr('text', transform(self._tokens,  transform_fn, **kwargs))

    def _task_tokens_to_lowercase(self):
        self._update_docs_attr('text', self._get_docs_attr('lower_', custom_attr=False))

    # def _task_stem(self):   # TODO: disable or optional?
    #     self._tokens = self.stemmer(self._tokens)

    def _task_remove_chars(self, chars):
        self._update_docs_attr('text', remove_chars(self._tokens, chars=chars))

    def _task_pos_tag(self):
        if 'pos' not in self._std_attrs:
            self.tagger(self._docs)
            self._std_attrs.append('pos')

    def _task_lemmatize(self):   # TODO: update _docs? how?
        self._update_docs_attr('text', self._get_docs_attr('lemma_', custom_attr=False))

        if 'lemma' not in self._std_attrs:
            self._std_attrs.append('lemma')

    def _task_expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        # TODO: https://spacy.io/usage/linguistic-features#retokenization
        self._tokens = expand_compounds(self._tokens, split_chars=split_chars, split_on_len=split_on_len,
                                        split_on_casechange=split_on_casechange)

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

    def _task_clean_tokens(self, tokens_to_remove, remove_shorter_than, remove_longer_than, remove_numbers):
        # TODO: masking
        self._tokens, self._tokens_meta = clean_tokens(self._tokens, self._tokens_meta, remove_punct=False,
                                                       remove_stopwords=tokens_to_remove, remove_empty=False,
                                                       remove_shorter_than=remove_shorter_than,
                                                       remove_longer_than=remove_longer_than,
                                                       remove_numbers=remove_numbers)

    def _task_get_kwic(self, search_tokens, highlight_keyword, with_metadata, with_window_indices, context_size,
                       match_type, ignore_case, glob_method, inverse):

        docs = list(zip(self._tokens, self._tokens_meta))

        kwic = _build_kwic(docs, search_tokens,
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
        # TODO: https://spacy.io/usage/linguistic-features#retokenization
        new_tokens_and_meta, glued_tokens = glue_tokens(list(zip(self._tokens, self._tokens_meta)), patterns,
                                                        glue=glue, match_type=match_type, ignore_case=ignore_case,
                                                        glob_method=glob_method, inverse=inverse,
                                                        return_glued_tokens=True)
        if new_tokens_and_meta:
            self._tokens, self._tokens_meta = zip(*new_tokens_and_meta)

        # result is a set of glued tokens
        self.results_queue.put(glued_tokens)

    def _task_filter_tokens_by_mask(self, mask, inverse):
        # TODO: masking
        mask_list = [mask[dl] for dl in self._doc_labels]
        self._tokens, self._tokens_meta = filter_tokens_by_mask(self._tokens, mask_list, self._tokens_meta,
                                                                inverse=inverse)

    def _task_filter_tokens(self, search_tokens, match_type, ignore_case, glob_method, inverse, by_meta):
        # TODO: masking
        if by_meta:
            by_meta = 'meta_' + by_meta

        self._tokens, self._tokens_meta = filter_tokens(self._tokens, search_tokens, self._tokens_meta,
                                                        match_type=match_type, ignore_case=ignore_case,
                                                        glob_method=glob_method, inverse=inverse, by_meta=by_meta)

    def _task_filter_tokens_with_kwic(self, search_tokens, context_size, match_type, ignore_case,
                                      glob_method, inverse):
        # TODO: masking
        self._tokens, self._tokens_meta = filter_tokens_with_kwic(self._tokens, search_tokens, self._tokens_meta,
                                                                  context_size=context_size, match_type=match_type,
                                                                  ignore_case=ignore_case, glob_method=glob_method,
                                                                  inverse=inverse)

    def _task_filter_documents(self, search_tokens, by_meta, matches_threshold, match_type, ignore_case, glob_method,
                               inverse_result, inverse_matches):
        # TODO: masking
        if by_meta:
            by_meta = 'meta_' + by_meta

        self._tokens, self._tokens_meta, self._doc_labels = filter_documents(
            self._tokens, search_tokens, by_meta=by_meta, docs_meta=self._tokens_meta, doc_labels=self._doc_labels,
            matches_threshold=matches_threshold, match_type=match_type, ignore_case=ignore_case,
            glob_method=glob_method, inverse_result=inverse_result, inverse_matches=inverse_matches
        )

    def _task_filter_documents_by_name(self, name_patterns, match_type, ignore_case, glob_method, inverse):
        # TODO: masking
        self._tokens, self._doc_labels, self._tokens_meta = filter_documents_by_name(self._tokens, self._doc_labels,
                                                                                     name_patterns, self._tokens_meta,
                                                                                     match_type=match_type,
                                                                                     ignore_case=ignore_case,
                                                                                     glob_method=glob_method,
                                                                                     inverse=inverse)

    def _task_filter_for_pos(self, required_pos, pos_tagset, simplify_pos, inverse):
        # TODO: masking
        self._tokens, self._tokens_meta = filter_for_pos(self._tokens, self._tokens_meta,
                                                         required_pos=required_pos,
                                                         tagset=pos_tagset,
                                                         simplify_pos=simplify_pos,
                                                         inverse=inverse)

    def _clear_metadata(self):
        for k in self._metadata_keys:
            self._task_remove_metadata(k)

        assert len(self._metadata_keys) == 0
