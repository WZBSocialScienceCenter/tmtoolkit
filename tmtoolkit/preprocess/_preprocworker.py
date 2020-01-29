"""
Preprocessing worker class for parallel text processing.
"""

import multiprocessing as mp
import logging

from spacy.tokens import Doc, Token

from ._common import ngrams, vocabulary, vocabulary_counts, doc_frequencies, sparse_dtm, \
    glue_tokens, remove_chars, transform, _build_kwic, expand_compounds, clean_tokens, filter_tokens, \
    filter_documents, filter_documents_by_name, filter_for_pos, filter_tokens_by_mask, filter_tokens_with_kwic, \
    _get_docs_attr, _get_docs_tokenattrs


logger = logging.getLogger('tmtoolkit')
logger.addHandler(logging.NullHandler())


Doc.set_extension('label', default='')
Token.set_extension('text', default='')


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

        self._docs = []               # SpaCy documents

        self._std_attrs = ['lemma', 'whitespace']
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

    def _get_tokens_with_metadata(self, as_dict=True, only_metadata=False):
        if as_dict:
            res = {}
        else:
            res = []

        for doc in self._docs:
            if only_metadata:
                resdoc = {}
            else:
                resdoc = {'token': [t._.text for t in doc]}

            for meta_key in self._std_attrs:
                assert meta_key not in resdoc
                if meta_key == 'whitespace':
                    resdoc[meta_key] = [bool(getattr(t, meta_key + '_')) for t in doc]
                else:
                    resdoc[meta_key] = [getattr(t, meta_key + '_') for t in doc]

            for meta_key in self._metadata_keys:
                k = 'meta_' + meta_key
                assert k not in resdoc
                resdoc[k] = [getattr(t._, k) for t in doc]

            if as_dict:
                res[doc._.label] = resdoc
            else:
                res.append(resdoc)

        return res

    @property
    def _tokens(self):
        return _get_docs_tokenattrs(self._docs, 'text')

    @property
    def _doc_labels(self):
        return _get_docs_attr(self._docs, 'label')

    @property
    def _tokens_meta(self):
        return self._get_tokens_with_metadata(as_dict=False, only_metadata=True)

    def _update_docs_tokenattrs(self, attr_name, token_attrs):
        assert len(self._docs) == len(token_attrs)
        for doc, new_attr_vals in zip(self._docs, token_attrs):
            assert len(doc) == len(new_attr_vals)
            for t, v in zip(doc, new_attr_vals):
                setattr(t._, attr_name, v)

    def _remove_metadata(self, key):
        if key in self._metadata_keys:
            Token.remove_extension('meta_' + key)
            self._metadata_keys.pop(self._metadata_keys.index(key))

    def _clear_metadata(self):
        for k in self._metadata_keys:
            self._remove_metadata(k)

        assert len(self._metadata_keys) == 0

    def _task_init(self, docs, docs_are_tokenized):
        logger.debug('worker `%s`: docs = %s' % (self.name, str(set(docs.keys()))))

        self._docs = []
        self._ngrams = []

        if docs_are_tokenized:
            logger.info('got %d already tokenized documents' % len(docs))

            for doc_i, doc_data in enumerate(docs.values()):
                doc_kwargs = dict(words=doc_data['token'])
                if 'whitespace' in doc_data:
                    doc_kwargs['spaces'] = doc_data['whitespace']
                new_doc = Doc(self.nlp.vocab, **doc_kwargs)

                for k, metadata in doc_data.items():
                    if k in {'token', 'whitespace'}: continue
                    is_std = k in {'pos', 'lemma'}

                    assert len(new_doc) == len(metadata)
                    for t, v in zip(new_doc, metadata):
                        if is_std:
                            attr = k + '_'
                            obj = t
                        else:
                            attr = k
                            obj = t._

                        setattr(obj, attr, v)

                    if doc_i == 0:
                        if is_std:
                            if k not in self._std_attrs:
                                self._std_attrs.append(k)
                        else:
                            meta_k = k[5:]       # strip "meta_"
                            if meta_k not in self._metadata_keys:
                                self._metadata_keys.append(meta_k)

                self._docs.append(new_doc)

            self._update_docs_tokenattrs('text', [[t.text for t in doc] for doc in self._docs])
        else:
            # directly tokenize documents
            logger.info('tokenizing %d documents' % len(docs))

            self._docs = [self.nlp.make_doc(d) for d in docs.values()]
            self._update_docs_tokenattrs('text', [[t.text for t in doc] for doc in self._docs])

        assert len(docs) == len(self._docs)
        for dl, doc in zip(docs.keys(), self._docs):
            doc._.label = dl

    def _task_get_doc_labels(self):
        self.results_queue.put(self._doc_labels)

    def _task_get_tokens(self):
        # tokens with metadata
        self.results_queue.put(self._get_tokens_with_metadata())

    def _task_replace_tokens(self, tokens):
        assert set(tokens.keys()) == set(self._doc_labels)
        for dl, new_tok in tokens.items():
            doc = self._docs[self._doc_labels.index(dl)]
            assert len(doc) == len(new_tok)
            for t, nt in zip(doc, new_tok):
                setattr(t._, 'text', nt)

    def _task_get_available_metadata_keys(self):
        self.results_queue.put(self._std_attrs + self._metadata_keys)

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
        logger.info('creating sparse DTM for %d documents' % len(self._docs))
        dtm, vocab = sparse_dtm(self._tokens)

        # put tuple in queue with:
        # DTM, document labels that correspond to DTM rows and vocab that corresponds to DTM columns
        self.results_queue.put((dtm, self._doc_labels, vocab))

    def _task_get_state(self):
        logger.debug('worker `%s`: getting state' % self.name)

        state_attrs = (
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

        for doc in self._docs:
            meta_vals = data.get(doc._.label, [default] * len(doc))
            assert len(doc) == len(meta_vals)
            for t, v in zip(doc, meta_vals):
                setattr(t._, attr_name, v)

        if key not in self._metadata_keys:
            self._metadata_keys.append(key)

    def _task_remove_metadata(self, key):
        logger.debug('worker `%s`: removing metadata column' % self.name)
        self._remove_metadata(key)

    def _task_generate_ngrams(self, n):
        self._ngrams = ngrams(self._tokens, n, join=False)

    def _task_use_joined_ngrams_as_tokens(self, join_str):   # TODO: update _docs? how?
        #self._tokens = [list(map(lambda g: join_str.join(g), dngrams)) for dngrams in self._ngrams]

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

        # reset ngrams as they're used as normal tokens now
        self._ngrams = {}

    def _task_transform_tokens(self, transform_fn, **kwargs):
        self._update_docs_tokenattrs('text', transform(self._tokens, transform_fn, **kwargs))

    def _task_tokens_to_lowercase(self):
        self._update_docs_tokenattrs('text', _get_docs_tokenattrs(self._docs, 'lower_', custom_attr=False))

    # def _task_stem(self):   # TODO: disable or optional?
    #     self._tokens = self.stemmer(self._tokens)

    def _task_remove_chars(self, chars):
        self._update_docs_tokenattrs('text', remove_chars(self._tokens, chars=chars))

    def _task_pos_tag(self):
        if 'pos' not in self._std_attrs:
            for d in self._docs:
                self.tagger(d)
            self._std_attrs.append('pos')

    def _task_lemmatize(self):
        docs_lemmata = _get_docs_tokenattrs(self._docs, 'lemma_', custom_attr=False)

        # SpaCy lemmata sometimes contain special markers like -PRON- instead of the lemma;
        # fix this here by resorting to the original token
        new_docs_lemmata = []
        assert len(docs_lemmata) == len(self._tokens)
        for doc_tok, doc_lem in zip(self._tokens, docs_lemmata):
            assert len(doc_tok) == len(doc_lem)
            new_docs_lemmata.append([t if l.startswith('-') and l.endswith('-') else l
                                     for t, l in zip(doc_tok, doc_lem)])

        self._update_docs_tokenattrs('text', new_docs_lemmata)

        # if 'lemma' in self._std_attrs:
        #     self._std_attrs.pop(self._std_attrs.index('lemma'))

    def _task_expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        self._docs = expand_compounds(self._docs, split_chars=split_chars, split_on_len=split_on_len,
                                      split_on_casechange=split_on_casechange)

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

    def _task_clean_tokens(self, tokens_to_remove, remove_punct, remove_empty,
                           remove_shorter_than, remove_longer_than, remove_numbers):
        self._docs = clean_tokens(self._docs, remove_punct=remove_punct,
                                  remove_stopwords=tokens_to_remove, remove_empty=remove_empty,
                                  remove_shorter_than=remove_shorter_than,
                                  remove_longer_than=remove_longer_than,
                                  remove_numbers=remove_numbers)

    def _task_get_kwic(self, search_tokens, highlight_keyword, with_metadata, with_window_indices, context_size,
                       match_type, ignore_case, glob_method, inverse):
        kwic = _build_kwic(self._docs, search_tokens,
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
        _, glued_tokens = glue_tokens(self._docs, patterns,
                                      glue=glue, match_type=match_type, ignore_case=ignore_case,
                                      glob_method=glob_method, inverse=inverse,
                                      return_glued_tokens=True)

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

        # result is a set of glued tokens
        self.results_queue.put(glued_tokens)

    def _task_filter_tokens_by_mask(self, mask, inverse):
        mask_list = [mask[dl] for dl in self._doc_labels]
        self._docs = filter_tokens_by_mask(self._docs, mask_list, inverse=inverse)

    def _task_filter_tokens(self, search_tokens, match_type, ignore_case, glob_method, inverse, by_meta):
        if by_meta:
            by_meta = 'meta_' + by_meta

        self._docs = filter_tokens(self._docs, search_tokens, match_type=match_type, ignore_case=ignore_case,
                                   glob_method=glob_method, inverse=inverse, by_meta=by_meta)

    def _task_filter_tokens_with_kwic(self, search_tokens, context_size, match_type, ignore_case,
                                      glob_method, inverse):
        self._docs = filter_tokens_with_kwic(self._docs, search_tokens,
                                             context_size=context_size, match_type=match_type,
                                             ignore_case=ignore_case, glob_method=glob_method,
                                             inverse=inverse)

    def _task_filter_documents(self, search_tokens, by_meta, matches_threshold, match_type, ignore_case, glob_method,
                               inverse_result, inverse_matches):
        if by_meta:
            by_meta = 'meta_' + by_meta

        self._docs = filter_documents(
            self._docs, search_tokens, by_meta=by_meta,
            matches_threshold=matches_threshold, match_type=match_type, ignore_case=ignore_case,
            glob_method=glob_method, inverse_result=inverse_result, inverse_matches=inverse_matches
        )

    def _task_filter_documents_by_name(self, name_patterns, match_type, ignore_case, glob_method, inverse):
        self._docs = filter_documents_by_name(self._docs, name_patterns, match_type=match_type,
                                              ignore_case=ignore_case, glob_method=glob_method, inverse=inverse)

    def _task_filter_for_pos(self, required_pos, simplify_pos, inverse):
        self._docs = filter_for_pos(self._docs, required_pos=required_pos, simplify_pos=simplify_pos, inverse=inverse)
