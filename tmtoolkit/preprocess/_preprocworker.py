"""
Preprocessing worker class for parallel text processing.
"""

import multiprocessing as mp
import logging

import numpy as np
import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc, Token

from ._docfuncs import (
    ngrams, vocabulary, vocabulary_counts, doc_frequencies, sparse_dtm, compact_documents, glue_tokens, doc_labels,
    expand_compounds, clean_tokens, filter_tokens_by_mask, filter_tokens, filter_tokens_with_kwic, filter_documents,
    filter_documents_by_name, filter_for_pos, transform, remove_chars, lemmatize, to_lowercase,
    _build_kwic, _filtered_doc_tokens, _filtered_doc_arr, _init_doc, _replace_doc_tokens
)


logger = logging.getLogger('tmtoolkit')
logger.addHandler(logging.NullHandler())


class PreprocWorker(mp.Process):
    def __init__(self, worker_id, nlp, language, tasks_queue, results_queue, shutdown_event, worker_error_event,
                 group=None, target=None, name=None, args=(), kwargs=None):
        super().__init__(group, target, name, args, kwargs or {}, daemon=True)
        logger.debug('worker `%s`: init with worker ID %d' % (name, worker_id))
        self.worker_id = worker_id
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue
        self.shutdown_event = shutdown_event
        self.worker_error_event = worker_error_event

        self.language = language

        self.nlp = nlp   # can be None when later using set_state
        if nlp:
            self.tagger = dict(nlp.pipeline).get('tagger', None)
        else:
            self.tagger = None

        self._docs = []               # SpaCy documents

        self._std_attrs = ['lemma', 'whitespace']
        self._metadata_attrs = {}     # metadata key -> default value
        self._ngrams = []             # generated ngrams as list of token strings

    def run(self):
        logger.debug('worker `%s`: run' % self.name)

        while not self.shutdown_event.is_set():  # accept tasks from queue until shutdown event
            q_item = self.tasks_queue.get()

            if q_item is None:
                break

            next_task, task_kwargs = q_item
            logger.debug('worker `%s`: received task `%s`' % (self.name, next_task))

            exec_task_fn = getattr(self, '_task_' + next_task)
            if exec_task_fn:
                try:
                    exec_task_fn(**task_kwargs)
                except Exception as exc:
                    logger.error('worker `%s`: an exception occured: "%s"' % (self.name, str(exc)))
                    self.tasks_queue.task_done()
                    self.worker_error_event.set()   # signal worker error
                    self.shutdown_event.set()    # signal shutdown for all workers
                    raise exc                    # re-raise exception

                self.tasks_queue.task_done()
            else:
                self.tasks_queue.task_done()
                self.worker_error_event.set()  # signal worker error
                self.shutdown_event.set()        # signal shutdown for all workers
                raise NotImplementedError("Task not implemented: `%s`" % next_task)

        # normal shutdown
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
                resdoc = {'token': _filtered_doc_tokens(doc)}

            for meta_key in self._std_attrs:
                assert meta_key not in resdoc
                if meta_key == 'whitespace':
                    resdoc[meta_key] = _filtered_doc_arr([bool(t.whitespace_) for t in doc], doc)
                else:
                    resdoc[meta_key] = _filtered_doc_arr([getattr(t, meta_key + '_') for t in doc], doc)

            for meta_key in self._metadata_attrs.keys():
                k = 'meta_' + meta_key
                assert k not in resdoc
                resdoc[k] = _filtered_doc_arr([getattr(t._, k) for t in doc], doc)

            if as_dict:
                res[doc._.label] = resdoc
            else:
                res.append(resdoc)

        return res

    @property
    def _tokens(self):
        return [_filtered_doc_tokens(doc) for doc in self._docs]

    @property
    def _doc_labels(self):
        return doc_labels(self._docs)

    @property
    def _tokens_meta(self):
        return self._get_tokens_with_metadata(as_dict=False, only_metadata=True)

    def _remove_metadata(self, key):
        if key in self._metadata_attrs:
            Token.remove_extension('meta_' + key)
            del self._metadata_attrs[key]

    def _clear_metadata(self, pos=True):
        keys = list(self._metadata_attrs.keys())
        for k in keys:
            self._remove_metadata(k)

        assert len(self._metadata_attrs) == 0

        if pos and 'pos' in self._std_attrs:
            self._std_attrs.remove('pos')

    def _task_init(self, docs, docs_are_tokenized, enable_vectors):
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

                    if doc_i == 0:
                        if is_std:
                            if k not in self._std_attrs:
                                self._std_attrs.append(k)
                        else:
                            meta_k = k[5:]       # strip "meta_"
                            if meta_k not in self._metadata_attrs:
                                self._metadata_attrs[meta_k] = None   # cannot infer correct default value here
                                Token.set_extension(k, default=None)

                    assert len(new_doc) == len(metadata)
                    for t, v in zip(new_doc, metadata):
                        if is_std:
                            attr = k + '_'
                            obj = t
                        else:
                            attr = k
                            obj = t._

                        setattr(obj, attr, v)

                self._docs.append(new_doc)
        else:
            # directly tokenize documents
            logger.info('tokenizing %d documents' % len(docs))

            if enable_vectors:
                self._docs = [self.nlp(d) for d in docs.values()]
            else:
                self._docs = [self.nlp.make_doc(d) for d in docs.values()]

        # set attributes for transformed text and filter mask
        # will use user_data directly because this is much faster than <token>._.<attr>
        self._init_docs(docs.keys())

    def _task_get_doc_labels(self):
        self.results_queue.put(self._doc_labels)

    def _task_get_tokens(self):
        # tokens with metadata
        self.results_queue.put(self._get_tokens_with_metadata())

    def _task_get_spacydocs(self):
        # spaCy documents
        self.results_queue.put(dict(zip(self._doc_labels, self._docs)))

    def _task_get_doc_vectors(self):
        # document vectors
        self.results_queue.put(dict(zip(self._doc_labels, (d.vector for d in self._docs))))

    def _task_get_token_vectors(self):
        # document token vectors
        self.results_queue.put(
            {dl: np.vstack([t.vector for t in doc])
             for dl, doc in zip(self._doc_labels, self._docs)}
        )

    def _task_replace_tokens(self, tokens):
        assert set(tokens.keys()) == set(self._doc_labels)
        for dl, new_tok in tokens.items():
            doc = self._docs[self._doc_labels.index(dl)]
            _replace_doc_tokens(doc, new_tok)

    def _task_get_available_metadata_keys(self):
        self.results_queue.put(self._std_attrs + list(self._metadata_attrs.keys()))

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

        # serialize SpaCy docs
        #
        # tried out several approaches for both serializing and de-serializing:
        #
        # 1. return self._docs as is
        #    -> copies lot of data, takes a lot of time on main process
        # 2. return serialization of each Doc via `to_bytes()` method *and* serialization of full vocab data
        #    -> serialization and de-serial. of vocab seems slow and unnecessary
        # 3. return serialization of each Doc via `to_bytes()` method and serialization of vocab excluding vectors
        #    -> seems to be the fastest so far although de-serial. seems still quite slow
        #
        # see _task_set_state() method below
        #

        state = {
            'docs_bytes': [doc.to_bytes() for doc in self._docs],
            'nlp_bytes': self.nlp.to_bytes(exclude=['vocab']),
            'tagger_bytes': self.tagger.to_bytes() if self.tagger is not None else None,
            'vocab_bytes': self.nlp.vocab.to_bytes(),
            'doc_labels': self._doc_labels,               # for TMPreproc master process
        }

        other_attrs = (
            '_ngrams',
            '_std_attrs',
            '_metadata_attrs'
        )

        state.update({attr: getattr(self, attr) for attr in other_attrs})
        logger.debug('worker `%s`: got state with %d items' % (self.name, len(state)))
        self.results_queue.put(state)

    def _task_set_state(self, **state):
        logger.debug('worker `%s`: setting state' % self.name)

        for key, default in state['_metadata_attrs'].items():
            Token.set_extension('meta_' + key, default=default)

        # de-serialize SpaCy docs
        lang_cls = spacy.util.get_lang_class(self.language)
        vocab = Vocab().from_bytes(state.pop('vocab_bytes'))
        self.nlp = lang_cls(vocab).from_bytes(state.pop('nlp_bytes'))
        tagger_bytes = state.pop('tagger_bytes')
        if tagger_bytes is not None:
            self.tagger = spacy.pipeline.Tagger(self.nlp.vocab).from_bytes()
            self.nlp.pipeline = [('tagger', self.tagger)]
        else:
            self.tagger = None

        self._docs = []
        for doc_bytes in state.pop('docs_bytes'):
            doc = Doc(self.nlp.vocab).from_bytes(doc_bytes)

            # document tensor array and user_data arrays may only be immutable "views" -> create mutable copies
            if not doc.tensor.flags.owndata:
                doc.tensor = doc.tensor.copy()

            for k, docdata in doc.user_data.items():
                if isinstance(docdata, np.ndarray) and not docdata.flags.owndata:
                    doc.user_data[k] = docdata.copy()

            self._docs.append(doc)

        for attr, val in state.items():
            setattr(self, attr, val)

    def _task_add_metadata_per_token(self, key, data, default):
        logger.debug('worker `%s`: adding metadata per token' % self.name)

        attr_name = 'meta_' + key
        Token.set_extension(attr_name, default=default)

        for doc in self._docs:
            for t, tok_text in zip(doc, _filtered_doc_tokens(doc)):
                setattr(t._, attr_name, data.get(tok_text, default))

        if key not in self._metadata_attrs.keys():
            self._metadata_attrs[key] = default

    def _task_add_metadata_per_doc(self, key, data, default):
        logger.debug('worker `%s`: adding metadata per document' % self.name)

        attr_name = 'meta_' + key
        Token.set_extension(attr_name, default=default)

        for doc in self._docs:
            meta_vals = data.get(doc._.label, [default] * len(doc))
            assert sum(doc.user_data['mask']) == len(meta_vals)
            for t, v, m in zip(doc, meta_vals, doc.user_data['mask']):
                if m:
                    setattr(t._, attr_name, v)

        if key not in self._metadata_attrs:
            self._metadata_attrs[key] = default

    def _task_remove_metadata(self, key):
        logger.debug('worker `%s`: removing metadata column' % self.name)
        self._remove_metadata(key)

    def _task_generate_ngrams(self, n):
        tokens = [[t.strip() for t in doc if t.strip()] for doc in self._tokens]   # make sure to remove line breaks
        self._ngrams = ngrams(tokens, n, join=False)

    def _task_use_joined_ngrams_as_tokens(self, join_str):
        doc_labels = self._doc_labels
        joined_tokens = [list(map(lambda g: join_str.join(g), dngrams)) for dngrams in self._ngrams]

        self._docs = [Doc(self.nlp.vocab, words=tok) for tok in joined_tokens]
        self._init_docs(doc_labels)

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

        # reset ngrams as they're used as normal tokens now
        self._ngrams = {}

    def _task_transform_tokens(self, transform_fn, **kwargs):
        for doc, new_tok in zip(self._docs, transform(self._tokens, transform_fn, **kwargs)):
            _replace_doc_tokens(doc, new_tok)

    def _task_tokens_to_lowercase(self):
        for doc, new_tok in zip(self._docs, to_lowercase(self._tokens)):
            _replace_doc_tokens(doc, new_tok)

    def _task_remove_chars(self, chars):
        for doc, new_tok in zip(self._docs, remove_chars(self._tokens, chars=chars)):
            _replace_doc_tokens(doc, new_tok)

    def _task_pos_tag(self):
        if 'pos' not in self._std_attrs:
            for doc in self._docs:
                # this will be done for all tokens in the document, i.e. also for masked tokens,
                # unless "compact" is called before
                if self.tagger is not None:
                    self.tagger(doc)
            self._std_attrs.append('pos')

    def _task_lemmatize(self):
        docs_lemmata = lemmatize(self._docs)

        for doc, new_tok in zip(self._docs, docs_lemmata):
            _replace_doc_tokens(doc, new_tok)

    def _task_expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        self._docs = expand_compounds(self._docs, split_chars=split_chars, split_on_len=split_on_len,
                                      split_on_casechange=split_on_casechange)

        # do reset because meta data doesn't match any more:
        self._clear_metadata()

    def _task_clean_tokens(self, tokens_to_remove, remove_punct, remove_empty,
                           remove_shorter_than, remove_longer_than, remove_numbers):
        clean_tokens(self._docs, remove_punct=remove_punct,
                     remove_stopwords=tokens_to_remove, remove_empty=remove_empty,
                     remove_shorter_than=remove_shorter_than, remove_longer_than=remove_longer_than,
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

    def _task_compact_documents(self):
        self._docs = compact_documents(self._docs)

    def _task_filter_tokens_by_mask(self, mask, inverse):
        mask_list = [mask[dl] for dl in self._doc_labels]
        filter_tokens_by_mask(self._docs, mask_list, inverse=inverse)

    def _task_filter_tokens(self, search_tokens, match_type, ignore_case, glob_method, inverse, by_meta):
        if by_meta:
            by_meta = 'meta_' + by_meta

        filter_tokens(self._docs, search_tokens, match_type=match_type, ignore_case=ignore_case,
                      glob_method=glob_method, inverse=inverse, by_meta=by_meta)

    def _task_filter_tokens_with_kwic(self, search_tokens, context_size, match_type, ignore_case,
                                      glob_method, inverse):
        filter_tokens_with_kwic(self._docs, search_tokens,
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

    def _init_docs(self, doc_labels):
        for doc in self._docs:
            _init_doc(doc)

        assert len(doc_labels) == len(self._docs)
        for dl, doc in zip(doc_labels, self._docs):
            doc._.label = dl
