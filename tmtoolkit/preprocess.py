# -*- coding: utf-8 -*-
from __future__ import division
import logging
import sys
import os
import string
import multiprocessing as mp
import atexit
from importlib import import_module
from collections import defaultdict, Counter
from copy import deepcopy
import pickle
import operator

import nltk

from .germalemma import GermaLemma
from .filter_tokens import filter_for_tokenpattern, filter_for_pos
from .dtm import create_sparse_dtm, get_vocab_and_terms
from .utils import require_listlike, require_dictlike, pickle_data, unpickle_file, \
    apply_to_mat_column, pos_tag_convert_penn_to_wn, simplified_pos, \
    flatten_list, tuplize, greedy_partitioning, ith_column


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.join(MODULE_PATH, 'data')
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


logger = logging.getLogger('tmtoolkit')


class TMPreproc(object):
    def __init__(self, docs=None, language='english', n_max_processes=None,
                 stopwords=None, punctuation=None, special_chars=None,
                 custom_tokenizer=None, custom_stemmer=None, custom_pos_tagger=None, custom_lemmata_dict=None):
        if docs is not None:
            require_dictlike(docs)

        self.n_max_workers = n_max_processes or mp.cpu_count()
        if self.n_max_workers < 1:
            raise ValueError('`n_max_processes` must be at least 1')

        logger.info('init with max. %d workers' % self.n_max_workers)

        self.tasks_queues = None
        self.results_queue = None
        self.workers = []
        self.n_workers = 0
        self._cur_workers_tokens = None
        self._cur_workers_ngrams = None
        self._cur_vocab = None
        self._cur_vocab_doc_freqs = None
        self.n_docs = len(docs) if docs is not None else None

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

        self.tokenizer = self._load_tokenizer(custom_tokenizer)
        self.stemmer = self._load_stemmer(custom_stemmer)

        # lemmata dictionary with POS -> word -> lemma mapping
        self.lemmata_dict = self._load_lemmata_dict(custom_lemmata_dict)

        self.pos_tagger, self.pos_tagset = self._load_pos_tagger(custom_pos_tagger)

        self.tokenized = False
        self.pos_tagged = False
        self.ngrams_generated = False
        self.ngrams_as_tokens = False

        if self.docs:
            self._setup_workers()

        atexit.register(self.shutdown_workers)

    def __del__(self):
        """destructor. shutdown all workers"""
        self.shutdown_workers()

    def shutdown_workers(self):
        logger.info('sending shutdown signal to workers')
        self._send_task_to_workers(None)

    def get_tokens(self, non_empty=True, with_pos_tags=False):
        self._require_tokens()

        if with_pos_tags:
            self._require_pos_tags()
            self._require_no_ngrams_as_tokens()

            tokens = self._workers_tokens
        else:
            tokens = {dl: ith_column(dt) for dl, dt in self._workers_tokens.items()}

        if non_empty:
            return {dl: dt for dl, dt in tokens.items() if dt}
        else:
            return tokens

    def get_ngrams(self, non_empty=True):
        self._require_ngrams()
        if non_empty:
            return {dl: dt for dl, dt in self._workers_ngrams.items() if dt}
        else:
            return self._workers_ngrams

    @property
    def tokens(self):
        return self.get_tokens()

    @property
    def tokens_with_pos_tags(self):
        return self.get_tokens(with_pos_tags=True)

    @property
    def vocabulary(self):
        self._require_tokens()

        if self._cur_vocab is not None:
            return self._cur_vocab

        vocab = set()
        for t in self.tokens.values():
            vocab |= set(t)

        self._cur_vocab = vocab

        return self._cur_vocab

    @property
    def vocabulary_abs_doc_frequency(self):
        self._require_tokens()

        if self._cur_vocab_doc_freqs is not None:
            return self._cur_vocab_doc_freqs

        # get document frequency from each worker
        self._send_task_to_workers('get_vocab_doc_freq')

        # sum up the worker's doc. frequencies
        self._cur_vocab_doc_freqs = sum([self.results_queue.get() for _ in range(self.n_workers)], Counter())

        return self._cur_vocab_doc_freqs

    @property
    def vocabulary_rel_doc_frequency(self):
        return {t: n/self.n_docs for t, n in self.vocabulary_abs_doc_frequency.items()}

    @property
    def ngrams(self):
        return self.get_ngrams()

    def save_state(self, picklefile):
        # attributes for this instance ("manager instance")
        logger.info('saving state to file `%s`' % picklefile)

        state_attrs = {}
        attr_blacklist = ('tokenizer', 'stemmer', 'lemmata_dict',
                          'tasks_queues', 'results_queue',
                          'n_max_workers', 'workers', 'n_workers')
        for attr in dir(self):
            if attr.startswith('_') or attr in attr_blacklist:
                continue
            classattr = getattr(type(self), attr, None)
            if classattr is not None and (callable(classattr) or isinstance(classattr, property)):
                continue
            state_attrs[attr] = getattr(self, attr)

        # worker states
        self._send_task_to_workers('get_state')
        worker_states = [self.results_queue.get() for _ in range(self.n_workers)]

        # save to pickle
        pickle_data({'manager_state': state_attrs, 'worker_states': worker_states}, picklefile)

        return self

    def load_state(self, picklefile):
        logger.info('loading state from file `%s`' % picklefile)

        state_data = unpickle_file(picklefile)

        if set(state_data.keys()) != {'manager_state', 'worker_states'}:
            raise ValueError('invalid data in state file `%s`' % picklefile)

        # load saved state attributes for this instance ("manager instance")
        for attr, val in state_data['manager_state'].items():
            setattr(self, attr, val)

        # recreate worker processes
        self.shutdown_workers()
        self._setup_workers(state_data['worker_states'])

        self._invalidate_workers_tokens()
        self._invalidate_workers_ngrams()

        return self

    @classmethod
    def from_state(cls, file, **init_kwargs):
        return cls(**init_kwargs).load_state(file)

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

    def tokenize(self):
        self._invalidate_workers_tokens()

        logger.info('tokenizing')
        self._send_task_to_workers('tokenize')
        self.tokenized = True

        return self

    def generate_ngrams(self, n, join=True, join_str=' ', reassign_tokens=False):
        self._require_tokens()
        self._invalidate_workers_ngrams()

        logger.info('generating ngrams')
        self._send_task_to_workers('generate_ngrams', n=n, join=join, join_str=join_str)

        if reassign_tokens:
            self.use_ngrams_as_tokens(join=False)

        self.ngrams_generated = True

        return self

    def use_ngrams_as_tokens(self, join=False, join_str=' '):
        self._require_ngrams()
        self._invalidate_workers_tokens()

        self._send_task_to_workers('use_ngrams_as_tokens', join=join, join_str=join_str)

        self.pos_tagged = False
        self.ngrams_as_tokens = True

        return self

    def transform_tokens(self, transform_fn):
        if not callable(transform_fn):
            raise ValueError('transform_fn must be callable')

        try:
            pickle.dumps(transform_fn)
        except (pickle.PicklingError, AttributeError):
            raise ValueError('transform_fn cannot be pickled')

        self._require_tokens()
        self._invalidate_workers_tokens()

        logger.info('transforming tokens')
        self._send_task_to_workers('transform_tokens', transform_fn=transform_fn)

        return self

    def tokens_to_lowercase(self):
        return self.transform_tokens(string.lower if sys.version_info[0] < 3 else str.lower)

    def stem(self):
        self._require_tokens()
        self._require_no_ngrams_as_tokens()

        self._invalidate_workers_tokens()

        logger.info('stemming tokens')
        self._send_task_to_workers('stem')

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

        self._invalidate_workers_tokens()
        logger.info('POS tagging tokens')
        self._send_task_to_workers('pos_tag')
        self.pos_tagged = True

        return self

    def lemmatize(self, use_dict=False, use_patternlib=False, use_germalemma=None):
        self._require_pos_tags()
        self._require_no_ngrams_as_tokens()

        self._invalidate_workers_tokens()

        logger.info('lemmatizing tokens')
        self._send_task_to_workers('lemmatize',
                                   pos_tagset=self.pos_tagset,
                                   use_dict=use_dict,
                                   use_patternlib=use_patternlib,
                                   use_germalemma=use_germalemma)

        return self

    def expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        self._require_no_pos_tags()
        self._require_no_ngrams_as_tokens()

        self._invalidate_workers_tokens()
        logger.info('expanding compound tokens')
        self._send_task_to_workers('expand_compound_tokens',
                                   split_chars=split_chars,
                                   split_on_len=split_on_len,
                                   split_on_casechange=split_on_casechange)

        return self

    def remove_special_chars_in_tokens(self):
        self._require_tokens()

        self._invalidate_workers_tokens()

        logger.info('removing special characters in tokens')
        self._send_task_to_workers('remove_special_chars_in_tokens', special_chars=self.special_chars)

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

            self._invalidate_workers_tokens()

            logger.info('cleaning tokens')
            self._send_task_to_workers('clean_tokens', tokens_to_remove=tokens_to_remove)

        return self

    def filter_for_token(self, search_token, ignore_case=False, remove_found_token=False):
        self.filter_for_tokenpattern(search_token, fixed=True, ignore_case=ignore_case,
                                     remove_found_token=remove_found_token)

        return self

    def filter_for_tokenpattern(self, tokpattern, fixed=False, ignore_case=False, remove_found_token=False):
        self._require_tokens()

        self._invalidate_workers_tokens()
        logger.info('filtering tokens for pattern')
        self._send_task_to_workers('filter_for_tokenpattern',
                                   tokpattern=tokpattern,
                                   fixed=fixed,
                                   ignore_case=ignore_case,
                                   remove_found_token=remove_found_token)

        return self

    def filter_for_pos(self, required_pos, simplify_pos=True):
        self._require_pos_tags()

        self._invalidate_workers_tokens()

        logger.info('filtering tokens for POS tag `%s`' % required_pos)
        self._send_task_to_workers('filter_for_pos',
                                   required_pos=required_pos,
                                   pos_tagset=self.pos_tagset,
                                   simplify_pos=simplify_pos)

        return self

    def remove_tokens_by_doc_frequency(self, which, df_threshold, absolute=False, save_orig_tokens=False):
        if which not in ('common', 'uncommon'):
            raise ValueError('`which` must be "common" or "uncommon"')

        if absolute:
            if not type(df_threshold) is int and 1 <= df_threshold <= self.n_docs:
                raise ValueError('`df_threshold` must be integer in range [1, %d]' % self.n_docs)
        else:
            if not 0 <= df_threshold <= 1:
                raise ValueError('`df_threshold` must be in range [0, 1]')

        comp = operator.ge if which == 'common' else operator.le

        logger.info('removing %s tokens by document frequency %s or equal than %f (%s value)'
                    % (which, 'greater' if which == 'common' else 'less',
                       df_threshold, 'absolute' if absolute else 'relative'))

        doc_freqs = self.vocabulary_abs_doc_frequency if absolute else self.vocabulary_rel_doc_frequency
        blacklist = set(t for t, f in doc_freqs.items() if comp(f, df_threshold))

        if blacklist:
            self._invalidate_workers_tokens()

            logger.debug('will remove the following %d tokens: %s' % (len(blacklist), blacklist))
            self._send_task_to_workers('clean_tokens', tokens_to_remove=blacklist, save_orig_tokens=save_orig_tokens)

        return self

    def remove_common_tokens(self, df_threshold, absolute=False, save_orig_tokens=False):
        return self.remove_tokens_by_doc_frequency('common', df_threshold=df_threshold, absolute=absolute,
                                                   save_orig_tokens=save_orig_tokens)

    def remove_uncommon_tokens(self, df_threshold, absolute=False, save_orig_tokens=False):
        return self.remove_tokens_by_doc_frequency('uncommon', df_threshold=df_threshold, absolute=absolute,
                                                   save_orig_tokens=save_orig_tokens)

    def reset_filter(self):
        self._require_tokens()
        self._invalidate_workers_tokens()

        logger.info('reseting filters')
        self._send_task_to_workers('reset_filter')

        return self

    def get_dtm(self, from_ngrams=False):
        self._require_tokens()

        if from_ngrams:
            self._require_ngrams()
            tok = self.ngrams
        else:
            tok = self.tokens

        logger.info('generating DTM')
        vocab, doc_labels, docs_terms, dtm_alloc_size = get_vocab_and_terms(tok)
        dtm = create_sparse_dtm(vocab, doc_labels, docs_terms, dtm_alloc_size)

        return doc_labels, vocab, dtm

    def _load_stemmer(self, custom_stemmer=None):
        logger.info('loading stemmer')

        if custom_stemmer:
            stemmer = custom_stemmer
        else:
            stemmer = nltk.stem.SnowballStemmer(self.language)

        if not hasattr(stemmer, 'stem') or not callable(stemmer.stem):
            raise ValueError('stemmer must have a callable method `stem`')

        return stemmer

    def _load_tokenizer(self, custom_tokenizer):
        logger.info('loading tokenizer')

        if custom_tokenizer:
            tokenizer = custom_tokenizer
        else:
            tokenizer = lambda x: nltk.tokenize.word_tokenize(x, self.language)

        if not callable(tokenizer):
            raise ValueError('tokenizer must be callable')

        return tokenizer

    def _load_pos_tagger(self, custom_pos_tagger=None):
        logger.info('loading POS tagger')

        pos_tagset = None
        if custom_pos_tagger:
            tagger = custom_pos_tagger
        else:
            try:
                picklefile = os.path.join(DATAPATH, self.language, POS_TAGGER_PICKLE)
                tagger = unpickle_file(picklefile)
                if self.language == u'german':
                    pos_tagset = 'stts'
            except IOError:
                tagger = GenericPOSTagger
                pos_tagset = GenericPOSTagger.tag_set

        if not hasattr(tagger, 'tag') or not callable(tagger.tag):
            raise ValueError("pos_tagger must have a callable attribute `tag`")

        return tagger, pos_tagset

    def _load_lemmata_dict(self, custom_lemmata_dict=None):
        logger.info('loading lemmata dictionary')

        if custom_lemmata_dict:
            return custom_lemmata_dict
        else:
            picklefile = os.path.join(DATAPATH, self.language, LEMMATA_PICKLE)
            try:
                unpickled_obj = unpickle_file(picklefile)
                return unpickled_obj
            except IOError:
                logger.error('could not load lemmata dict from `%s`' % picklefile)
                return None

    def _setup_workers(self, initial_states=None):
        """
        Create worker processes and queues. Distribute the work evenly across worker processes. Optionally
        send initial states defined in list `initial_states` to each worker process.
        """
        if initial_states is not None:
            require_listlike(initial_states)

        self.tasks_queues = []
        self.results_queue = mp.Queue()
        self.workers = []

        common_kwargs = dict(tokenizer=self.tokenizer,
                             stemmer=self.stemmer,
                             lemmata_dict=self.lemmata_dict,
                             pos_tagger=self.pos_tagger)

        if initial_states is not None:
            logger.info('setting up %d worker processes with initial states' % len(initial_states))

            for i_worker, w_state in enumerate(initial_states):
                task_q = mp.JoinableQueue()
                w = _PreprocWorker(w_state.pop('docs'), self.language, task_q, self.results_queue,
                                   name='_PreprocWorker#%d' % i_worker, **common_kwargs)
                w.start()

                task_q.put(('set_state', w_state))

                self.workers.append(w)
                self.tasks_queues.append(task_q)

            [q.join() for q in self.tasks_queues]

        else:
            # distribute work evenly across the worker processes
            # we assume that the longer a document is, the longer the processing time for it is
            # hence we distribute the work evenly by document length
            docs_and_lengths = {dl: len(dt) for dl, dt in self.docs.items()}
            docs_per_worker = greedy_partitioning(docs_and_lengths, k=self.n_max_workers)

            logger.info('setting up %d worker processes' % len(docs_per_worker))

            for i_worker, doc_labels in enumerate(docs_per_worker):
                if not doc_labels: continue
                task_q = mp.JoinableQueue()
                w_docs = {dl: self.docs.get(dl) for dl in doc_labels}

                w = _PreprocWorker(w_docs, self.language, task_q, self.results_queue,
                                   name='_PreprocWorker#%d' % i_worker, **common_kwargs)
                w.start()

                self.workers.append(w)
                self.tasks_queues.append(task_q)

        self.n_workers = len(self.workers)

    def _send_task_to_workers(self, task, **kwargs):
        if not (self.tasks_queues and self.workers):
            return

        shutdown = task is None
        task_item = None if shutdown else (task, kwargs)

        logger.debug('sending task `%s` to all workers' % task)

        [q.put(task_item) for q in self.tasks_queues]
        [q.join() for q in self.tasks_queues]

        if shutdown:
            logger.debug('shutting down worker processes')
            self.tasks_queues = None
            [w.join() for w in self.workers]
            self.workers = []
            self.n_workers = 0

    def _get_results_from_workers(self, task, **kwargs):
        logger.debug('getting results for task `%s` from all workers' % task)
        self._send_task_to_workers(task, **kwargs)

        res = {}
        for _ in range(self.n_docs):
            pair = self.results_queue.get()
            if pair:
                res[pair[0]] = pair[1]

        return res

    @property
    def _workers_tokens(self):
        if self._cur_workers_tokens is not None:
            return self._cur_workers_tokens

        self._cur_workers_tokens = {dl: dt for dl, dt in self._get_results_from_workers('get_tokens').items()}

        return self._cur_workers_tokens

    def _invalidate_workers_tokens(self):
        self._cur_workers_tokens = None
        self._cur_vocab = None
        self._cur_vocab_doc_freqs = None

    @property
    def _workers_ngrams(self):
        if self._cur_workers_ngrams is not None:
            return self._cur_workers_ngrams

        self._cur_workers_ngrams = {dl: dt for dl, dt in self._get_results_from_workers('get_ngrams').items()}

        return self._cur_workers_ngrams

    def _invalidate_workers_ngrams(self):
        self._cur_workers_ngrams = None
        self._cur_vocab = None
        self._cur_vocab_doc_freqs = None

    def _require_tokens(self):
        if not self.tokenized:
            raise ValueError("documents must be tokenized before this operation")

    def _require_pos_tags(self):
        self._require_tokens()

        if not self.pos_tagged:
            raise ValueError("tokens must be POS-tagged before this operation")

    def _require_ngrams(self):
        if not self.ngrams_generated:
            raise ValueError("ngrams must be created before this operation")

    def _require_no_ngrams_as_tokens(self):
        if self.ngrams_as_tokens:
            raise ValueError("ngrams are used as tokens -- this is not possible for this operation")

    def _require_no_pos_tags(self):
        self._require_tokens()

        if self.pos_tagged:
            raise ValueError("tokens shall not be POS-tagged before this operation")


class _PreprocWorker(mp.Process):
    def __init__(self, docs, language, tasks_queue, results_queue, tokenizer, stemmer, lemmata_dict, pos_tagger,
                 group=None, target=None, name=None, args=(), kwargs=None):
        super(_PreprocWorker, self).__init__(group, target, name, args, kwargs or {})
        logger.debug('worker `%s`: init' % name)
        logger.debug('worker `%s`: docs = ' % set(docs.keys()))
        self.docs = docs
        self.language = language
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue

        # set a tokenizer
        self.tokenizer = tokenizer      # self.tokenizer is a function with a document text as argument

        # set a stemmer
        self.stemmer = stemmer                # stemmer instance (must have a callable attribute `stem`)

        # set a POS tagger
        self.pos_tagger = pos_tagger             # POS tagger instance (must have a callable attribute `tag`)

        self.lemmata_dict = lemmata_dict
        self.pattern_module = None          # dynamically loaded CLiPS pattern library module
        self.germalemma = None              # GermaLemma instance
        self.wordnet_lemmatizer = None      # nltk.stem.WordNetLemmatizer instance

        self._tokens = {}             # tokens for this worker at the current processing stage. dict with document label -> tokens list
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
        self._put_items_in_results_queue(self._tokens)

    def _task_get_ngrams(self):
        self._put_items_in_results_queue(self._ngrams)

    def _task_get_vocab_doc_freq(self):
        counts = Counter()
        for dt in self._tokens.values():
            counts.update(set(ith_column(dt)))
        self.results_queue.put(counts)

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

    def _task_set_state(self, **state):
        logger.debug('worker `%s`: setting state' % self.name)

        for attr, val in state.items():
            setattr(self, attr, val)

    def _task_tokenize(self):
        self._tokens = {dl: tuplize(self.tokenizer(txt)) for dl, txt in self.docs.items()}

    def _task_generate_ngrams(self, n, join=True, join_str=' '):
        self._ngrams = {dl: create_ngrams(ith_column(dt), n=n, join=join, join_str=join_str)
                        for dl, dt in self._tokens.items()}

    def _task_use_ngrams_as_tokens(self, join=False, join_str=' '):
        if join:
            new_tok = {dl: tuplize([join_str.join(g_tuple) for g_tuple in dg])
                       for dl, dg in self._ngrams.items()}
        else:
            new_tok = {dl: tuplize(dg) for dl, dg in self._ngrams.items()}

        self._tokens = new_tok

    def _task_transform_tokens(self, transform_fn):
        self._tokens = {dl: apply_to_mat_column(dt, 0, transform_fn) if dt else []
                        for dl, dt in self._tokens.items()}

    def _task_stem(self):
        self._tokens = {dl: apply_to_mat_column(dt, 0, lambda t: self.stemmer.stem(t)) if dt else []
                        for dl, dt in self._tokens.items()}

    def _task_pos_tag(self):
        self._tokens = {dl: apply_to_mat_column(dt, 0, self.pos_tagger.tag, map_func=False, expand=True) if dt else []
                        for dl, dt in self._tokens.items()}

    def _task_lemmatize(self, pos_tagset, use_dict=False, use_patternlib=False, use_germalemma=None):
        tmp_lemmata = defaultdict(list)

        if use_germalemma is None and self.language == 'german':
            use_germalemma = True

        if use_germalemma:
            if not self.germalemma:
                if type(self.lemmata_dict) is not tuple or len(self.lemmata_dict) != 2:
                    raise ValueError("Unexpected data type or length for `self.lemmata_dict`")
                lemmata_dict, lemmata_lower_dict = self.lemmata_dict
                self.germalemma = GermaLemma(lemmata=lemmata_dict, lemmata_lower=lemmata_lower_dict)

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

    def _task_clean_tokens(self, tokens_to_remove, save_orig_tokens=False):
        if save_orig_tokens:
            self._save_orig_tokens()

        if type(tokens_to_remove) is not set:   # using a set is much faster than other sequence types for "in" tests
            tokens_to_remove = set(tokens_to_remove)

        self._tokens = {dl: [t for t in dt if t[0] not in tokens_to_remove]
                        for dl, dt in self._tokens.items()}

    def _task_filter_for_tokenpattern(self, tokpattern, fixed=False, ignore_case=False, remove_found_token=False):
        self._save_orig_tokens()
        self._tokens = filter_for_tokenpattern(self._tokens, tokpattern, fixed=fixed, ignore_case=ignore_case,
                                               remove_found_token=remove_found_token, remove_empty_docs=False)

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


class GenericPOSTagger(object):
    tag_set = 'penn'

    @staticmethod
    def tag(tokens):
        return nltk.pos_tag(tokens)


def str_multisplit(s, split_chars):
    parts = [s]
    for c in split_chars:
        parts_ = []
        for p in parts:
            parts_.extend(p.split(c))
        parts = parts_

    return parts


def expand_compound_token(t, split_chars=('-',), split_on_len=2, split_on_casechange=False):
    #print('expand_compound_token', t)
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
