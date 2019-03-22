"""
Parallel text processing with `TMPreproc` class.
"""

import os
import string
import multiprocessing as mp
import atexit
from collections import Counter, defaultdict
from copy import deepcopy
import pickle
import operator

import numpy as np
import pandas as pd
import nltk

from .. import logger
from ..corpus import Corpus
from ..bow.dtm import create_sparse_dtm
from ..bow.bow_stats import get_doc_frequencies
from ..utils import require_listlike_or_set, require_dictlike, pickle_data, unpickle_file, greedy_partitioning, flatten_list
from tmtoolkit.utils import empty_chararray
from ._preprocworker import PreprocWorker


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.normpath(os.path.join(MODULE_PATH, '..', 'data'))
POS_TAGGER_PICKLE = 'pos_tagger.pickle'
LEMMATA_PICKLE = 'lemmata.pickle'


class TMPreproc(object):
    def __init__(self, docs, language='english', n_max_processes=None,
                 stopwords=None, punctuation=None, special_chars=None,
                 custom_tokenizer=None, custom_stemmer=None, custom_pos_tagger=None):
        if isinstance(docs, Corpus):
            docs = docs.docs

        if docs is not None:
            require_dictlike(docs)

        self.n_max_workers = n_max_processes or mp.cpu_count()
        if self.n_max_workers < 1:
            raise ValueError('`n_max_processes` must be at least 1')

        logger.info('init with max. %d workers' % self.n_max_workers)

        self.tasks_queues = None
        self.results_queue = None
        self.workers = []
        self.docs2workers = {}
        self.n_workers = 0
        self._cur_doc_labels = None
        self._cur_workers_tokens = None
        self._cur_workers_vocab = None
        self._cur_workers_ngrams = None
        self._cur_vocab_counts = None
        self._cur_dtm = None

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

        self.pos_tagger, self.pos_tagset = self._load_pos_tagger(custom_pos_tagger)

        self.ngrams_as_tokens = False

        if docs is not None:
            self._setup_workers(docs=docs)

        atexit.register(self.shutdown_workers)

    def __del__(self):
        """destructor. shutdown all workers"""
        self.shutdown_workers()

    def __str__(self):
        return 'TMPreproc with %d documents' % self.n_docs

    @property
    def n_docs(self):
        return len(self.doc_labels)

    @property
    def n_tokens(self):
        return sum(self.doc_lengths.values())

    @property
    def doc_labels(self):
        if self._cur_doc_labels is None:
            self._cur_doc_labels = np.sort(flatten_list(self._get_results_seq_from_workers('get_doc_labels')))

        return self._cur_doc_labels if len(self._cur_doc_labels) > 0 else empty_chararray()

    @property
    def doc_lengths(self):
        tok = self.tokens
        return dict(zip(tok.keys(), map(len, tok.values())))

    @property
    def tokens(self):
        return self.get_tokens(with_metadata=False)

    @property
    def tokens_with_metadata(self):
        return self.get_tokens()

    @property
    def tokens_dataframe(self):
        tokens = self.get_tokens(non_empty=True)
        dfs = []
        for dl, df in tokens.items():
            df = df.copy()
            df['doc'] = dl
            df['position'] = np.arange(len(df))
            dfs.append(df)

        if dfs:
            res = pd.concat(dfs, ignore_index=True)
        else:
            res = pd.DataFrame({'doc': [], 'position': [], 'token': []})

        return res.set_index(['doc', 'position']).sort_index()

    @property
    def tokens_with_pos_tags(self):
        self._require_pos_tags()
        return {dl: df.loc[:, ['token', 'meta_pos']] for dl, df in self.get_tokens().items()}

    @property
    def vocabulary(self):
        return self.get_vocabulary()

    @property
    def vocabulary_counts(self):
        if self._cur_vocab_counts is not None:
            return self._cur_vocab_counts

        # get vocab counts
        workers_vocab_counts = self._get_results_seq_from_workers('get_vocab_counts')

        # sum up the worker's doc. frequencies
        self._cur_vocab_counts = sum(map(Counter, workers_vocab_counts), Counter())

        return self._cur_vocab_counts

    @property
    def vocabulary_abs_doc_frequency(self):
        return dict(zip(self.vocabulary, get_doc_frequencies(self.dtm)))

    @property
    def vocabulary_rel_doc_frequency(self):
        return dict(zip(self.vocabulary, get_doc_frequencies(self.dtm, proportions=True)))

    @property
    def ngrams_generated(self):
        return len(self.ngrams) > 0

    @property
    def ngrams(self):
        return self.get_ngrams()

    @property
    def pos_tagged(self):
        meta_keys = self.get_available_metadata_keys()
        return 'pos' in meta_keys

    @property
    def dtm(self):
        return self.get_dtm()

    def shutdown_workers(self):
        try:   # may cause exception when the logger is actually already destroyed
            logger.info('sending shutdown signal to workers')
        except: pass
        self._send_task_to_workers(None)

    def save_state(self, picklefile):
        # attributes for this instance ("manager instance")
        logger.info('saving state to file `%s`' % picklefile)

        # save to pickle
        pickle_data(self._create_state_object(deepcopy_attrs=False), picklefile)

        return self

    def load_state(self, file_or_stateobj):
        if isinstance(file_or_stateobj, str):
            logger.info('loading state from file `%s`' % file_or_stateobj)
            state_data = unpickle_file(file_or_stateobj)
        else:
            logger.info('loading state from object')
            state_data = file_or_stateobj

        if set(state_data.keys()) != {'manager_state', 'worker_states'}:
            raise ValueError('invalid data in state object')

        # load saved state attributes for this instance ("manager instance")
        for attr, val in state_data['manager_state'].items():
            setattr(self, attr, val)

        # recreate worker processes
        self.shutdown_workers()
        self._setup_workers(initial_states=state_data['worker_states'])

        self._invalidate_docs_info()
        self._invalidate_workers_tokens()
        self._invalidate_workers_ngrams()

        return self

    def load_tokens(self, tokens):
        """
        Load tokens `tokens` into TMPreproc in the same format as they are returned by `self.tokens`, i.e. as dict
        with document label -> terms array mapping.
        """
        require_dictlike(tokens)

        logger.info('loading tokens of %d documents' % len(tokens))

        # recreate worker processes
        self.shutdown_workers()
        self._setup_workers(tokens, docs_are_tokenized=True)

        self._invalidate_docs_info()
        self._invalidate_workers_tokens()
        self._invalidate_workers_ngrams()

        return self

    def load_tokens_dataframe(self, tokendf):
        """
        Load tokens dataframe `tokendf` into TMPreproc in the same format as they are returned by `self.tokens_frame`,
        i.e. as data frame with hierarchical indices "doc" and "position" and at least a column "token" plus optional
        columns like "meta_pos", etc.
        """
        if not isinstance(tokendf, pd.DataFrame):
            raise ValueError('`tokendf` must be a pandas DataFrame')

        ind_names = tokendf.index.names

        if set(ind_names) != {'doc', 'position'}:
            raise ValueError('`tokendf` must have hierarchical indices "doc" and "position"')

        if 'token' not in tokendf.columns:
            raise ValueError('`tokendf` must contain a column named "token"')

        # convert big dataframe to dict of document token data frames to be used in load_tokens
        tokens = {}
        for dl, doc_df in tokendf.groupby(level=0):
            doc_df = doc_df.reset_index()
            tokens[dl] = doc_df.loc[:, doc_df.columns.difference(ind_names)]

        return self.load_tokens(tokens)

    def __copy__(self):
        """
        Copy a TMPreproc object including all its present state (tokens, meta data, etc.).
        Performs a deep copy.
        """
        return self.copy()

    def __deepcopy__(self, memodict=None):
        """
        Copy a TMPreproc object including all its present state (tokens, meta data, etc.).
        Performs a deep copy.
        """
        return self.copy()

    def copy(self):
        """
        Copy a TMPreproc object including all its present state (tokens, meta data, etc.).
        Performs a deep copy.
        """
        return TMPreproc.from_state(self._create_state_object(deepcopy_attrs=True))

    @classmethod
    def from_state(cls, file_or_stateobj, **init_kwargs):
        if 'docs' in init_kwargs.keys():
            raise ValueError('`docs` cannot be passed as argument when loading state')
        init_kwargs['docs'] = None

        return cls(**init_kwargs).load_state(file_or_stateobj)

    @classmethod
    def from_tokens(cls, tokens, **init_kwargs):
        if 'docs' in init_kwargs.keys():
            raise ValueError('`docs` cannot be passed as argument when loading tokens')
        init_kwargs['docs'] = None

        return cls(**init_kwargs).load_tokens(tokens)

    @classmethod
    def from_tokens_dataframe(cls, tokensdf, **init_kwargs):
        if 'docs' in init_kwargs.keys():
            raise ValueError('`docs` cannot be passed as argument when loading token dataframes')
        init_kwargs['docs'] = None

        return cls(**init_kwargs).load_tokens_dataframe(tokensdf)

    def get_tokens(self, non_empty=False, with_metadata=True):
        tokens = self._workers_tokens

        if not with_metadata:
            tokens = {dl: df.token.values.astype(np.str) for dl, df in tokens.items()}

        if non_empty:
            return {dl: dt for dl, dt in tokens.items() if len(dt) > 0}
        else:
            return tokens

    def get_vocabulary(self):
        """
        Return the vocabulary, i.e. the list of unique words across all documents, as sorted NumPy array.
        """
        nonempty_vocab = [v for v in self._workers_vocab if v is not None and len(v) > 0]

        return np.sort(np.unique(np.concatenate(nonempty_vocab)) if nonempty_vocab else empty_chararray())

    def get_ngrams(self, non_empty=False):
        if non_empty:
            return {dl: dt for dl, dt in self._workers_ngrams.items() if len(dt) > 0}
        else:
            return self._workers_ngrams

    def get_available_metadata_keys(self):
        keys = self._get_results_seq_from_workers('get_available_metadata_keys')

        return set(np.unique(np.concatenate(keys))) if keys else set()

    def add_stopwords(self, stopwords):
        require_listlike_or_set(stopwords)
        self.stopwords += stopwords

        return self

    def add_punctuation(self, punctuation):
        require_listlike_or_set(punctuation)
        self.punctuation += punctuation

        return self

    def add_special_chars(self, special_chars):
        require_listlike_or_set(special_chars)
        self.special_chars += special_chars

        return self

    def add_metadata_per_token(self, key, data, default=None, dtype=None):
        self._add_metadata('add_metadata_per_token', key, data, default, dtype)
        return self

    def add_metadata_per_doc(self, key, data, default=None, dtype=None):
        self._add_metadata('add_metadata_per_doc', key, data, default, dtype)
        return self

    def remove_metadata(self, key):
        self._invalidate_workers_tokens()

        logger.info('removing metadata key')

        if key not in self.get_available_metadata_keys():
            raise ValueError('unkown metadata key: `%s`' % key)

        self._send_task_to_workers('remove_metadata', key=key)

        return self

    def generate_ngrams(self, n):
        self._invalidate_workers_ngrams()

        logger.info('generating ngrams')
        self._send_task_to_workers('generate_ngrams', n=n)

        return self

    def use_joined_ngrams_as_tokens(self, join_str=' '):
        """
        Use the generated n-grams as tokens by joining them via `join_str`. After this operation, the joined n-grams
        are available as `.tokens` but the original n-grams will be removed and `.ngrams_generated` is reset to False.
        Requires that n-grams have been generated with `.generate_ngrams()` before.
        """
        self._require_ngrams()
        self._invalidate_workers_tokens()
        self._invalidate_workers_ngrams()

        self._send_task_to_workers('use_joined_ngrams_as_tokens', join_str=join_str)

        self.ngrams_as_tokens = True

        return self

    def transform_tokens(self, transform_fn, vectorize=True):
        if not callable(transform_fn):
            raise ValueError('`transform_fn` must be callable')

        process_on_workers = True

        try:
            pickle.dumps(transform_fn)
        except (pickle.PicklingError, AttributeError):
            process_on_workers = False

        self._invalidate_workers_tokens()

        logger.info('transforming tokens')

        if process_on_workers:
            self._send_task_to_workers('transform_tokens', transform_fn=transform_fn, vectorize=vectorize)
        else:
            logger.debug('transforming tokens on main thread')
            if vectorize:
                transform_fn = np.vectorize(transform_fn)

            workers_vocab = self._get_results_per_worker('get_vocab', with_worker_id=True)

            for worker_id, old_vocab in workers_vocab.items():
                self.tasks_queues[worker_id].put(('set_vocab', {'vocab': transform_fn(old_vocab)}))

            [q.join() for q in self.tasks_queues]

        return self

    def tokens_to_lowercase(self):
        self._invalidate_workers_tokens()

        logger.info('transforming tokens to lowercase')
        self._send_task_to_workers('tokens_to_lowercase')

        return self

    def stem(self):
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
        self._require_no_ngrams_as_tokens()

        self._invalidate_workers_tokens()
        logger.info('POS tagging tokens')
        self._send_task_to_workers('pos_tag')

        return self

    def lemmatize(self):
        self._require_pos_tags()
        self._require_no_ngrams_as_tokens()

        self._invalidate_workers_tokens()

        logger.info('lemmatizing tokens')
        self._send_task_to_workers('lemmatize', pos_tagset=self.pos_tagset)

        return self

    def expand_compound_tokens(self, split_chars=('-',), split_on_len=2, split_on_casechange=False):
        self._require_no_ngrams_as_tokens()

        self._invalidate_workers_tokens()

        logger.info('expanding compound tokens')
        self._send_task_to_workers('expand_compound_tokens',
                                   split_chars=split_chars,
                                   split_on_len=split_on_len,
                                   split_on_casechange=split_on_casechange)

        return self

    def remove_special_chars_in_tokens(self):
        return self.remove_chars_in_tokens(self.special_chars)

    def remove_chars_in_tokens(self, chars):
        self._invalidate_workers_tokens()

        logger.info('removing characters in tokens')
        self._send_task_to_workers('remove_chars_in_tokens', chars=chars)

        return self

    def clean_tokens(self, remove_punct=True, remove_stopwords=True, remove_empty=True,
                     remove_shorter_than=None, remove_longer_than=None, remove_numbers=False):
        tokens_to_remove = [''] if remove_empty else []

        if remove_punct:
            tokens_to_remove.extend(self.punctuation)
        if remove_stopwords:
            tokens_to_remove.extend(self.stopwords)

        if tokens_to_remove or remove_shorter_than is not None or remove_longer_than is not None:
            self._invalidate_workers_tokens()

            logger.info('cleaning tokens')
            self._send_task_to_workers('clean_tokens',
                                       tokens_to_remove=tokens_to_remove,
                                       remove_shorter_than=remove_shorter_than,
                                       remove_longer_than=remove_longer_than,
                                       remove_numbers=remove_numbers)

        return self

    def filter_tokens(self, search_token, match_type='exact', ignore_case=False, glob_method='match', inverse=False):
        self._invalidate_workers_tokens()
        logger.info('filtering tokens')
        self._send_task_to_workers('filter_tokens',
                                   search_token=search_token,
                                   match_type=match_type,
                                   ignore_case=ignore_case,
                                   glob_method=glob_method,
                                   inverse=inverse)

        return self

    def filter_tokens_by_pattern(self, tokpattern, ignore_case=False, glob_method='match', inverse=False):
        return self.filter_tokens(search_token=tokpattern, match_type='regex',
                                  ignore_case=ignore_case, glob_method=glob_method,
                                  inverse=inverse)

    def filter_documents(self, search_token, match_type='exact', ignore_case=False, glob_method='match', inverse=False):
        n_docs_orig = self.n_docs

        self._invalidate_docs_info()
        self._invalidate_workers_tokens()

        logger.info('filtering %d documents' % n_docs_orig)
        self._send_task_to_workers('filter_documents',
                                   search_token=search_token,
                                   match_type=match_type,
                                   ignore_case=ignore_case,
                                   glob_method=glob_method,
                                   inverse=inverse)

        return self

    def filter_documents_by_pattern(self, tokpattern, ignore_case=False, glob_method='match', inverse=False):
        return self.filter_documents(search_token=tokpattern, match_type='regex',
                                  ignore_case=ignore_case, glob_method=glob_method,
                                  inverse=inverse)

    def filter_for_pos(self, required_pos, simplify_pos=True):
        """
        Filter tokens for a specific POS tag (if `required_pos` is a string) or several POS tags (if `required_pos`
        is a list/tuple/set of strings). The POS tag depends on the tagset used during tagging. If `simplify_pos` is
        True, then the tags are matched to the following simplified forms:
        - 'N' for nouns
        - 'V' for verbs
        - 'ADJ' for adjectives
        - 'ADV' for adverbs
        - None for all other
        """
        if type(required_pos) not in (tuple, list, set) \
                and required_pos is not None \
                and not isinstance(required_pos, str):
            raise ValueError('`required_pos` must be a string, tuple, list, set or None')

        self._require_pos_tags()

        self._invalidate_workers_tokens()

        logger.info('filtering tokens for POS tag `%s`' % required_pos)
        self._send_task_to_workers('filter_for_pos',
                                   required_pos=required_pos,
                                   pos_tagset=self.pos_tagset,
                                   simplify_pos=simplify_pos)

        return self

    def remove_tokens_by_doc_frequency(self, which, df_threshold, absolute=False):
        which_opts = ('common', '>', '>=', 'uncommon', '<', '<=')
        if which not in which_opts:
            raise ValueError('`which` must be one of: %s' % ', '.join(which_opts))

        if absolute:
            if not type(df_threshold) is int and 1 <= df_threshold <= self.n_docs:
                raise ValueError('`df_threshold` must be integer in range [1, %d]' % self.n_docs)
        else:
            if not 0 <= df_threshold <= 1:
                raise ValueError('`df_threshold` must be in range [0, 1]')

        if which in ('common', '>='):
            comp = operator.ge
        elif which == '>':
            comp = operator.gt
        elif which == '<':
            comp = operator.lt
        else:
            comp = operator.le

        logger.info('removing tokens by document frequency %f (%s value) with comparison operator %s'
                    % (df_threshold, 'absolute' if absolute else 'relative', str(comp)))

        doc_freqs = self.vocabulary_abs_doc_frequency if absolute else self.vocabulary_rel_doc_frequency
        blacklist = set(t for t, f in doc_freqs.items() if comp(f, df_threshold))

        if blacklist:
            self._invalidate_workers_tokens()

            logger.debug('will remove the following %d tokens: %s' % (len(blacklist), blacklist))
            self._send_task_to_workers('clean_tokens', tokens_to_remove=blacklist)

        return self

    def remove_common_tokens(self, df_threshold, absolute=False):
        return self.remove_tokens_by_doc_frequency('common', df_threshold=df_threshold, absolute=absolute)

    def remove_uncommon_tokens(self, df_threshold, absolute=False):
        return self.remove_tokens_by_doc_frequency('uncommon', df_threshold=df_threshold, absolute=absolute)

    def apply_custom_filter(self, filter_func, to_tokens_dataframe=False):
        """
        Apply a custom filter function `filter_func` to all tokens or tokens dataframe.
        `filter_func` must accept a single parameter: a dictionary of structure `{<doc_label>: <tokens list>}` as from
        `.tokens` if `to_tokens_dataframe` is False or a data frame as from `tokens_dataframe`. It must return a result
        with the same structure.

        This function can only be run on a single process, hence it could be slow for large corpora.
        """

        # Because it is not possible to send a function to the workers, all tokens must be fetched from the workers
        # first and then the custom function is called and run in a single process (the main process). After that, the
        # filtered tokens are send back to the worker processes.

        if not callable(filter_func):
            raise ValueError('`filter_func` must be callable')

        self._invalidate_workers_tokens()

        if to_tokens_dataframe:
            logger.info('applying custom filter function to tokens data frame')
            data = self.tokens_dataframe
        else:
            logger.info('applying custom filter function to tokens')
            data = self.tokens

        res = filter_func(data)

        if to_tokens_dataframe:
            self.load_tokens_dataframe(res)
        else:
            self.load_tokens(res)

        return self

    def get_dtm(self):
        logger.info('generating DTM')

        if self._cur_dtm is not None:
            return self._cur_dtm

        workers_res = self._get_results_seq_from_workers('get_num_unique_tokens_per_doc')
        dtm_alloc_size = sum(flatten_list([list(num_unique_per_doc.values()) for num_unique_per_doc in workers_res]))

        self._cur_dtm = create_sparse_dtm(self.vocabulary, self.doc_labels, self.tokens, dtm_alloc_size)

        return self._cur_dtm

    @property
    def _workers_tokens(self):
        if self._cur_workers_tokens is not None:
            return self._cur_workers_tokens

        self._cur_workers_tokens = {}
        workers_res = self._get_results_seq_from_workers('get_tokens')
        for w_res in workers_res:
            assert all(k not in self._cur_workers_tokens.keys() for k in w_res.keys())
            self._cur_workers_tokens.update(w_res)


        return self._cur_workers_tokens

    @property
    def _workers_vocab(self):
        if self._cur_workers_vocab is not None:
            return self._cur_workers_vocab

        self._cur_workers_vocab = self._get_results_seq_from_workers('get_vocab')

        return self._cur_workers_vocab

    @property
    def _workers_ngrams(self):
        if self._cur_workers_ngrams is not None:
            return self._cur_workers_ngrams

        self._cur_workers_ngrams = {}
        workers_res = self._get_results_seq_from_workers('get_ngrams')
        for w_res in workers_res:
            assert all(k not in self._cur_workers_ngrams.keys() for k in w_res.keys())
            self._cur_workers_ngrams.update(w_res)

        return self._cur_workers_ngrams

    def _add_metadata(self, task, key, data, default, dtype):
        if not isinstance(data, dict):
            raise ValueError('`data` must be a dict')

        doc_lengths = self.doc_lengths
        self._invalidate_workers_tokens()

        logger.info('adding metadata per token')

        existing_meta = self.get_available_metadata_keys()

        if len(existing_meta) > 0 and key in existing_meta:
            logger.warning('metadata key `%s` already exists and will be overwritten')

        if task == 'add_metadata_per_doc':
            meta_per_worker = defaultdict(dict)
            for dl, meta in data.items():
                if dl not in doc_lengths.keys():
                    raise ValueError('document `%s` does not exist' % dl)
                if doc_lengths[dl] != len(meta):
                    raise ValueError('length of meta data array does not match number of tokens in document `%s`' % dl)
                meta_per_worker[self.docs2workers[dl]][dl] = meta

            # add column of default values to all documents that were not specified
            docs_without_meta = set(doc_lengths.keys()) - set(data.keys())
            for dl in docs_without_meta:
                meta_per_worker[self.docs2workers[dl]][dl] = [default] * doc_lengths[dl]

            for worker_id, meta in meta_per_worker.items():
                self.tasks_queues[worker_id].put((task, {
                    'key': key,
                    'data': meta,
                    'dtype': dtype
                }))

            for worker_id in meta_per_worker.keys():
                self.tasks_queues[worker_id].join()
        else:
            self._send_task_to_workers(task, key=key, data=data, default=default, dtype=dtype)

    def _load_stemmer(self, custom_stemmer=None):
        logger.info('loading stemmer')

        if custom_stemmer:
            stemmer = custom_stemmer
        else:
            stemmer = nltk.stem.SnowballStemmer(self.language)

        if not hasattr(stemmer, 'stem') or not callable(stemmer.stem):
            raise ValueError('stemmer must have a callable method `stem`')

        return stemmer

    def _load_tokenizer(self, custom_tokenizer=None):
        logger.info('loading tokenizer')

        if custom_tokenizer:
            tokenizer = custom_tokenizer
        else:
            tokenizer = GenericTokenizer(self.language)

        if not hasattr(tokenizer, 'tokenize') or not callable(tokenizer.tokenize):
            raise ValueError('tokenizer must have a callable attribute `tokenize`')

        return tokenizer

    def _load_pos_tagger(self, custom_pos_tagger=None):
        logger.info('loading POS tagger')

        pos_tagset = None
        if custom_pos_tagger:
            tagger = custom_pos_tagger
        else:
            picklefile = os.path.join(DATAPATH, self.language, POS_TAGGER_PICKLE)
            try:
                tagger = unpickle_file(picklefile)
                logger.debug('loaded POS tagger from file `%s`' % picklefile)

                if self.language == 'german':
                    pos_tagset = 'stts'
            except IOError:
                tagger = GenericPOSTagger
                pos_tagset = GenericPOSTagger.tag_set

                logger.debug('loaded GenericPOSTagger (no POS tagger found at `%s`)' % picklefile)

        if not hasattr(tagger, 'tag') or not callable(tagger.tag):
            raise ValueError("pos_tagger must have a callable attribute `tag`")

        return tagger, pos_tagset

    def _create_state_object(self, deepcopy_attrs):
        state_attrs = {}
        attr_blacklist = ('tasks_queues', 'results_queue',
                          'workers', 'n_workers')
        for attr in dir(self):
            if attr.startswith('_') or attr in attr_blacklist:
                continue
            classattr = getattr(type(self), attr, None)
            if classattr is not None and (callable(classattr) or isinstance(classattr, property)):
                continue

            attr_obj = getattr(self, attr)
            if deepcopy_attrs:
                attr_obj = deepcopy(attr_obj)
            state_attrs[attr] = attr_obj

        # worker states
        worker_states = self._get_results_seq_from_workers('get_state')

        return {
            'manager_state': state_attrs,
            'worker_states': worker_states
        }

    def _setup_workers(self, docs=None, initial_states=None, docs_are_tokenized=False):
        """
        Create worker processes and queues. Distribute the work evenly across worker processes. Optionally
        send initial states defined in list `initial_states` to each worker process.
        """
        if initial_states is not None:
            require_listlike_or_set(initial_states)

        self.tasks_queues = []
        self.results_queue = mp.Queue()
        self.workers = []
        self.docs2workers = {}

        common_kwargs = dict(tokenizer=self.tokenizer,
                             stemmer=self.stemmer,
                             pos_tagger=self.pos_tagger)

        if initial_states is not None:
            if docs is not None:
                raise ValueError('`docs` must be None when loading from initial states')
            logger.info('setting up %d worker processes with initial states' % len(initial_states))

            for i_worker, w_state in enumerate(initial_states):
                task_q = mp.JoinableQueue()

                w = PreprocWorker(i_worker, None, self.language, task_q, self.results_queue,
                                  name='_PreprocWorker#%d' % i_worker, **common_kwargs)
                w.start()

                task_q.put(('set_state', w_state))

                self.workers.append(w)
                for dl in w_state['_tokens'].keys():
                    self.docs2workers[dl] = i_worker
                self.tasks_queues.append(task_q)

            [q.join() for q in self.tasks_queues]

        else:
            if docs is None:
                raise ValueError('`docs` must not be None when not loading from initial states')
            if initial_states is not None:
                raise ValueError('`initial_states` must be None when not loading from initial states')

            # distribute work evenly across the worker processes
            # we assume that the longer a document is, the longer the processing time for it is
            # hence we distribute the work evenly by document length
            logger.info('distributing work via greedy partitioning')

            docs_and_lengths = {dl: len(doc) for dl, doc in docs.items()}
            docs_per_worker = greedy_partitioning(docs_and_lengths, k=self.n_max_workers)

            logger.info('setting up %d worker processes' % len(docs_per_worker))

            for i_worker, doc_labels in enumerate(docs_per_worker):
                if not doc_labels: continue
                task_q = mp.JoinableQueue()
                w_docs = {dl: docs.get(dl) for dl in doc_labels}

                w = PreprocWorker(i_worker, w_docs, self.language, task_q, self.results_queue,
                                  docs_are_tokenized=docs_are_tokenized,
                                  name='_PreprocWorker#%d' % i_worker,
                                  **common_kwargs)
                w.start()

                self.workers.append(w)
                for dl in w_docs:
                    self.docs2workers[dl] = i_worker
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
            self.docs2workers = {}
            self.n_workers = 0

    def _get_results_seq_from_workers(self, task, **kwargs):
        logger.debug('getting results sequence for task `%s` from all workers' % task)
        self._send_task_to_workers(task, **kwargs)

        return [self.results_queue.get() for _ in range(self.n_workers)]

    def _get_results_per_worker(self, task, **kwargs):
        logger.debug('getting results per worker for task `%s` from all workers' % task)
        self._send_task_to_workers(task, **kwargs)

        res = [self.results_queue.get() for _ in range(self.n_workers)]

        if res and set(map(len, res)) != {2}:
            raise RuntimeError('all workers must return a 2-tuple as result')

        return dict(res)

    def _invalidate_docs_info(self):
        self._cur_doc_labels = None

    def _invalidate_workers_tokens(self):
        self._cur_workers_tokens = None
        self._cur_workers_vocab = None
        self._cur_vocab_counts = None
        self._cur_dtm = None

    def _invalidate_workers_ngrams(self):
        self._cur_workers_ngrams = None
        self._cur_workers_vocab = None
        self._cur_vocab_counts = None

    def _require_pos_tags(self):
        if not self.pos_tagged:
            raise ValueError("tokens must be POS-tagged before this operation")

    def _require_ngrams(self):
        if not self.ngrams_generated:
            raise ValueError("ngrams must be created before this operation")

    def _require_no_ngrams_as_tokens(self):
        if self.ngrams_as_tokens:
            raise ValueError("ngrams are used as tokens -- this is not possible for this operation")


class GenericPOSTagger(object):
    tag_set = 'penn'

    @staticmethod
    def tag(tokens):
        return nltk.pos_tag(tokens)


class GenericTokenizer(object):
    def __init__(self, language=None):
        self.language = language

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text, self.language)

