"""
Parallel text processing with ``TMPreproc`` class.

Implements several methods for text processing similar to the functional API but processing is done in parallel on a
multi-core machine. For large amounts of text, using TMPreproc on a multi-core machine will be much faster than
sequential processing with the functional API.

Markus Konrad <markus.konrad@wzb.eu>
"""

import string
import multiprocessing as mp
import atexit
from collections import Counter, defaultdict, OrderedDict
from copy import deepcopy
from functools import partial
import pickle
import logging

import numpy as np
from scipy.sparse import csr_matrix
import datatable as dt
from deprecation import deprecated

from .. import defaults
from ..corpus import Corpus
from ..bow.dtm import dtm_to_datatable, dtm_to_dataframe
from ..utils import require_listlike, require_listlike_or_set, require_dictlike, pickle_data, unpickle_file,\
    greedy_partitioning, flatten_list, combine_sparse_matrices_columnwise
from ._preprocworker import PreprocWorker
from ._common import tokenize, stem, pos_tag, load_pos_tagger_for_language, lemmatize, load_lemmatizer_for_language,\
    doc_lengths, _finalize_kwic_results, _datatable_from_kwic_results, remove_tokens_by_doc_frequency

logger = logging.getLogger('tmtoolkit')
logger.addHandler(logging.NullHandler())


class TMPreproc:
    """
    TMPreproc implements a class for parallel text processing. The API implements a state machine, i.e. you create a
    TMPreproc instance with text documents and modify them by calling methods like "to_lowercase", etc.
    """
    def __init__(self, docs, language=defaults.language, n_max_processes=None,
                 stopwords=None, punctuation=None, special_chars=None,
                 tokenizer=None, pos_tagger=None, pos_tagset=None, stemmer=None, lemmatizer=None):
        """
        Create a parallel text processing instance by passing a dictionary of raw texts ``docs`` with document label
        to document text mapping. You can pass a ``Corpus`` instance because ``Corpus`` implements the dictionary
        methods.

        TMPreproc will start ``n_max_processes`` sub-processes and distribute the documents on them for parallel
        processing.

        :param docs: documents dictionary ("corpus") with document label to document text mapping
        :param language: documents language used for language-dependent methods such as POS tagging or lemmatization
        :param n_max_processes: max. number of sub-processes for parallel processing; uses the number of CPUs on the
                                current machine if None is passed
        :param stopwords: provide manual stopword list or use default stopword list for given language
        :param punctuation: provide manual punctuation symbols list or use default list from ``string.punctuation``
        :param special_chars: provide manual special characters list or use default list from ``string.punctuation``
        :param tokenizer: provide custom tokenizer function or use ``tokenize()``
        :param pos_tagger: provide custom POS tagger function or use ``pos_tag()``
        :param pos_tagset: custom tagset label for custom POS tagger
        :param stemmer: provide custom stemmer function or use ``stem()``
        :param lemmatizer: provide custom lemmatizer function or use ``lemmatize()``
        """

        if docs is not None:
            require_dictlike(docs)
            logger.info('init with %d documents' % len(docs))

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
        self._cur_workers_vocab_doc_freqs = None
        self._cur_workers_ngrams = None
        self._cur_vocab_counts = None
        self._cur_dtm = None

        self.language = language   # document language
        self.ngrams_as_tokens = False

        if stopwords is None:      # load default stopword list for this language
            import nltk
            self.stopwords = nltk.corpus.stopwords.words(language)
        else:                      # set passed stopword list
            require_listlike(stopwords)
            self.stopwords = stopwords

        if punctuation is None:    # load default punctuation list
            self.punctuation = list(string.punctuation)
        else:
            require_listlike(punctuation)
            self.punctuation = punctuation

        if special_chars is None:
            self.special_chars = list(string.punctuation)
        else:
            require_listlike(special_chars)
            self.special_chars = special_chars

        if tokenizer is None:
            self.tokenizer = partial(tokenize, language=language)
        else:
            self.tokenizer = tokenizer

        if pos_tagger is None:
            tagger_instance, self.pos_tagset = load_pos_tagger_for_language(language)
            self.pos_tagger = partial(pos_tag, tagger_instance=tagger_instance)
        else:
            self.pos_tagger = pos_tagger
            self.pos_tagset = pos_tagset

        if stemmer is None:
            import nltk
            self.stemmer = partial(stem, stemmer_instance=nltk.stem.SnowballStemmer(language))
        else:
            self.stemmer = stemmer

        if lemmatizer is None:
            self.lemmatizer = partial(lemmatize, lemmatizer_fn=load_lemmatizer_for_language(language))
        else:
            self.lemmatizer = lemmatizer

        if docs is not None:
            self._setup_workers(docs=docs)

        atexit.register(self.shutdown_workers)

    def __del__(self):
        """destructor. shutdown all workers"""
        self.shutdown_workers()

    def __str__(self):
        return 'TMPreproc with %d documents' % self.n_docs

    def __repr__(self):
        return '<TMPreproc [%d documents]>' % self.n_docs

    @property
    def n_docs(self):
        """Number of documents."""
        return len(self.doc_labels)

    @property
    def n_tokens(self):
        """Number of tokens in all documents (sum of document lengths)."""
        return sum(self.doc_lengths.values())

    @property
    def doc_labels(self):
        """Document labels as sorted list."""
        if self._cur_doc_labels is None:
            self._cur_doc_labels = sorted(flatten_list(self._get_results_seq_from_workers('get_doc_labels')))

        return self._cur_doc_labels

    @property
    def doc_lengths(self):
        """Document lengths as dict with mapping document label to document length (number of tokens in doc.)."""
        return dict(zip(self.tokens.keys(), doc_lengths(self.tokens.values())))

    @property
    def tokens(self):
        """Document tokens as dict with mapping document label to list of tokens."""
        return self.get_tokens(with_metadata=False)

    @property
    def tokens_with_metadata(self):
        """Document tokens with metadata (e.g. POS tag) as dict with mapping document label to datatable."""
        return self.get_tokens(with_metadata=True, as_data_tables=True)

    @property
    def tokens_datatable(self):
        """
        Tokens and metadata as datatable with columns "doc" (document label), "position" (token position in
        the document), "token" and optional meta data columns.
        """
        tokens = self.get_tokens(non_empty=True, with_metadata=True, as_data_tables=True)
        dfs = []
        for dl, df in tokens.items():
            n = df.shape[0]
            meta_df = dt.Frame({
                'doc': np.repeat(dl, n),
                'position': np.arange(n)
            })

            dfs.append(dt.cbind(meta_df, df))

        if dfs:
            res = dt.rbind(*dfs)
        else:
            res = dt.Frame({'doc': [], 'position': [], 'token': []})

        return res[:, :, dt.sort(dt.f.doc, dt.f.position)]

    @property
    def tokens_dataframe(self):
        """
        Tokens and metadata as pandas DataFrame with indices "doc" (document label) and "position" (token position in
        the document) and columns "token" plus optional meta data columns.
        """
        # note that generating a datatable first and converting it to pandas is faster than generating a pandas data
        # frame right away
        return self.tokens_datatable.to_pandas().set_index(['doc', 'position'])

    @property
    def tokens_with_pos_tags(self):
        """
        Document tokens with POS tag as dict with mapping document label to datatable. The datatables have two
        columns, ``token`` and ``meta_pos``.
        """
        self._require_pos_tags()
        return {dl: df[:, ['token', 'meta_pos']]
                for dl, df in self.get_tokens(with_metadata=True, as_data_tables=True).items()}

    @property
    def vocabulary(self):
        """Corpus vocabulary, i.e. sorted list of all tokens that occur across all documents."""
        return self.get_vocabulary()

    @property
    def vocabulary_counts(self):
        """``Counter()`` instance of vocabulary containing counts of occurrences of tokens across all documents."""
        if self._cur_vocab_counts is not None:
            return self._cur_vocab_counts

        # get vocab counts
        workers_vocab_counts = self._get_results_seq_from_workers('get_vocab_counts')

        # sum up the worker's doc. frequencies
        self._cur_vocab_counts = sum(map(Counter, workers_vocab_counts), Counter())

        return self._cur_vocab_counts

    @property
    def vocabulary_abs_doc_frequency(self):
        """
        Absolute document frequency per vocabulary token as dict with token to document frequency mapping.
        Document frequency is the measure of how often a token occurs *at least once* in a document.
        Example:
        ::

            doc tokens
            --- ------
            A   z, z, w, x
            B   y, z, y
            C   z, z, y, z

            document frequency df(z) = 3  (occurs in all 3 documents)
            df(x) = df(w) = 1 (occur only in A)
            df(y) = 2 (occurs in B and C)
            ...
        """
        return self._workers_vocab_doc_frequencies

    @property
    def vocabulary_rel_doc_frequency(self):
        """
        Same as ``vocabulary_abs_doc_frequency`` but normalized by number of documents, i.e. relative document
        frequency.
        """
        return {w: n/self.n_docs for w, n in self._workers_vocab_doc_frequencies.items()}

    @property
    def ngrams_generated(self):
        """Indicates if n-grams were generated before (True if yes, else False)."""
        return len(self.ngrams) > 0

    @property
    def ngrams(self):
        """
        Generated n-grams as dict with mapping document label to list of n-grams. Each list of n-grams (i.e. each
        document) in turn contains lists of size ``n`` (i.e. two if you generated bigrams).
        """
        return self.get_ngrams()

    @property
    def pos_tagged(self):
        """Indicates if documents were POS tagged. True if yes, otherwise False."""
        meta_keys = self.get_available_metadata_keys()
        return 'pos' in meta_keys

    @property
    def dtm(self):
        """
        Generate and return a sparse document-term matrix of shape ``(n_docs, n_vocab)`` where ``n_docs`` is the number
        of documents and ``n_vocab`` is the vocabulary size.
        """
        return self.get_dtm()

    def shutdown_workers(self):
        """
        Manually send the shutdown signal to all worker processes.

        Normally you don't need to call this manually as the worker processes are killed automatically when the
        TMPreproc instance is removed. However, if you need to free resources immediately, you can use this method
        as it is also used in the tests.
        """
        try:   # may cause exception when the logger is actually already destroyed
            logger.info('sending shutdown signal to workers')
        except: pass
        self._send_task_to_workers(None)

    def save_state(self, picklefile):
        """
        Save the current state of this TMPreproc instance to disk, i.e. to the pickle file ``picklefile``.
        The state can be restored from this file using ``load_state()`` or class method ``from_state()``.

        :param picklefile: disk file to store the state to
        :return: this instance
        """
        # attributes for this instance ("manager instance")
        logger.info('saving state to file `%s`' % picklefile)

        # save to pickle
        pickle_data(self._create_state_object(deepcopy_attrs=False), picklefile)

        return self

    def load_state(self, file_or_stateobj):
        """
        Restore a state by loading either a pickled state from disk as saved with ``save_state()`` or by loading a state
        object directly.
        :param file_or_stateobj: either path to a pickled file as saved with ``save_state()`` or a state object
        :return: this instance as restored from the passed file / object
        """
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
        Load tokens ``tokens`` into TMPreproc in the same format as they are returned by ``self.tokens`` or
        ``self.tokens_with_metadata``, i.e. as dict with mapping: document label -> document tokens array or
        document data frame.

        :param tokens: dict of tokens as returned by ``self.tokens`` or ``self.tokens_with_metadata``
        :return: this instance
        """
        require_dictlike(tokens)

        logger.info('loading tokens of %d documents' % len(tokens))

        # recreate worker processes
        self.shutdown_workers()

        tokens_dicts = {}
        if tokens:
            for dl, doc in tokens.items():
                if isinstance(doc, dt.Frame):
                    tokens_dicts[dl] = {col: coldata for col, coldata in zip(doc.names, doc.to_list())}
                elif isinstance(doc, list):
                    tokens_dicts[dl] = {'token': doc}
                else:
                    raise ValueError('document `%s` is of unknown type `%s`' % (dl, type(doc)))

        self._setup_workers(tokens_dicts, docs_are_tokenized=True)

        self._invalidate_docs_info()
        self._invalidate_workers_tokens()
        self._invalidate_workers_ngrams()

        return self

    def load_tokens_datatable(self, tokendf):
        """
        Load tokens dataframe ``tokendf`` into TMPreproc in the same format as they are returned by
        ``self.tokens_frame``, i.e. as data frame with hierarchical indices "doc" and "position" and at least a
        column "token" plus optional columns like "meta_pos", etc.

        :param tokendf: tokens datatable Frame object as returned by ``self.tokens_frame``
        :return: this instance
        """
        if not isinstance(tokendf, dt.Frame):
            raise ValueError('`tokendf` must be a datatable Frame object')

        if {'doc', 'position', 'token'} & set(tokendf.names) != {'doc', 'position', 'token'}:
            raise ValueError('`tokendf` must contain a columns "doc", "position" and "token"')

        # convert big dataframe to dict of document token dicts to be used in load_tokens
        tokens = {}
        for dl in dt.unique(tokendf[:, dt.f.doc]).to_list()[0]:
            doc_df = tokendf[dt.f.doc == dl, :]
            colnames = list(doc_df.names)
            colnames.pop(colnames.index('doc'))
            tokens[dl] = doc_df[:, colnames]

        return self.load_tokens(tokens)

    def __copy__(self):
        """
        Copy a TMPreproc object including all its present state (tokens, meta data, etc.).
        Performs a deep copy.

        :return: deep copy of the current TMPreproc instance
        """
        return self.copy()

    def __deepcopy__(self, memodict=None):
        """
        Copy a TMPreproc object including all its present state (tokens, meta data, etc.).
        Performs a deep copy.

        :return: deep copy of the current TMPreproc instance
        """
        return self.copy()

    def copy(self):
        """
        Copy a TMPreproc object including all its present state (tokens, meta data, etc.).
        Performs a deep copy.

        :return: deep copy of the current TMPreproc instance
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
    def from_tokens_datatable(cls, tokensdf, **init_kwargs):
        if 'docs' in init_kwargs.keys():
            raise ValueError('`docs` cannot be passed as argument when loading a token datatable')
        init_kwargs['docs'] = None

        return cls(**init_kwargs).load_tokens_datatable(tokensdf)

    @deprecated(deprecated_in='0.9.0', removed_in='0.10.0',
                details='Method not necessary anymore since documents are directly tokenized upon instantiation '
                        'of TMPreproc.')
    def tokenize(self):
        return self

    def get_tokens(self, non_empty=False, with_metadata=True, as_data_tables=False):
        tokens = self._workers_tokens
        meta_keys = self.get_available_metadata_keys()

        if not with_metadata:  # doc label -> token array
            tokens = {dl: doc['token'] for dl, doc in tokens.items()}

        if as_data_tables:
            if with_metadata:  # doc label -> doc data frame with token and meta data columns
                tokens_dfs = {}
                for dl, doc in tokens.items():
                    df_args = [('token', doc['token'])]
                    for k in meta_keys:  # to preserve the correct order of meta data columns
                        col = 'meta_' + k
                        df_args.append((col, doc[col]))
                    tokens_dfs[dl] = dt.Frame(OrderedDict(df_args))
                tokens = tokens_dfs
            else:              # doc label -> doc data frame only with "token" column
                tokens = {dl: dt.Frame({'token': doc}) for dl, doc in tokens.items()}

        if non_empty:
            return {dl: doc for dl, doc in tokens.items()
                    if (isinstance(doc, dict) and len(doc['token']) > 0)
                        or (not isinstance(doc, dict) and doc.shape[0] > 0)}
        else:
            return tokens

    def get_kwic(self, search_token, context_size=2, match_type='exact', ignore_case=False, glob_method='match',
                 inverse=False, with_metadata=False, as_data_table=False, non_empty=False, glue=None,
                 highlight_keyword=None):
        """
        Perform keyword-in-context (kwic) search for `search_token`. Uses similar search parameters as
        `filter_tokens()`.

        :param search_token: search pattern
        :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                             if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
        :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                           `search_token` must be a "glob" pattern like "hello w*"
                           (see https://github.com/metagriffin/globre).
        :param ignore_case: If True, ignore case for matching.
        :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                            behavior as Python's `re.match` or `re.search`).
        :param inverse: Invert the matching results.
        :param with_metadata: Also return metadata (like POS) along with each token.
        :param as_data_table: Return result as data frame with indices "doc" (document label) and "context" (context
                              ID per document) and optionally "position" (original token position in the document) if
                              tokens are not glued via `glue` parameter.
        :param non_empty: If True, only return non-empty result documents.
        :param glue: If not None, this must be a string which is used to combine all tokens per match to a single string
        :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                                  matched keyword.
        :return: Return dict with `document label -> kwic for document` mapping or a data frame, depending
        on `as_data_table`.
        """
        if isinstance(context_size, int):
            context_size = (context_size, context_size)
        else:
            require_listlike(context_size)

        if highlight_keyword is not None and not isinstance(highlight_keyword, str):
            raise ValueError('if `highlight_keyword` is given, it must be of type str')

        if glue:
            if with_metadata or as_data_table:
                raise ValueError('when `glue` is set to True, `with_metadata` and `as_data_table` must be False')
            if not isinstance(glue, str):
                raise ValueError('if `glue` is given, it must be of type str')

        # list of results from all workers
        kwic_results = self._get_results_seq_from_workers('get_kwic',
                                                          context_size=context_size,
                                                          search_token=search_token,
                                                          highlight_keyword=highlight_keyword,
                                                          with_metadata=with_metadata,
                                                          with_window_indices=as_data_table,
                                                          match_type=match_type,
                                                          ignore_case=ignore_case,
                                                          glob_method=glob_method,
                                                          inverse=inverse)

        # form kwic with doc label -> results
        kwic = {}
        for worker_kwic in kwic_results:
            kwic.update(worker_kwic)

        return _finalize_kwic_results(kwic, non_empty=non_empty, glue=glue,
                                      as_data_table=as_data_table, with_metadata=with_metadata)

    def get_kwic_table(self, search_token, context_size=2, match_type='exact', ignore_case=False, glob_method='match',
                       inverse=False, glue=' ', highlight_keyword='*'):
        """
        Shortcut for `get_kwic` to directly return a data frame table with highlighted keywords in context.

        :param search_token: search pattern
        :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                             if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
        :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                           `search_token` must be a "glob" pattern like "hello w*"
                           (see https://github.com/metagriffin/globre).
        :param ignore_case: If True, ignore case for matching.
        :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                            behavior as Python's `re.match` or `re.search`).
        :param inverse: Invert the matching results.
        :param glue: If not None, this must be a string which is used to combine all tokens per match to a single string
        :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                                  matched keyword.
        :return: Data frame with indices "doc" (document label) and "context" (context ID per document) and column
                 "kwic" containing strings with highlighted keywords in context.
        """

        kwic = self.get_kwic(search_token=search_token, context_size=context_size, match_type=match_type,
                             ignore_case=ignore_case, glob_method=glob_method, inverse=inverse,
                             with_metadata=False, as_data_table=False, non_empty=True,
                             glue=glue, highlight_keyword=highlight_keyword)

        return _datatable_from_kwic_results(kwic)

    def glue_tokens(self, patterns, glue='_', match_type='exact', ignore_case=False, glob_method='match',
                    inverse=False):
        """
        Match N *subsequent* tokens to the N patterns in `patterns` using match options like in `filter_tokens`.
        Join the matched tokens by glue string `glue`. Replace these tokens in the documents.

        If there is metadata, the respective entries for the joint tokens are set to None.

        Return a set of all joint tokens.

        :param patterns: A sequence of search patterns as excepted by `filter_tokens`.
        :param glue: String for joining the subsequent matches.
        :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                           `search_token` must be a "glob" pattern like "hello w*"
                           (see https://github.com/metagriffin/globre).
        :param ignore_case: If True, ignore case for matching.
        :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                            behavior as Python's `re.match` or `re.search`).
        :param inverse: Invert the matching results.
        :return: Set of all joint tokens.
        """

        require_listlike(patterns)

        if len(patterns) < 2:
            raise ValueError('`patterns` must contain at least two strings')

        if not isinstance(glue, str):
            raise ValueError('`glue` must be a string')

        self._invalidate_workers_tokens()

        # list of results from all workers
        glued_tokens_per_workers = self._get_results_seq_from_workers('glue_tokens', patterns=patterns, glue=glue,
                                                                      match_type=match_type, ignore_case=ignore_case,
                                                                      glob_method=glob_method, inverse=inverse)

        glued_tokens = set()
        for tok_set in glued_tokens_per_workers:
            glued_tokens.update(tok_set)

        return glued_tokens

    def get_vocabulary(self, sort=True):
        """
        Return the vocabulary, i.e. the list of unique words across all documents, as sorted NumPy array.
        """
        tokens = set(flatten_list(self._get_results_seq_from_workers('get_vocab')))

        if sort:
            return sorted(tokens)
        else:
            return list(tokens)

    def get_ngrams(self, non_empty=False):
        if non_empty:
            return {dl: dtok for dl, dtok in self._workers_ngrams.items() if len(dtok) > 0}
        else:
            return self._workers_ngrams

    def get_available_metadata_keys(self):
        keys = self._get_results_seq_from_workers('get_available_metadata_keys')

        return set(flatten_list(keys))

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

    def add_metadata_per_token(self, key, data, default=None):
        self._add_metadata('add_metadata_per_token', key, data, default)
        return self

    def add_metadata_per_doc(self, key, data, default=None):
        self._add_metadata('add_metadata_per_doc', key, data, default)
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

    def join_ngrams(self, join_str=' '):
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

    def transform_tokens(self, transform_fn):
        if not callable(transform_fn):
            raise ValueError('`transform_fn` must be callable')

        process_on_workers = True
        tokens = None

        try:
            pickle.dumps(transform_fn)
        except (pickle.PicklingError, AttributeError):
            process_on_workers = False
            tokens = self._workers_tokens

        self._invalidate_workers_tokens()

        logger.info('transforming tokens')

        if process_on_workers:
            self._send_task_to_workers('transform_tokens', transform_fn=transform_fn)
        else:
            logger.debug('transforming tokens on main thread')

            new_tokens = defaultdict(dict)
            for dl, doc in tokens.items():
                new_tokens[self.docs2workers[dl]][dl] = list(map(transform_fn, doc['token']))

            for worker_id, worker_tokens in new_tokens.items():
                self.tasks_queues[worker_id].put(('replace_tokens', {'tokens': worker_tokens}))

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
        Apply Part-of-Speech (POS) tagging to all documents. POS tags can then be retrieved via `tokens_with_metadata`
        or `tokens_with_pos_tags` properties or the `get_tokens()` method.

        POS tagging so far only works for English and German. The English tagger uses the Penn Treebank tagset
        (https://ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html), the
        German tagger uses STTS (http://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/TagSets/stts-table.html).
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
        self._send_task_to_workers('lemmatize')

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
        self._send_task_to_workers('remove_chars', chars=chars)

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

    def filter_tokens(self, search_tokens, match_type='exact', ignore_case=False, glob_method='match', inverse=False):
        self._check_filter_args(match_type=match_type, glob_method=glob_method)

        self._invalidate_workers_tokens()

        logger.info('filtering tokens')
        self._send_task_to_workers('filter_tokens',
                                   search_tokens=search_tokens,
                                   match_type=match_type,
                                   ignore_case=ignore_case,
                                   glob_method=glob_method,
                                   inverse=inverse)

        return self

    def remove_tokens(self, search_tokens, match_type='exact', ignore_case=False, glob_method='match'):
        return self.filter_tokens(search_tokens=search_tokens, match_type=match_type,
                                  ignore_case=ignore_case, glob_method=glob_method,
                                  inverse=True)

    def filter_documents(self, search_tokens, matches_threshold=1, match_type='exact', ignore_case=False,
                         glob_method='match', inverse=False):
        self._check_filter_args(match_type=match_type, glob_method=glob_method)

        n_docs_orig = self.n_docs

        self._invalidate_docs_info()
        self._invalidate_workers_tokens()

        logger.info('filtering %d documents' % n_docs_orig)
        self._send_task_to_workers('filter_documents',
                                   search_tokens=search_tokens,
                                   matches_threshold=matches_threshold,
                                   match_type=match_type,
                                   ignore_case=ignore_case,
                                   glob_method=glob_method,
                                   inverse=inverse)

        return self

    def remove_documents(self, search_token, matches_threshold=1, match_type='exact', ignore_case=False,
                         glob_method='match'):
        return self.filter_documents(search_tokens=search_token, matches_threshold=matches_threshold,
                                     match_type=match_type, ignore_case=ignore_case, glob_method=glob_method,
                                     inverse=True)

    def filter_documents_by_name(self, name_patterns, match_type='exact', ignore_case=False, glob_method='match',
                                 inverse=False):
        """
        Filter documents by their name (i.e. document label). Keep all documents whose name matches `name_pattern`
        according to additional matching options. If `inverse` is True, drop all those documents whose name matches,
        which is the same as calling `remove_documents_by_name`.
        :param name_patterns: either single search string or sequence of search strings
        :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                           `search_token` must be a "glob" pattern like "hello w*"
                           (see https://github.com/metagriffin/globre).
        :param ignore_case: If True, ignore case for matching.
        :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                            behavior as Python's `re.match` or `re.search`).
        :param inverse: Invert the matching results.
        :return: this TMPreproc instance
        """
        self._check_filter_args(match_type=match_type, glob_method=glob_method)

        n_docs_orig = self.n_docs

        self._invalidate_docs_info()
        self._invalidate_workers_tokens()

        logger.info('filtering %d documents' % n_docs_orig)
        self._send_task_to_workers('filter_documents_by_name',
                                   name_patterns=name_patterns,
                                   match_type=match_type,
                                   ignore_case=ignore_case,
                                   glob_method=glob_method,
                                   inverse=inverse)

        return self

    def remove_documents_by_name(self, name_patterns, match_type='exact', ignore_case=False, glob_method='match'):
        """
        Same as `filter_documents_by_name` with `inverse=True`: drop all those documents whose name matches.
        :param name_patterns: either single search string or sequence of search strings
        :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                           `search_token` must be a "glob" pattern like "hello w*"
                           (see https://github.com/metagriffin/globre).
        :param ignore_case: If True, ignore case for matching.
        :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                            behavior as Python's `re.match` or `re.search`).
        :return: this TMPreproc instance
        """
        return self.filter_documents_by_name(name_patterns, match_type=match_type,
                                             ignore_case=ignore_case, glob_method=glob_method,
                                             inverse=True)

    def filter_for_pos(self, required_pos, simplify_pos=True, inverse=False):
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
        if not isinstance(required_pos, (tuple, list, set, str)) \
                and required_pos is not None:
            raise ValueError('`required_pos` must be a string, tuple, list, set or None')

        self._require_pos_tags()

        self._invalidate_workers_tokens()

        logger.info('filtering tokens for POS tag `%s`' % required_pos)
        self._send_task_to_workers('filter_for_pos',
                                   required_pos=required_pos,
                                   pos_tagset=self.pos_tagset,
                                   simplify_pos=simplify_pos,
                                   inverse=inverse)

        return self

    def remove_tokens_by_doc_frequency(self, which, df_threshold, absolute=False):
        blacklist = remove_tokens_by_doc_frequency([doc['token'] for doc in self._workers_tokens.values()], which,
                                                   df_threshold=df_threshold, absolute=absolute, return_blacklist=True)

        if blacklist:
            self._invalidate_workers_tokens()

            logger.debug('will remove the following %d tokens: %s' % (len(blacklist), blacklist))
            self._send_task_to_workers('filter_tokens',
                                       search_tokens=blacklist,
                                       match_type='exact',
                                       ignore_case=False,
                                       glob_method='match',
                                       inverse=True)

        return self

    def remove_common_tokens(self, df_threshold, absolute=False):
        return self.remove_tokens_by_doc_frequency('common', df_threshold=df_threshold, absolute=absolute)

    def remove_uncommon_tokens(self, df_threshold, absolute=False):
        return self.remove_tokens_by_doc_frequency('uncommon', df_threshold=df_threshold, absolute=absolute)

    def apply_custom_filter(self, filter_func, to_tokens_dataframe=False):
        """
        Apply a custom filter function `filter_func` to all tokens or tokens dataframe.
        `filter_func` must accept a single parameter: a dictionary of structure `{<doc_label>: <tokens list>}` as from
        `.tokens` if `to_tokens_dataframe` is False or a data frame as from `tokens_datatable`. It must return a result
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
            data = self.tokens_datatable
        else:
            logger.info('applying custom filter function to tokens')
            data = self.tokens

        res = filter_func(data)

        if to_tokens_dataframe:
            self.load_tokens_datatable(res)
        else:
            self.load_tokens(res)

        return self

    def get_dtm(self, as_data_table=False, as_data_frame=False, dtype=None):
        """
        Generate a sparse document-term-matrix (DTM) for the current tokens with rows representing documents according
        to `self.doc_labels` and columns representing tokens according to `self.vocabulary`.
        :param as_data_table: Return result as data table with document labels in '_doc' column and vocabulary as
                              column names
        :param as_data_frame: Return result as pandas data frame with document labels in index and vocabulary as
                              column names
        :param dtype: optionally specify a DTM data type. by default it is 32bit integer
        :return: either a sparse document-term-matrix (in CSR format) or a datatable, depending on `as_data_table`
        """
        if as_data_table and as_data_frame:
            raise ValueError('you cannot set both `as_data_table` and `as_data_frame` to True')

        if self._cur_dtm is None:
            logger.info('generating DTM')

            if self.n_docs > 0:
                # get partial DTMs for each worker's set of documents
                workers_dtm_vocab = self._get_results_seq_from_workers('get_dtm')
                w_dtms, w_doc_labels, w_vocab = zip(*workers_dtm_vocab)

                # combine these to form a DTM of all documents
                dtm, dtm_vocab, dtm_doc_labels = combine_sparse_matrices_columnwise(w_dtms, w_vocab, w_doc_labels,
                                                                                    dtype=dtype)

                # sort according to document labels
                # dtm_vocab == self.vocabulary (both sorted) but dtm_doc_labels is not sorted
                self._cur_dtm = dtm[np.argsort(dtm_doc_labels), :]
                vocab = dtm_vocab.tolist()
            else:
                self._cur_dtm = csr_matrix((0, 0), dtype=dtype)  # empty sparse matrix
                vocab = list()
        else:
            vocab = self.get_vocabulary()

        if as_data_table:
            return dtm_to_datatable(self._cur_dtm.todense(), self.doc_labels, vocab)
        elif as_data_frame:
            return dtm_to_dataframe(self._cur_dtm.todense(), self.doc_labels, vocab)
        else:
            return self._cur_dtm

    @property
    def _workers_tokens(self):
        if self._cur_workers_tokens is not None:
            return self._cur_workers_tokens

        self._cur_workers_tokens = {}
        workers_res = self._get_results_seq_from_workers('get_tokens')
        for w_res in workers_res:
            self._cur_workers_tokens.update(w_res)

        return self._cur_workers_tokens

    @property
    def _workers_vocab_doc_frequencies(self):
        if self._cur_workers_vocab_doc_freqs is not None:
            return self._cur_workers_vocab_doc_freqs

        workers_doc_freqs = self._get_results_seq_from_workers('get_vocab_doc_frequencies')

        self._cur_workers_vocab_doc_freqs = Counter()
        for doc_freqs in workers_doc_freqs:
            self._cur_workers_vocab_doc_freqs.update(doc_freqs)

        return self._cur_workers_vocab_doc_freqs

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

    def _add_metadata(self, task, key, data, default):
        if not isinstance(data, dict):
            raise ValueError('`data` must be a dict')

        doc_sizes = self.doc_lengths
        self._invalidate_workers_tokens()

        logger.info('adding metadata per token')

        existing_meta = self.get_available_metadata_keys()

        if len(existing_meta) > 0 and key in existing_meta:
            logger.warning('metadata key `%s` already exists and will be overwritten')

        if task == 'add_metadata_per_doc':
            meta_per_worker = defaultdict(dict)
            for dl, meta in data.items():
                if dl not in doc_sizes.keys():
                    raise ValueError('document `%s` does not exist' % dl)
                if doc_sizes[dl] != len(meta):
                    raise ValueError('length of meta data array does not match number of tokens in document `%s`' % dl)
                meta_per_worker[self.docs2workers[dl]][dl] = meta

            # add column of default values to all documents that were not specified
            docs_without_meta = set(doc_sizes.keys()) - set(data.keys())
            for dl in docs_without_meta:
                meta_per_worker[self.docs2workers[dl]][dl] = [default] * doc_sizes[dl]

            for worker_id, meta in meta_per_worker.items():
                self.tasks_queues[worker_id].put((task, {
                    'key': key,
                    'data': meta
                }))

            for worker_id in meta_per_worker.keys():
                self.tasks_queues[worker_id].join()
        else:
            self._send_task_to_workers(task, key=key, data=data, default=default)

    def _create_state_object(self, deepcopy_attrs):
        state_attrs = {}
        attr_blacklist = ('tasks_queues', 'results_queue',
                          'workers', 'n_workers', 'lemmatizer')
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
                             lemmatizer=self.lemmatizer,
                             pos_tagger=self.pos_tagger)

        if initial_states is not None:
            if docs is not None:
                raise ValueError('`docs` must be None when loading from initial states')
            logger.info('setting up %d worker processes with initial states' % len(initial_states))

            for i_worker, w_state in enumerate(initial_states):
                task_q = mp.JoinableQueue()

                w = PreprocWorker(i_worker, self.language, task_q, self.results_queue,
                                  name='_PreprocWorker#%d' % i_worker, **common_kwargs)
                w.start()

                task_q.put(('set_state', w_state))

                self.workers.append(w)
                for dl in w_state['_doc_labels']:
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

            # create worker processes
            for i_worker, doc_labels in enumerate(docs_per_worker):
                if not doc_labels: continue
                task_q = mp.JoinableQueue()

                w = PreprocWorker(i_worker, self.language, task_q, self.results_queue,
                                  name='_PreprocWorker#%d' % i_worker,
                                  **common_kwargs)
                w.start()

                self.workers.append(w)
                for dl in doc_labels:
                    self.docs2workers[dl] = i_worker
                self.tasks_queues.append(task_q)

            # send init task
            for i_worker, doc_labels in enumerate(docs_per_worker):
                self.tasks_queues[i_worker].put(('init', dict(docs={dl: docs[dl] for dl in doc_labels},
                                                              docs_are_tokenized=docs_are_tokenized)))

            [q.join() for q in self.tasks_queues]

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
        self._cur_workers_vocab_doc_freqs = None
        self._cur_vocab_counts = None
        self._cur_dtm = None

    def _invalidate_workers_ngrams(self):
        self._cur_workers_ngrams = None
        self._cur_workers_vocab = None
        self._cur_workers_vocab_doc_freqs = None
        self._cur_vocab_counts = None

    @staticmethod
    def _check_filter_args(**kwargs):
        if 'match_type' in kwargs and kwargs['match_type'] not in {'exact', 'regex', 'glob'}:
            raise ValueError("`match_type` must be one of `'exact', 'regex', 'glob'`")

        if 'glob_method' in kwargs and kwargs['glob_method'] not in {'search', 'match'}:
            raise ValueError("`glob_method` must be one of `'search', 'match'`")

    def _require_pos_tags(self):
        if not self.pos_tagged:
            raise ValueError("tokens must be POS-tagged before this operation")

    def _require_ngrams(self):
        if not self.ngrams_generated:
            raise ValueError("ngrams must be created before this operation")

    def _require_no_ngrams_as_tokens(self):
        if self.ngrams_as_tokens:
            raise ValueError("ngrams are used as tokens -- this is not possible for this operation")
