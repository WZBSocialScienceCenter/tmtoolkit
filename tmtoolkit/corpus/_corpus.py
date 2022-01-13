"""
Internal module that implements :class:`Corpus` class representing a set of texts as token sequences in labelled
documents.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

from __future__ import annotations  # req. for classmethod return type; see https://stackoverflow.com/a/49872353

import logging
import multiprocessing as mp
import string
from copy import deepcopy
from typing import Dict, Union, List, Optional, Any, Iterator, Callable, Sequence, ItemsView, KeysView, ValuesView, \
    Generator, Tuple, Collection

import numpy as np
import spacy
from bidict import bidict
from spacy import Language
from spacy.tokens import Doc
from loky import get_reusable_executor, ProcessPoolExecutor

from ._common import DEFAULT_LANGUAGE_MODELS, SPACY_TOKEN_ATTRS, STD_TOKEN_ATTRS, BOOLEAN_SPACY_TOKEN_ATTRS, \
    TOKENMAT_ATTRS
from ._document import Document
from ..utils import greedy_partitioning, split_func_args


logger = logging.getLogger('tmtoolkit')


class Corpus:
    """
    The Corpus class represents text as *string token sequences* in labelled documents. It behaves like a Python dict,
    i.e. you can access document tokens via square brackets (``corp['my_doc']``).

    `SpaCy <https://spacy.io/>`_ is used for text parsing and all documents are
    `SpaCy Doc <https://spacy.io/api/doc/>`_ objects with special user data. The SpaCy documents can be accessed by
    using the :attr:`~tmtookit.corpus.spacydocs` function. The SpaCy instance can be accessed via the
    :attr:`~Corpus.nlp` property. Many more properties are defined in the Corpus class.

    The Corpus class allows to attach attributes (or "meta data") to documents and individual tokens inside documents.
    This can be done using the :func:`~tmtoolkit.corpus.set_document_attr` and :func:`~tmtoolkit.corpus.set_token_attr`
    functions.

    Because of the functional programming approach used in tmtoolkit, this class doesn't implement any methods besides
    special Python "dunder" methods to provide dict-like behaviour and (deep)-copy functionality. Functions that operate
    on Corpus objects are defined in the :mod:`~tmtoolkit.corpus` module.

    Parallel processing is implemented for many tasks in order to improve processing speed with large text corpora
    when multiple processors are available. Parallel processing can be enabled setting the ``max_workers`` argument or
    :attr:`~Corpus.max_workers` property to the respective number or proportion of CPUs to be used. A *Reusable
    Process Pool Executor* from the `loky package <https://github.com/joblib/loky/>`_ is used for job scheduling.
    It can be accessed via the :attr:`~Corpus.procexec` property.
    """

    _BUILTIN_CORPORA_LOAD_KWARGS = {
        'en-NewsArticles': {
            'id_column': 'article_id',
            'text_column': 'text',
            'prepend_columns': ['title', 'subtitle']
        },
        'en-News100': {
            'id_column': 'article_id',
            'text_column': 'text',
            'prepend_columns': ['title', 'subtitle']
        },
        'de-parlspeech-v2-sample-bundestag': {
            'id_column': 'parlspeech_row',
            'text_column': 'text',
        },
        'en-parlspeech-v2-sample-houseofcommons': {
            'id_column': 'parlspeech_row',
            'text_column': 'text',
        },
        'es-parlspeech-v2-sample-congreso': {
            'id_column': 'parlspeech_row',
            'text_column': 'text',
        },
        'nl-parlspeech-v2-sample-tweedekamer': {
            'id_column': 'parlspeech_row',
            'text_column': 'text',
        },
    }

    def __init__(self, docs: Optional[Union[Dict[str, str], Sequence[Document]]] = None,
                 language: Optional[str] = None, language_model: Optional[str] = None,
                 load_features: Optional[Collection[str]] = None,
                 add_features: Collection[str] = (),
                 spacy_token_attrs: Optional[Collection[str]] = None,
                 spacy_instance: Optional[spacy.Language] = None,
                 spacy_opts: Optional[dict] = None,
                 punctuation: Optional[Sequence[str]] = None,
                 max_workers: Optional[Union[int, float]] = None,
                 workers_timeout: int = 10) -> None:
        """
        Create a new :class:`Corpus` class using *raw text* data (i.e. the document text as string) from the dict
        `docs` that maps document labels to document text.

        The documents will be parsed right away using a newly generated SpaCy instance or one that is provided via
        `spacy_instance`. If no `spacy_instance` is given, either `language` or `language_model` must be given.

        :param docs: either dict mapping document labels to document text strings or a sequence of
                     :class:`~tmtoolkit.corpus.Document` objects
        :param language: documents language as two-letter ISO 639-1 language code; will be used to load the appropriate
                         `SpaCy language model <https://spacy.io/models>`_ if `language_model` is not set
        :param language_model: `SpaCy language model <https://spacy.io/models>`_ to be loaded if neither `language` nor
                               `spacy_instance` is given
        :param spacy_instance: a SpaCy `Language text-processing pipeline <https://spacy.io/api/language>`_; set this
                               if you want to use your already loaded pipeline, otherwise specify either `language` or
                               `language_model`
        :param load_features: SpaCy pipeline components to load; see
                              `spacy.load <https://spacy.io/api/top-level#spacy.load>`_; only in effective if not
                              providing your own `spacy_instance`; has special feature `vectors` that determines the
                              default language model to load, if no `language_model` is given; by default will use the
                              set provided by "pipeline" model meta information except for NER
        :param add_features: shortcut for providing pipeline components *additional* to the default list in
                             `load_features`
        :param spacy_token_attrs: SpaCy token attributes to be loaded from each parsed document; see attributes list
                                  for `spacy.Token <https://spacy.io/api/token#attributes>`_
        :param spacy_opts: other SpaCy pipeline parameters passed to
                           `spacy.load <https://spacy.io/api/top-level#spacy.load>`_; only in effective if not
                           providing your own `spacy_instance`
        :param punctuation: provide custom punctuation characters list or use default list from
                            :attr:`string.punctuation` and common whitespace characters
        :param max_workers: number of worker processes used for parallel processing; set to None, 0 or 1 to disable
                            parallel processing; set to positive integer to use up to this amount of worker processes;
                            set to negative integer to use all available CPUs except for this amount; set to float in
                            interval [0, 1] to use this proportion of available CPUs
        :param workers_timeout: timeout in seconds until worker processes are stopped
        """

        logger.debug(f'creating new Corpus instance with language "{language.lower()}" / '
                     f'language model "{language_model} / SpaCy instance "{spacy_instance}"')

        # declare public attributes
        #: SpaCy Language instance
        self.nlp: Language
        #: bijective maps (bidirectional dictionaries) for each token attribute that is represented with hashes
        self.bimaps: Dict[str, bidict] = {}
        #: sequence of punctuation characters
        self.punctuation: Sequence[str] = list(string.punctuation) + [' ', '\r', '\n', '\t'] if punctuation is None \
            else punctuation
        #: *Reusable Process Pool Executor* from the `loky package <https://github.com/joblib/loky/>`_ used for job
        #: scheduling
        self.procexec: Optional[ProcessPoolExecutor] = None
        #: timeout in seconds until worker processes are stopped (used for parallel processing)
        self.workers_timeout: int = workers_timeout
        #: max. number of characters to display in :func:`tmtoolkit.corpus.corpus_summary` for document tokens
        self.print_summary_default_max_tokens_string_length: int = 50
        #: max. number of documents to display in :func:`tmtoolkit.corpus.corpus_summary`
        self.print_summary_default_max_documents: int = 10

        # declare private attributes
        #: keyword arguments passed to ``spacy.load`` when creating the Language instance
        self._spacy_opts: Dict[str, Any]
        #: n-grams setting: if 1, use unigrams, else use respective n-grams
        self._ngrams: int = 1
        #: character used to join n-grams
        self._ngrams_join_str: str = ' '
        #: number of workers used in parallel processing; if this is 0 or 1, parallel processing is disabled
        self._n_max_workers: int = 0
        #: document attributes and their defaults
        self._doc_attrs_defaults: Dict[str, Any] = {'label': '', 'has_sents': False}
        #: custom token attributes and their defaults
        self._token_attrs_defaults: Dict[str, Any] = {}
        #: dict that maps document labels to Document objects
        self._docs: Dict[str, Document] = {}
        #: structure to distribute documents among worker processes when using parallel processing;
        #: list is of size N for N workers; each element i in self._workers_docs is a list of variable length that
        #: contains the document labels for the documents assigned to the i-th worker process
        self._workers_docs: List[List[str]] = []

        # set or initialize SpaCy Language instance
        if spacy_instance:
            self.nlp = spacy_instance
            spacy_kwargs = {}
        else:
            if language is None and language_model is None:
                raise ValueError('either `language`, `language_model` or `spacy_instance` must be given')

            if language_model is None:
                if load_features is not None and ('vectors' in load_features or 'vectors' in add_features):
                    model_suffix = 'md'
                else:
                    model_suffix = 'sm'

                # if language_model is not given, load the default language model of the given language
                if not isinstance(language, str) or len(language) != 2:
                    raise ValueError('`language` must be a two-letter ISO 639-1 language code')

                language = language.lower()

                if language not in DEFAULT_LANGUAGE_MODELS:
                    raise ValueError('language "%s" is not supported' % language)
                language_model = DEFAULT_LANGUAGE_MODELS[language] + '_' + model_suffix

            # model meta information
            try:
                model_info = spacy.info(language_model)
            except RuntimeError:
                raise ValueError(f'language model "{language_model}" cannot be loaded; are you sure it is installed?')

            # the default pipeline compenents for SpaCy language models – these would be loaded *and enabled* if not
            # explicitly excluded
            default_components = set(model_info['pipeline'])

            # set the "features", i.e. pipeline components to load
            load_features = default_components.copy() - {'ner'} if load_features is None else set(load_features)
            load_features.update(add_features)

            # set difference with `load_features` in order to get a set of components to be excluded from loading
            # example: "ner" is loaded by default, but not listed in `load_features` -> will be excluded from loading
            spacy_exclude = tuple(default_components - load_features)

            # set keyword arguments passed to `spacy.load`
            spacy_kwargs = dict(exclude=spacy_exclude)
            if spacy_opts:
                spacy_kwargs.update(spacy_opts)

            # load the language model
            self.nlp = spacy.load(language_model, **spacy_kwargs)

            # set difference with `default_components` in order to get a set of components to be enabled after loading;
            # restrict this set to those components that were actually loaded
            # example: "senter" is requested (i.e. it's in `load_features`) but is not enabled by default (but it is
            # loaded) -> will be enabled now
            additional_components = (load_features - set(self.nlp.pipe_names)) & set(self.nlp.component_names)
            for comp in additional_components:
                self.nlp.enable_pipe(comp)

        # store pipeline configuration for possible re-creation of the instance during copy/deserialize
        nlp_conf_allowed_keys = {'lang', 'pipeline', 'disabled', 'before_creation', 'after_creation',
                                 'after_pipeline_creation', 'batch_size'}
        nlp_conf = {k: v for k, v in self.nlp.config['nlp'].items() if k in nlp_conf_allowed_keys}
        self._spacy_opts = spacy_kwargs
        self._spacy_opts.update({'config': {'nlp': nlp_conf}})

        # set number of workers -> this calls the property setter and distributes the documents to the workers
        self.max_workers = max_workers

        # record the SpaCy Token attributes that should be used; they depend on the set of allowed attributes
        # `spacy_token_attrs` and on the loaded and enabled pipelines in `self.nlp.pipe_names`
        spacy_token_attrs_is_default = spacy_token_attrs is None
        spacy_token_attrs = STD_TOKEN_ATTRS if spacy_token_attrs is None else set(spacy_token_attrs)
        if not spacy_token_attrs <= TOKENMAT_ATTRS:
            raise ValueError('all token attributes given in `spacy_token_attrs` must be valid SpaCy token attribute '
                             'names')

        spacy_attrs_checked = []
        for pipeline_comp, token_attrs in SPACY_TOKEN_ATTRS.items():
            if pipeline_comp == '_default' or pipeline_comp in self.nlp.pipe_names:
                spacy_attrs_checked.extend([a for a in token_attrs
                                            if a in spacy_token_attrs and a not in spacy_attrs_checked])

        if not spacy_token_attrs_is_default and spacy_token_attrs != set(spacy_attrs_checked):
            raise ValueError(f'the following SpaCy attributes are not available due to your language model and/or '
                             f'pipeline configuration: {spacy_token_attrs - set(spacy_attrs_checked)}; you should '
                             f'consider adding pipeline components via `load_features` or `add_features` parameter')

        self._spacy_token_attrs = tuple(spacy_attrs_checked)

        # initialize bijective maps for hash <-> token / attr. string conversion
        self._init_bimaps()

        logger.info(f'creating Corpus instance with {"no" if docs is None else len(docs)} documents')
        if self.max_workers <= 1:
            logger.info('using serial processing')
        else:
            logger.info(f'using parallel processing with {self.max_workers} workers')

        if docs is not None:
            if isinstance(docs, Sequence):
                for d in docs:
                    if not isinstance(d, Document):
                        raise ValueError('if `docs` is a Sequence, its values must be Document objects')
                    d.bimaps = self.bimaps
                    self._docs[d.label] = d
            else:
                self._init_docs(docs)

            self._update_bimaps()
            self._update_workers_docs()

        logger.debug(f'finished creating new Corpus instance: {str(self)}')

    def __str__(self) -> str:
        """String representation of this Corpus object"""
        return self.__repr__()

    def __repr__(self) -> str:
        """String representation of this Corpus object"""
        if self.procexec:
            parallel_info = f' / {self.max_workers} worker processes'
        else:
            parallel_info = ''

        return f'<Corpus [{self.n_docs} document{"s" if self.n_docs > 1 else ""} ' \
               f'{parallel_info} / language "{self.language}"]>'

    def __len__(self) -> int:
        """
        Dict method to return number of documents.

        :return: number of documents
        """
        return len(self._docs)

    def __getitem__(self, k: Union[str, int, slice]) -> Union[Document, List[Document]]:
        """
        Dict method for retrieving a document with label, integer index or slice object `k` via ``corpus[<k>]``.

        :param k: if `k` is a string, retrieve the document with that document label; if `k` is an integer, retrieve the
                  document at that position in the list of document labels; if `k` is a slice, return multiple documents
                  corresponding to the selected slice in the list of document labels
        :return: token sequence for document `k` or, if `k` is a slice, a list of token sequences corresponding to the
                 selected slice of documents
        """
        if isinstance(k, slice):
            return [self._docs[lbl] for lbl in self.doc_labels[k]]

        if isinstance(k, int):
            k = self.doc_labels[k]
        elif k not in self.keys():
            raise KeyError(f'document "{k}" not found in corpus')
        return self._docs[k]

    def __setitem__(self, doc_label: str, doc: Union[str, Doc, Document]):
        """
        Dict method for inserting a new document or updating an existing document
        either as text, as `SpaCy Doc <https://spacy.io/api/doc/>`_ object or as :class:`~tmtoolkit.corpus.Document`
        object.

        :param doc_label: document label
        :param doc: document text as string, as `SpaCy Doc <https://spacy.io/api/doc/>`_ object or as
                    :class:`~tmtoolkit.corpus.Document` object
        """
        if not isinstance(doc_label, str):
            raise KeyError('`doc_label` must be a string')

        if not isinstance(doc, (str, Doc, Document)):
            raise ValueError('`doc` must be a string, a spaCy Doc object or a tmtoolkit Document object')

        if isinstance(doc, str):
            doc = self.nlp(doc)   # create Doc object

        if isinstance(doc, Doc):
            doc = self._init_document(doc, label=doc_label)

        # insert or update
        self._docs[doc_label] = doc

        # update bimaps
        self._update_bimaps({doc_label})

        # update assignments of documents to workers
        self._update_workers_docs()

    def __delitem__(self, doc_label):
        """
        Dict method for removing a document with label `doc_label` via ``del corpus[<doc_label>]``.

        :param doc_label: document label
        """
        if doc_label not in self.keys():
            raise KeyError(f'document "{doc_label}" not found in corpus')

        # remove document
        del self._docs[doc_label]

        # update bimaps
        self._update_bimaps()

        # update assignments of documents to workers
        self._update_workers_docs()

    def __iter__(self) -> Iterator[str]:
        """Dict method for iterating through all documents."""
        return self._docs.__iter__()

    def __contains__(self, doc_label) -> bool:
        """
        Dict method for checking whether `doc_label` exists in this corpus.

        :param doc_label: document label
        :return True if `doc_label` exists, else False
        """
        return doc_label in self.keys()

    def __copy__(self) -> Corpus:
        """
        Make a copy of this Corpus, returning a new object with the same data but using the *same* SpaCy instance.

        :return: new Corpus object
        """
        return self._deserialize(self._serialize(deepcopy_attrs=True, store_nlp_instance_pointer=True))

    def __deepcopy__(self, memodict=None) -> Corpus:
        """
        Make a copy of this Corpus, returning a new object with the same data and a *new* SpaCy instance.

        :return: new Corpus object
        """
        return self._deserialize(self._serialize(deepcopy_attrs=True, store_nlp_instance_pointer=False))

    def items(self) -> ItemsView[str, Document]:
        """Dict method to retrieve pairs of document labels and their Document objects."""
        return self._docs.items()

    def keys(self) -> KeysView[str]:
        """Dict method to retrieve document labels of unmasked documents."""
        return self._docs.keys()

    def values(self) -> ValuesView[Document]:
        """Dict method to retrieve Document objects."""
        return self._docs.values()

    def get(self, *args) -> Document:
        """
        Dict method to retrieve a specific document like ``corpus.get(<doc_label>, <default>)``.

        :return: token sequence
        """
        return self._docs.get(*args)

    def update(self, new_docs: Union[Dict[str, Union[str, Doc, Document]], Sequence[Document]]):
        """
        Dict method for inserting new documents or updating existing documents as either:

        - dict mapping document label to text, to `SpaCy Doc <https://spacy.io/api/doc/>`_ objects or to
          :class:`~tmtoolkit.corpus.Document` object;
        - sequence of :class:`~tmtoolkit.corpus.Document` objects

        :param new_docs: dict mapping document labels to text, `SpaCy Doc <https://spacy.io/api/doc/>`_ objects or
                         :class:`~tmtoolkit.corpus.Document` objects; or sequence of :class:`~tmtoolkit.corpus.Document`
                         objects
        """
        if isinstance(new_docs, Sequence):
            new_docs = {d.label: d for d in new_docs}

        logger.debug(f'updating Corpus instance with {len(new_docs)} new documents')

        new_docs_text = {}
        for lbl, d in new_docs.items():
            if isinstance(d, str):
                new_docs_text[lbl] = d
            else:
                if isinstance(d, Doc):
                    d = self._init_document(d, label=lbl)
                elif not isinstance(d, Document):
                    raise ValueError('one or more documents in `new_docs` are neither raw text documents, nor SpaCy '
                                     'documents nor tmtoolkit Documents')

                self._docs[lbl] = d

        if new_docs_text:
            self._init_docs(new_docs_text)

        self._update_bimaps(new_docs.keys())
        self._update_workers_docs()

    @property
    def uses_unigrams(self) -> bool:
        """Returns True when this Corpus is set up for unigram tokens, i.e. :attr:`~Corpus.tokens_processed` is 1."""
        return self._ngrams == 1

    @property
    def spacy_token_attrs(self) -> Tuple[str, ...]:
        """
        Return tuple of available SpaCy token attributes.
        """
        return self._spacy_token_attrs

    @property
    def token_attrs(self) -> Tuple[str, ...]:
        """
        Return tuple of available token attributes (SpaCy attributes like "pos" or "lemma" and custom attributes).
        """
        return self._spacy_token_attrs + tuple(self._token_attrs_defaults.keys())

    @property
    def custom_token_attrs_defaults(self) -> Dict[str, Any]:
        """Return dict of available custom token attributes along with their default values."""
        return self._token_attrs_defaults

    @property
    def doc_attrs(self) -> Tuple[str, ...]:
        """Return list of available document attributes."""
        return tuple(self._doc_attrs_defaults.keys())

    @property
    def doc_attrs_defaults(self) -> Dict[str, Any]:
        """Return list of available document attributes along with their default values."""
        return self._doc_attrs_defaults

    @property
    def ngrams(self) -> int:
        """Return n-gram setting, e.g. *1* if Corpus is set up for unigrams, *2* if set up for bigrams, etc."""
        return self._ngrams

    @property
    def ngrams_join_str(self) -> str:
        """Return string that is used for joining n-grams."""
        return self._ngrams_join_str

    @property
    def language(self) -> str:
        """Return Corpus language as two-letter ISO 639-1 language code."""
        return self.nlp.lang

    @property
    def language_model(self) -> str:
        """Return name of the language model that was loaded."""
        return self.nlp.lang + '_' + self.nlp.meta['name']

    @property
    def has_sents(self) -> bool:
        """Return True if information sentence borders were parsed for documents in this corpus, else return False."""
        return 'parser' in self.nlp.pipe_names or 'senter' in self.nlp.pipe_names

    @property
    def doc_labels(self) -> List[str]:
        """Return document label names."""
        return list(self.keys())

    @property
    def n_docs(self) -> int:
        """Same as :meth:`~Corpus.__len__`."""
        return len(self)

    @property
    def workers_docs(self) -> List[List[str]]:
        """
        When *N* is the number of worker processes for parallel processing, return list of size *N* with each item
        being a list of document labels for the respective worker process. Returns an empty list when parallel
        processing is disabled.
        """
        return self._workers_docs

    @property
    def max_workers(self):
        """Return the number of worker processes for parallel processing."""
        return self._n_max_workers

    @max_workers.setter
    def max_workers(self, max_workers):
        """
        Set the number of worker processes for parallel processing.

        :param max_workers: number of worker processes used for parallel processing; set to None, 0 or 1 to disable
                            parallel processing; set to positive integer to use up to this amount of worker processes;
                            set to negative integer to use all available CPUs except for this amount; set to float in
                            interval [0, 1] to use this proportion of available CPUs
        """
        old_max_workers = self.max_workers

        if max_workers is None:
            self._n_max_workers = 1
        else:
            if not isinstance(max_workers, (int, float)) or \
                    (isinstance(max_workers, float) and not 0 <= max_workers <= 1):
                raise ValueError('`max_workers` must be an integer, a float in [0, 1] or None')

            if isinstance(max_workers, float):
                self._n_max_workers = max(round(mp.cpu_count() * max_workers), 1)
            else:
                if max_workers >= 0:
                   self._n_max_workers = max(max_workers, 1)
                else:
                    self._n_max_workers = max(mp.cpu_count() + max_workers, 1)

        assert self._n_max_workers > 0 and isinstance(self._n_max_workers, int), \
            'self._n_max_workers must be strictly positive integer'

        if self._n_max_workers <= 1:
            logger.debug('setting Corpus instance to serial processing')
        else:
            logger.debug(f'setting Corpus instance to parallel processing with {self._n_max_workers} workers')

        # number of workers has changed
        if old_max_workers != self.max_workers:
            logger.debug('number of workers has changed')
            self.procexec = get_reusable_executor(max_workers=self.max_workers, timeout=self.workers_timeout) \
                if self.max_workers > 1 else None   # self.max_workers == 1 means parallel proc. disabled
            self._update_workers_docs()

    @classmethod
    def from_files(cls, files: Union[str, Collection[str], Dict[str, str]], **kwargs) -> Corpus:
        """
        Construct Corpus object by loading files. Pass arguments for Corpus initialization and file loading as keyword
        arguments via `kwargs`. See :meth:`~tmtoolkit.corpus.Corpus.__init__` for Corpus constructor arguments and
        :func:`~tmtoolkit.corpus.corpus_add_files` for file loading arguments.

        :param files: single file path string or sequence of file paths or dict mapping document label to file path
        :return: Corpus instance
        """
        from ._corpusfuncs import corpus_add_files
        return cls._construct_from_func(corpus_add_files, files, **kwargs)

    @classmethod
    def from_folder(cls, folder: str, **kwargs) -> Corpus:
        """
        Construct Corpus object by loading files from a folder `folder`. Pass arguments for Corpus initialization and
        file loading as keyword arguments via `kwargs`. See :meth:`~tmtoolkit.corpus.Corpus.__init__` for Corpus
        constructor arguments and :func:`~tmtoolkit.corpus.corpus_add_folder` for file loading arguments.

        :param folder: folder from where the files are read
        :return: Corpus instance
        """
        from ._corpusfuncs import corpus_add_folder
        return cls._construct_from_func(corpus_add_folder, folder, **kwargs)

    @classmethod
    def from_tabular(cls, files: Union[str, Collection[str]], **kwargs):
        """
        Construct Corpus object by loading documents from a tabular file, i.e. CSV or Excel file. Pass arguments for
        Corpus initialization and file loading as keyword arguments via `kwargs`. See
        :meth:`~tmtoolkit.corpus.Corpus.__init__` for Corpus constructor arguments and
        :func:`~tmtoolkit.corpus.corpus_add_tabular` for file loading arguments.

        :param files: single string or list of strings with path to file(s) to load
        :return: Corpus instance
        """
        from ._corpusfuncs import corpus_add_tabular
        return cls._construct_from_func(corpus_add_tabular, files, **kwargs)


    @classmethod
    def from_zip(cls, zipfile: str, **kwargs):
        """
        Construct Corpus object by loading files from a ZIP file. Pass arguments for
        Corpus initialization and file loading as keyword arguments via `kwargs`. See
        :meth:`~tmtoolkit.corpus.Corpus.__init__` for Corpus constructor arguments and
        :func:`~tmtoolkit.corpus.corpus_add_zip` for file loading arguments.

        :param zipfile: path to ZIP file to be loaded
        :return: Corpus instance
        """
        from ._corpusfuncs import corpus_add_zip
        return cls._construct_from_func(corpus_add_zip, zipfile, **kwargs)

    @classmethod
    def from_builtin_corpus(cls, corpus_label, **kwargs):
        """
        Construct Corpus object by loading one of the built-in datasets specified by `corpus_label`. To get a list
        of available built-in datasets, use :func:`~tmtoolkit.corpus.builtin_corpora_info`.

        :param corpus_label: the corpus to load (one of the labels listed in
                             :func:`~tmtoolkit.corpus.builtin_corpora_info`)
        :param kwargs: override arguments of Corpus constructor
        :return: Corpus instance
        """
        from tmtoolkit.corpus._corpusfuncs import builtin_corpora_info
        available = builtin_corpora_info(with_paths=True)

        if corpus_label in available:
            load_opts = {
                'add_tabular_opts': cls._BUILTIN_CORPORA_LOAD_KWARGS[corpus_label],
                'language': corpus_label[:2]
            }
            load_opts.update(kwargs)

            return cls.from_zip(available[corpus_label], **load_opts)
        else:
            raise ValueError(f'built-in corpus does not exist: {corpus_label}')

    def _nlppipe(self, docs: ValuesView[str]) -> Union[Iterator[Doc], Generator[Doc]]:
        """
        Helper method to set up the SpaCy pipeline.
        """
        if self.max_workers > 1:   # pipeline for parallel processing
            logger.debug(f'using parallel processing NLP pipeline with {self.max_workers} workers')
            return self.nlp.pipe(docs, n_process=self.max_workers)
        else:   # serial processing
            logger.debug('using serial processing NLP pipeline')
            return (self.nlp(txt) for txt in docs)

    def _init_bimaps(self):
        """
        Initialize bijective maps for hash <-> token / attr. string conversion.
        """
        for attr in ('whitespace', 'token', ) + self._spacy_token_attrs:
            if attr not in BOOLEAN_SPACY_TOKEN_ATTRS:
                self.bimaps[attr] = bidict()

    def _init_docs(self, docs: Dict[str, str]):
        """
        Helper method to process the raw text documents using a SpaCy pipeline and initialize the Document objects.
        """
        pipe = self._nlppipe(docs.values())

        # tokenize each document which yields a Document object `d` for each document label `lbl`
        logger.debug(f'initializing {len(docs)} new documents')
        for lbl, sp_d in dict(zip(docs.keys(), pipe)).items():
            self._docs[lbl] = self._init_document(sp_d, label=lbl)

    def _init_document(self, spacydoc: Doc, label: str):
        """Helper method to create a new tmtoolkit `Document` object from a SpaCy document `spacydoc`."""
        # somehow, the whitespace attribute is only available as string attribute, not as hash
        whitespace = np.array([self.nlp.vocab.strings[t.whitespace_] for t in spacydoc], dtype='uint64')\
            .reshape((len(spacydoc), 1))
        load_token_attrs = ['orth']

        # unfortunately, spacydoc.is_sentenced cannot be trusted: it is always True for empty documents or documents
        # without sentences even if the sentences were not parsed; hence we check the SpaCy pipeline in `self.has_sents`

        if self.has_sents:
            load_token_attrs.append('sent_start')

        load_token_attrs.extend(self.spacy_token_attrs)

        # get token attributes as matrix of uin64 hashes
        spacy_token_attrs = spacydoc.to_array(load_token_attrs)

        # get document attributes as dict
        spacydoc_attrs = set(dir(spacydoc._)) - {'get', 'has', 'set', 'label', 'has_sents'}
        doc_attrs = {a: getattr(spacydoc._, a) for a in spacydoc_attrs}

        # construct Document object
        return Document(self.bimaps, label,
                        has_sents=self.has_sents,
                        tokenmat=np.hstack((whitespace, spacy_token_attrs)),
                        doc_attrs=doc_attrs,
                        tokenmat_attrs=self.spacy_token_attrs)

    def _update_bimaps(self, which_docs: Union[str, Optional[Collection[str]]] = None,
                       which_attrs: Union[str, Optional[Collection[str]]] = None):
        """Helper function to update bijective maps in `self.bimaps`."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'updating bimaps for documents: '
                         f'"{str(which_docs) if which_docs is not None else "all"}" / '
                         f'attributes: "{str(which_attrs) if which_attrs is not None else "all"}"')
        all_docs = False
        if isinstance(which_docs, str):
            which_docs = (which_docs, )
        elif which_docs is None:
            which_docs = self.keys()
            all_docs = True

        if isinstance(which_attrs, str):
            which_attrs = (which_attrs, )
        elif which_attrs is None:
            which_attrs = list(self.bimaps.keys())   # copy keys

        for attr in which_attrs:
            bimap = self.bimaps[attr]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'bimap size for attribute "{attr}" before update is {len(bimap)}')
            unique_attr_hashes = set()
            for lbl in which_docs:
                d = self._docs[lbl]
                hashes = []
                strings = []
                # iterate through hashes `h` for the attribute `attr` in document `d`
                try:
                    attr_hashes = set(d.tokenmat[:, d.tokenmat_attrs.index(attr)])
                except ValueError:  # `attr` not in `d.tokenmat_attrs`, i.e. the attribute is not defined in a document;
                    # this happens when new documents are loaded which don't contain the same token attributes
                    # that were defined before in the corpus

                    # remove the attribute from the bimaps dict
                    bimap = None
                    del self.bimaps[attr]
                    # remove the attribute from all documents if it exists there somewhere
                    for d in self._docs.values():
                        if attr in d.tokenmat_attrs:
                            d.tokenmat_attrs.remove(attr)
                    break

                for h in attr_hashes:
                    # this hash is unknown so far
                    if h not in bimap:
                        # collect hash and its string representation
                        hashes.append(h)
                        strings.append(self.nlp.vocab.strings[h])

                # update bimap
                bimap.update(zip(hashes, strings))

                # update unique hashes
                unique_attr_hashes.update(attr_hashes)

            if all_docs and bimap is not None:  # only remove unused hashes if all documents' hashes were checked
                unused_hashes = set(bimap.keys()) - set(unique_attr_hashes)
                for h in unused_hashes:
                    del bimap[h]

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'bimap size for attribute "{attr}" after update is {len(bimap)}')

    def _update_workers_docs(self):
        """Helper method to update the worker <-> document assignments."""
        if self.max_workers > 1 and self._docs:     # parallel processing enabled
            # make assignments based on number of tokens per document
            logger.debug(f'updating document assignments for {self.max_workers} workers')
            self._workers_docs = greedy_partitioning({lbl: len(d) for lbl, d in self._docs.items()},
                                                     k=self.max_workers, return_only_labels=True)
        else:   # parallel processing disabled or no documents
            logger.debug(f'purging document assignments (parallel proc. disabled or empty corpus)')
            self._workers_docs = []

    def _serialize(self, deepcopy_attrs: bool, store_nlp_instance_pointer: bool, documents: bool = True) \
            -> Dict[str, Any]:
        """
        Helper method to serialize this Corpus object to a dict.

        If `store_nlp_instance_pointer` is True, a pointer to the current SpaCy instance will be stored, otherwise
        the language model name will be stored instead. For deserialization this means that for the former option the
        same SpaCy instance as the current one will be used in a deserialized object and for the latter options this
        means that a new SpaCy instance with the same language model (and options) will be loaded. The former should
        only be used for a local shallow copy, the latter for deep copies and storing serialized Corpus instances to
        disk.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'serializing Corpus instance {"with" if deepcopy_attrs else "without"} attrib. deepcopy, '
                         f'{"with" if store_nlp_instance_pointer else "without"} SpaCy NLP pipeline instance pointer, '
                         f'{"with" if documents else "without"} documents')

        state_attrs = {'state': {}}
        attr_deny = {'nlp', 'procexec', 'spacydocs', 'workers_docs',
                     '_docs', '_n_max_workers', '_workers_docs'}

        if not attr_deny:
            attr_deny.update('bimaps')

        # 1. general object attributes
        for attr in dir(self):
            # dismiss "dunder" attributes, all-caps attributes and attributes in deny list
            if attr.startswith('__') or attr.isupper() or attr in attr_deny:
                continue

            # dismiss methods and properties
            classattr = getattr(type(self), attr, None)
            if classattr is not None and (callable(classattr) or isinstance(classattr, property)):
                continue

            # all others are copied
            attr_obj = getattr(self, attr)
            if deepcopy_attrs:
                state_attrs['state'][attr] = deepcopy(attr_obj)
            else:
                state_attrs['state'][attr] = attr_obj

        state_attrs['max_workers'] = self.max_workers

        # 2. documents
        if documents:
            state_attrs['docs_data'] = [d._serialize(store_bimaps_pointer=False) for d in self.values()]
        else:
            state_attrs['docs_data'] = []

        if store_nlp_instance_pointer:
            state_attrs['spacy_instance'] = self.nlp
        else:
            state_attrs['spacy_instance'] = self.language_model

        return state_attrs

    @classmethod
    def _deserialize(cls, data: Dict[str, Any]) -> Corpus:
        """
        Helper method to deserialize a Corpus object from a dict. All SpaCy documents must be in
        ``data['spacy_data']`` as `DocBin <https://spacy.io/api/docbin/>`_ object.
        """

        logger.debug('deserializing Corpus instance')

        # load a SpaCy language pipeline
        if isinstance(data['spacy_instance'], str):
            # a language model name is given -> will load a new SpaCy model with the same language model and with
            # parameters given in _spacy_opts
            kwargs = dict(language_model=data['spacy_instance'], spacy_opts=data['state']['_spacy_opts'])
            logger.debug(f'will load a new SpaCy model "{data["spacy_instance"]}"')
        elif isinstance(data['spacy_instance'], Language):
            # a SpaCy instance is given -> will use this right away
            kwargs = dict(spacy_instance=data['spacy_instance'])
            logger.debug(f'will reuse instantiated SpaCy model "{data["spacy_instance"]}"')
        else:
            raise ValueError('spacy_instance in serialized data must be either a language model name string or a '
                             '`Language` instance')

        # create the Corpus instance
        instance = cls(max_workers=data['max_workers'], workers_timeout=data['state']['workers_timeout'],
                       **kwargs)

        # set all other properties
        for attr, val in data['state'].items():
            if attr not in {'_spacy_opts', 'workers_timeout'}:
                setattr(instance, attr, val)

        # load documents
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'deserializing {len(data["docs_data"])} documents')
        instance.update([Document._deserialize(d_data, bimaps=instance.bimaps) for d_data in data['docs_data']])

        return instance

    @classmethod
    def _construct_from_func(cls, add_fn: Callable, *args, **kwargs) -> Corpus:
        add_fn_args, corpus_args = split_func_args(add_fn, kwargs)
        add_fn_args['inplace'] = True

        corp = cls(**corpus_args)
        add_fn(corp, *args, **add_fn_args)
        return corp

