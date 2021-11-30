"""
Internal module that implements :class:`Corpus` class representing text as token sequences in labelled documents.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

from __future__ import annotations  # req. for classmethod return type; see https://stackoverflow.com/a/49872353
import multiprocessing as mp
import string
from copy import deepcopy
from typing import Dict, Union, List, Optional, Any, Iterator, Callable, Sequence, ItemsView, KeysView, ValuesView, \
    Generator

import spacy
from spacy import Language
from spacy.tokens import Doc
from loky import get_reusable_executor

from ._common import DEFAULT_LANGUAGE_MODELS, SPACY_TOKEN_ATTRS
from ._document import Document
from ..utils import greedy_partitioning, split_func_args
from ..types import OrdStrCollection, UnordStrCollection


class Corpus:
    """
    The Corpus class represents text as *string token sequences* in labelled documents. It behaves like a Python dict,
    i.e. you can access document tokens via square brackets (``corp['my_doc']``).

    `SpaCy <https://spacy.io/>`_ is used for text parsing and all documents are
    `SpaCy Doc <https://spacy.io/api/doc/>`_ objects with special user data. The SpaCy documents can be accessed using
    the :attr:`~Corpus.spacydocs` property. The SpaCy instance can be accessed via the :attr:`~Corpus.nlp` property.
    Many more properties are defined in the Corpus class.

    The Corpus class allows to attach attributes (or "meta data") to documents and individual tokens inside documents.
    This can be done using the :func:`~tmtoolkit.corpus.set_document_attr` and :func:`~tmtoolkit.corpus.set_token_attr`
    functions. A special attribute at document and token level is the *mask*. It allows for filtering documents and/or
    tokens. It is implemented as a boolean array where *1* indicates that the document or token is *unmasked* or
    "active" and *0* indicates that the document or token is *masked* or "inactive".

    Because of the functional programming approach used in tmtoolkit, this class doesn't implement any methods besides
    special Python "dunder" methods to provide dict-like behaviour and (deep)-copy functionality. Functions that operate
    on Corpus objects are defined in the :mod:`~tmtoolkit.corpus` module.

    Parallel processing is implemented for many tasks in order to improve processing speed with large text corpora
    when multiple processors are available. Parallel processing can be enabled setting the ``max_workers`` argument or
    :attr:`~Corpus.max_workers` property to the respective number or proportion of CPUs to be used. A *Reusable
    Process Pool Executor* from the `joblib package <https://github.com/joblib/loky/>`_ is used for job scheduling.
    It can be accessed via the :attr:`~Corpus.procexec` property.
    """

    _BUILTIN_CORPORA_LOAD_KWARGS = {
        'en-NewsArticles': {
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
                 load_features: UnordStrCollection = ('tok2vec', 'tagger', 'morphologizer', 'parser',
                                                      'attribute_ruler', 'lemmatizer'),
                 add_features: UnordStrCollection = (),
                 spacy_instance: Optional[Any] = None,
                 spacy_opts: Optional[dict] = None,
                 punctuation: Optional[OrdStrCollection] = None,
                 max_workers: Optional[Union[int, float]] = None,
                 workers_timeout: int = 10):
        """
        TODO: param to set token attr. to extract instead of STD_TOKEN_ATTRS / replace load/add features?

        Create a new :class:`Corpus` class using *raw text* data (i.e. the document text as string) from the dict
        `docs` that maps document labels to document text.

        The documents will be parsed right away using a newly generated SpaCy instance or one that is provided via
        `spacy_instance`. If no `spacy_instance` is given, either `language` or `language_model` must be given.

        :param docs: either dict mapping document labels to document text strings or a SpaCy
                     `DocBin <https://spacy.io/api/docbin/>`_ object
        :param language: documents language as two-letter ISO 639-1 language code; will be used to load the appropriate
                         `SpaCy language model <https://spacy.io/models>`_ if `language_model` is not set
        :param language_model: `SpaCy language model <https://spacy.io/models>`_ to be loaded if neither `language` nor
                               `spacy_instance` is given
        :param spacy_instance: a SpaCy `Language text-processing pipeline <https://spacy.io/api/language>`_; set this
                               if you want to use your already loaded pipeline, otherwise specify either `language` or
                               `language_model`
        :param load_features: SpaCy pipeline components to load; see
                              `spacy.load <https://spacy.io/api/top-level#spacy.load>_; only in effective if not
                              providing your own `spacy_instance`; has special feature `vectors` that determines the
                              default language model to load, if no `language_model` is given
        :param add_features: shortcut for providing pipeline components *additional* to the default list in
                             `load_features`
        :param spacy_opts: other SpaCy pipeline parameters passed to
                           `spacy.load <https://spacy.io/api/top-level#spacy.load>_; only in effective if not
                           providing your own `spacy_instance`
        :param punctuation: provide custom punctuation characters list or use default list from
                            :attr:`string.punctuation` and common whitespace characters
        :param max_workers: number of worker processes used for parallel processing; set to None, 0 or 1 to disable
                            parallel processing; set to positive integer to use up to this amount of worker processes;
                            set to negative integer to use all available CPUs except for this amount; set to float in
                            interval [0, 1] to use this proportion of available CPUs
        :param workers_timeout: timeout in seconds until worker processes are stopped
        """
        self.print_summary_default_max_tokens_string_length = 50
        self.print_summary_default_max_documents = 10

        if spacy_instance:
            self.nlp = spacy_instance
            self._spacy_opts = {}       # can't know the options with which this instance was created
        else:
            if language is None and language_model is None:
                raise ValueError('either `language`, `language_model` or `spacy_instance` must be given')

            load_features = set(load_features)
            load_features.update(set(add_features))
            load_vectors = 'vectors' in load_features
            if load_vectors:
                load_features.remove('vectors')
                model_suffix = 'md'
            else:
                model_suffix = 'sm'

            if language_model is None:
                if not isinstance(language, str) or len(language) != 2:
                    raise ValueError('`language` must be a two-letter ISO 639-1 language code')

                if language not in DEFAULT_LANGUAGE_MODELS:
                    raise ValueError('language "%s" is not supported' % language)
                language_model = DEFAULT_LANGUAGE_MODELS[language] + '_' + model_suffix

            default_components = {'tok2vec', 'tagger', 'morphologizer', 'parser', 'attribute_ruler',
                                  'lemmatizer', 'ner'}
            spacy_exclude = tuple(default_components - load_features)
            spacy_kwargs = dict(exclude=spacy_exclude)
            if spacy_opts:
                spacy_kwargs.update(spacy_opts)

            self.nlp = spacy.load(language_model, **spacy_kwargs)

            additional_components = tuple(load_features - default_components)
            for comp in additional_components:
                self.nlp.enable_pipe(comp)

            self._spacy_opts = spacy_kwargs     # used for possible re-creation of the instance during copy/deserialize

        self.punctuation = list(string.punctuation) + [' ', '\r', '\n', '\t'] if punctuation is None else punctuation
        self.procexec = None
        self._ngrams = 1
        self._ngrams_join_str = ' '
        self._n_max_workers = 0
        # document attribute name -> attribute default value
        self._doc_attrs_defaults = {'label': '', 'has_sents': False}    # type: Dict[str, Any]
        # token attribute name -> attribute default value
        self._token_attrs_defaults = {}  # type: Dict[str, Any]
        self._docs = {}             # type: Dict[str, Document]
        self._workers_docs = []     # type: List[List[str]]

        self.workers_timeout = workers_timeout
        self.max_workers = max_workers

        if docs is not None:
            if isinstance(docs, Sequence):
                for d in docs:
                    d.vocab = self.nlp.vocab
                    self._docs[d.label] = d
            else:
                self._init_docs(docs)

            self._update_workers_docs()

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
        from ._helpers import _init_document

        if not isinstance(doc_label, str):
            raise KeyError('`doc_label` must be a string')

        if not isinstance(doc, (str, Doc, Document)):
            raise ValueError('`doc` must be a string, a spaCy Doc object or a tmtoolkit Document object')

        if isinstance(doc, str):
            doc = self.nlp(doc)   # create Doc object

        if isinstance(doc, Doc):
            doc = _init_document(self.nlp.vocab, doc, label=doc_label, token_attrs=SPACY_TOKEN_ATTRS)

        # insert or update
        self._docs[doc_label] = doc

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

    def get(self, *args) -> List[str]:
        """
        Dict method to retrieve a specific document like ``corpus.get(<doc_label>, <default>)``.

        :return: token sequence
        """
        return self._docs.get(*args)

    def update(self, new_docs: Dict[str, Union[str, Doc, Document]]):
        """
        Dict method for inserting new documents or updating existing documents
        either as text, as `SpaCy Doc <https://spacy.io/api/doc/>`_ object or as :class:`~tmtoolkit.corpus.Document`
        object.

        :param new_docs: dict mapping document labels to text, `SpaCy Doc <https://spacy.io/api/doc/>`_ objects or
                         :class:`~tmtoolkit.corpus.Document` objects
        """
        from ._helpers import _init_document

        new_docs_text = {}
        for lbl, d in new_docs.items():
            if isinstance(d, str):
                new_docs_text[lbl] = d
            else:
                if isinstance(d, Doc):
                    d = _init_document(self.nlp.vocab, d, label=lbl, token_attrs=SPACY_TOKEN_ATTRS)
                elif not isinstance(d, Document):
                    raise ValueError('one or more documents in `new_docs` are neither raw text documents, nor SpaCy '
                                     'documents nor tmtoolkit Documents')

                self._docs[lbl] = d

        if new_docs_text:
            self._init_docs(new_docs_text)

        self._update_workers_docs()

    @property
    def uses_unigrams(self) -> bool:
        """Returns True when this Corpus is set up for unigram tokens, i.e. :attr:`~Corpus.tokens_processed` is 1."""
        return self._ngrams == 1

    @property
    def token_attrs(self) -> List[str]:
        """
        Return list of available token attributes (standard attributes like "pos" or "lemma" and custom attributes).
        """
        return list(SPACY_TOKEN_ATTRS) + list(self._token_attrs_defaults.keys())

    @property
    def custom_token_attrs_defaults(self) -> Dict[str, Any]:
        """Return dict of available custom token attributes along with their default values."""
        return self._token_attrs_defaults

    @property
    def doc_attrs(self) -> List[str]:
        """Return list of available document attributes."""
        return list(self._doc_attrs_defaults.keys())

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
    def doc_labels(self) -> List[str]:
        """Return document label names."""
        return list(self.keys())

    @property
    def n_docs(self) -> int:
        """Same as :meth:`~Corpus.__len__`."""
        return len(self)

    @property
    def spacydocs(self) -> Dict[str, Doc]:
        """
        Return dict mapping document labels to `SpaCy Doc <https://spacy.io/api/doc/>`_ objects.
        """

        # need to re-generate the SpaCy documents here, since the document texts could have been changed during
        # processing (e.g. token transformation, filtering, etc.)
        from ._corpusfuncs import doc_texts

        # set document extensions for document attributes
        for attr, default in self.doc_attrs_defaults.items():
            Doc.set_extension(attr, default=default, force=True)

        # set up
        pipe = self._nlppipe(doc_texts(self).values())
        sp_docs = {}
        for d, sp_d in dict(zip(self.values(), pipe)).items():
            for attr, val in d.doc_attrs.items():
                setattr(sp_d._, attr, val)
            sp_docs[sp_d._.label] = sp_d

        return sp_docs

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

        assert self._n_max_workers > 0, 'self._n_max_workers must be strictly positive'

        # number of workers has changed
        if old_max_workers != self.max_workers:
            self.procexec = get_reusable_executor(max_workers=self.max_workers, timeout=self.workers_timeout) \
                if self.max_workers > 1 else None   # self.max_workers == 1 means parallel proc. disabled
            self._update_workers_docs()

    @classmethod
    def from_files(cls, files: Union[str, UnordStrCollection, Dict[str, str]], **kwargs) -> Corpus:
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
    def from_tabular(cls, files: Union[str, UnordStrCollection], **kwargs):
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

    def _nlppipe(self, docs: ValuesView[str]) -> Generator[Doc]:
        """
        Helper method to set up the SpaCy pipeline.
        """
        if self.max_workers > 1:   # pipeline for parallel processing
            return self.nlp.pipe(docs, n_process=self.max_workers)
        else:   # serial processing
            return (self.nlp(txt) for txt in docs)

    def _init_docs(self, docs: Dict[str, str]):
        """
        Helper method to process the raw text documents using a SpaCy pipeline and initialize the Document objects.
        """
        from ._helpers import _init_document

        pipe = self._nlppipe(docs.values())

        # tokenize each document which yields a Document object `d` for each document label `lbl`
        for lbl, sp_d in dict(zip(docs.keys(), pipe)).items():
            self._docs[lbl] = _init_document(self.nlp.vocab, sp_d, label=lbl, token_attrs=SPACY_TOKEN_ATTRS)

    def _update_workers_docs(self):
        """Helper method to update the worker <-> document assignments."""
        if self.max_workers > 1 and self._docs:     # parallel processing enabled
            # make assignments based on number of tokens per document
            self._workers_docs = greedy_partitioning({lbl: len(d) for lbl, d in self._docs.items()},
                                                     k=self.max_workers, return_only_labels=True)
        else:   # parallel processing disabled or no documents
            self._workers_docs = []

    def _serialize(self, deepcopy_attrs: bool, store_nlp_instance_pointer: bool) -> Dict[str, Any]:
        """
        Helper method to serialize this Corpus object to a dict.

        If `store_nlp_instance_pointer` is True, a pointer to the current SpaCy instance will be stored, otherwise
        the language model name will be stored instead. For deserialization this means that for the former option the
        same SpaCy instance as the current one will be used in a deserialized object and for the latter options this
        means that a new SpaCy instance with the same language model (and options) will be loaded. The former should
        only be used for a local shallow copy, the latter for deep copies and storing serialized Corpus instances to
        disk.
        """
        state_attrs = {'state': {}}
        attr_deny = {'nlp', 'procexec', 'spacydocs', 'workers_docs',
                     '_docs', '_n_max_workers', '_workers_docs'}

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
        state_attrs['docs_data'] = [d._serialize(store_vocab_instance_pointer=False) for d in self.values()]

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
        # load documents
        docs = [Document._deserialize(d_data) for d_data in data['docs_data']]

        # load a SpaCy language pipeline
        if isinstance(data['spacy_instance'], str):
            # a language model name is given -> will load a new SpaCy model with the same language model and with
            # parameters given in _spacy_opts
            kwargs = dict(language_model=data['spacy_instance'], spacy_opts=data['state'].pop('_spacy_opts'))
        elif isinstance(data['spacy_instance'], Language):
            # a SpaCy instance is given -> will use this right away
            kwargs = dict(spacy_instance=data['spacy_instance'])
        else:
            raise ValueError('spacy_instance in serialized data must be either a language model name string or a '
                             '`Language` instance')

        # create the Corpus instance
        instance = cls(docs, max_workers=data['max_workers'], workers_timeout=data['state'].pop('workers_timeout'),
                       **kwargs)

        # set all other properties
        for attr, val in data['state'].items():
            setattr(instance, attr, val)

        return instance

    @classmethod
    def _construct_from_func(cls, add_fn: Callable, *args, **kwargs) -> Corpus:
        add_fn_args, corpus_args = split_func_args(add_fn, kwargs)
        add_fn_args['inplace'] = True

        corp = cls(**corpus_args)
        add_fn(corp, *args, **add_fn_args)
        return corp

