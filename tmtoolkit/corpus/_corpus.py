"""
Internal module that implements :class:`Corpus` class representing text as token sequences in labelled documents.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import multiprocessing as mp
import string
from copy import deepcopy
from typing import Dict, Union, List, Optional, Any, Iterator

import spacy
from spacy import Language
from spacy.tokens import Doc, DocBin
from loky import get_reusable_executor

from ._common import DEFAULT_LANGUAGE_MODELS
from ..utils import greedy_partitioning
from ..types import OrdStrCollection


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

    STD_TOKEN_ATTRS = ['whitespace', 'pos', 'lemma']

    def __init__(self, docs: Optional[Union[Dict[str, str], DocBin]] = None,
                 language: Optional[str] = None, language_model: Optional[str] = None,
                 spacy_instance: Optional[Any] = None,
                 spacy_exclude: Optional[OrdStrCollection] = ('parser', 'ner'),
                 spacy_opts: Optional[dict] = None,
                 punctuation: Optional[OrdStrCollection] = None,
                 max_workers: Optional[Union[int, float]] = None,
                 workers_timeout: int = 10):
        """
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
        :param spacy_exclude: SpaCy pipeline components to exclude (i.e. not load); see
                              `spacy.load <https://spacy.io/api/top-level#spacy.load>_; only in effective if not
                               providing your own `spacy_instance`
        :param spacy_opts: other SpaCy pipeline parameters passed to
                           `spacy.load <https://spacy.io/api/top-level#spacy.load>_; only in effective if not
                           providing your own `spacy_instance`
        :param punctuation: provide custom punctuation characters list or use default list from
                            :attr:`string.punctuation`
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

            if language_model is None:
                if not isinstance(language, str) or len(language) != 2:
                    raise ValueError('`language` must be a two-letter ISO 639-1 language code')

                if language not in DEFAULT_LANGUAGE_MODELS:
                    raise ValueError('language "%s" is not supported' % language)
                language_model = DEFAULT_LANGUAGE_MODELS[language] + '_sm'

            spacy_kwargs = dict(exclude=spacy_exclude)
            if spacy_opts:
                spacy_kwargs.update(spacy_opts)

            self.nlp = spacy.load(language_model, **spacy_kwargs)
            self._spacy_opts = spacy_kwargs     # used for possible re-creation of the instance during copy/deserialize

        self.punctuation = list(string.punctuation) + [' ', '\r', '\n', '\t'] if punctuation is None else punctuation
        self.procexec = None
        self._tokens_masked = False
        self._tokens_processed = False
        self._ngrams = 1
        self._ngrams_join_str = ' '
        self._n_max_workers = 0
        self._ignore_doc_filter = False
        self._docs = {}
        self._doc_attrs_defaults = {}     # document attribute name -> attribute default value
        self._token_attrs_defaults = {}   # token attribute name -> attribute default value
        self._workers_docs = []

        self.workers_timeout = workers_timeout
        self.max_workers = max_workers

        if docs is not None:
            if isinstance(docs, DocBin):
                self._docs = {d._.label: d for d in docs.get_docs(self.nlp.vocab)}
            else:
                self._tokenize(docs)

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

        return f'<Corpus [{self.n_docs} document{"s" if self.n_docs > 1 else ""} ({self.n_docs_masked} masked)' \
               f'{parallel_info} / language "{self.language}"]>'

    def __len__(self) -> int:
        """
        Dict method to return number of documents -- taking into account the document filter.

        :return: number of documents
        """
        return len(self.spacydocs)

    def __getitem__(self, doc_label) -> List[str]:
        """
        Dict method for retrieving a document with label `doc_label` via ``corpus[<doc_label>]``.

        This method doesn't prevent you from retrieving a masked document.

        :return: token sequence for document `doc_label`
        """
        if doc_label not in self.spacydocs_ignore_filter.keys():
            raise KeyError('document `%s` not found in corpus' % doc_label)
        return self.docs[doc_label]

    def __setitem__(self, doc_label: str, doc: Union[str, Doc]):
        """
        Dict method for inserting a new document or updating an existing document
        either as text or as `SpaCy Doc <https://spacy.io/api/doc/>`_ object.

        :param doc_label: document label
        :param doc: document text as string or a `SpaCy Doc <https://spacy.io/api/doc/>`_ object
        """
        from ._helpers import _init_spacy_doc

        if not isinstance(doc_label, str):
            raise KeyError('`doc_label` must be a string')

        if not isinstance(doc, (str, Doc)):
            raise ValueError('`doc` must be a string or spaCy Doc object')

        if isinstance(doc, str):
            doc = self.nlp(doc)   # create Doc object

        # initialize Doc object
        _init_spacy_doc(doc, doc_label, additional_attrs=self._token_attrs_defaults)

        # insert or update
        self._docs[doc_label] = doc

        # update assignments of documents to workers
        self._update_workers_docs()

    def __delitem__(self, doc_label):
        """
        Dict method for removing a document with label `doc_label` via ``del corpus[<doc_label>]``.

        :param doc_label: document label
        """
        if doc_label not in self.spacydocs_ignore_filter.keys():
            raise KeyError('document `%s` not found in corpus' % doc_label)

        # remove document
        del self._docs[doc_label]

        # update assignments of documents to workers
        self._update_workers_docs()

    def __iter__(self) -> Iterator[str]:
        """Dict method for iterating through all unmasked documents."""
        return self.spacydocs.__iter__()

    def __contains__(self, doc_label) -> bool:
        """
        Dict method for checking whether `doc_label` exists in this corpus.

        :param doc_label: document label
        :return True if `doc_label` exists, else False
        """
        return doc_label in self.spacydocs.keys()

    def __copy__(self):
        """
        Make a copy of this Corpus, returning a new object with the same data but using the *same* SpaCy instance.

        :return: new Corpus object
        """
        return self._deserialize(self._serialize(deepcopy_attrs=True, store_nlp_instance_pointer=True))

    def __deepcopy__(self, memodict=None):
        """
        Make a copy of this Corpus, returning a new object with the same data and a *new* SpaCy instance.

        :return: new Corpus object
        """
        return self._deserialize(self._serialize(deepcopy_attrs=True, store_nlp_instance_pointer=False))

    def items(self):
        """Dict method to retrieve pairs of document labels and tokens of unmasked documents."""
        return self.docs.items()

    def keys(self):
        """Dict method to retrieve document labels of unmasked documents."""
        return self.spacydocs.keys()   # using "spacydocs" here is a bit faster b/c we don't call `doc_tokens`

    def values(self):
        """Dict method to retrieve unmasked document tokens."""
        return self.docs.values()

    def get(self, *args) -> List[str]:
        """
        Dict method to retrieve a specific document like ``corpus.get(<doc_label>, <default>)``.

        This method doesn't prevent you from retrieving a masked document.

        :return: token sequence
        """
        return self.docs.get(*args)

    @property
    def docs_filtered(self) -> bool:
        """Return True when any document in this Corpus is masked/filtered."""
        for d in self._docs.values():
            if not d._.mask:
                return True
        return False

    @property
    def tokens_filtered(self) -> bool:
        """Return True when any token in this Corpus is masked/filtered."""
        return self._tokens_masked

    @property
    def is_filtered(self) -> bool:
        """Return True when any document or any token in this Corpus is masked/filtered."""
        return self.tokens_filtered or self.docs_filtered

    @property
    def tokens_processed(self) -> bool:
        """
        Return True when any tokens in this Corpus were somehow changed/transformed, i.e. when they may not be the same
        as the original tokens from the SpaCy documents (e.g. transformed to all lowercase, lemmatized, etc.).
        """
        return self._tokens_processed

    @property
    def is_processed(self) -> bool:
        """Alias for :attr:`~Corpus.tokens_processed`."""
        return self.tokens_processed

    @property
    def uses_unigrams(self) -> bool:
        """Returns True when this Corpus is set up for unigram tokens, i.e. :attr:`~Corpus.tokens_processed` is 1."""
        return self._ngrams == 1

    @property
    def token_attrs(self) -> List[str]:
        """
        Return list of available token attributes (standard attrbitues like "pos" or "lemma" and custom attributes).
        """
        return self.STD_TOKEN_ATTRS + list(self._token_attrs_defaults.keys())

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
        return sorted(self.keys())

    @property
    def docs(self) -> Dict[str, List[str]]:
        """
        Return dict mapping document labels of filtered documents to filtered token sequences (same output as
        :func:`~tmtoolkit.corpus.doc_tokens` without additional arguments).
        """
        from ._corpusfuncs import doc_tokens
        return doc_tokens(self._docs)

    @property
    def n_docs(self) -> int:
        """Same as :meth:`~Corpus.__len__`."""
        return len(self)

    @property
    def n_docs_masked(self) -> int:
        """Return number of masked/filtered documents."""
        return len(self.spacydocs_ignore_filter) - self.n_docs

    @property
    def ignore_doc_filter(self) -> bool:
        """Status of ignoring the document filter mask. If True, the document filter mask is disabled."""
        return self._ignore_doc_filter

    @ignore_doc_filter.setter
    def ignore_doc_filter(self, ignore: bool):
        """
        Enable / disable document filter. If set to True, the document filter mask is disabled, i.e. ignored.

        :param ignore: if True, the document filter mask is disabled, i.e. ignored
        """
        self._ignore_doc_filter = ignore

    @property
    def spacydocs(self) -> Dict[str, Doc]:
        """
        Return dict mapping document labels to `SpaCy Doc <https://spacy.io/api/doc/>`_ objects, respecting the
        document mask unless :attr:`~Corpus.ignore_doc_filter` is True.
        """
        if self._ignore_doc_filter or not self.docs_filtered:
            return self.spacydocs_ignore_filter
        else:
            return {lbl: d for lbl, d in self._docs.items() if d._.mask}

    @property
    def spacydocs_ignore_filter(self) -> Dict[str, Doc]:
        """
        Return dict mapping document labels to `SpaCy Doc <https://spacy.io/api/doc/>`_ objects, ignoring the
        document mask.
        """
        return self._docs

    @spacydocs.setter
    def spacydocs(self, docs: Dict[str, Doc]):
        """
        Set all documents of this Corpus as dict mapping document labels to `SpaCy Doc <https://spacy.io/api/doc/>`_
        objects.

        :param docs: dict mapping document labels to `SpaCy Doc <https://spacy.io/api/doc/>`_ objects
        """
        self._docs = docs
        self._update_workers_docs()

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

    def _tokenize(self, docs: Dict[str, str]):
        """Helper method to tokenize the raw text documents using a SpaCy pipeline."""
        from ._helpers import _init_spacy_doc

        # set up the SpaCy pipeline
        if self.max_workers > 1:   # pipeline for parallel processing
            tokenizerpipe = self.nlp.pipe(docs.values(), n_process=self.max_workers)
        else:   # serial processing
            tokenizerpipe = (self.nlp(txt) for txt in docs.values())

        # tokenize each document which yields Doc object `d` for document label `lbl`
        for lbl, d in dict(zip(docs.keys(), tokenizerpipe)).items():
            _init_spacy_doc(d, lbl, additional_attrs=self._token_attrs_defaults)
            self._docs[lbl] = d

    def _update_workers_docs(self):
        """Helper method to update the worker <-> document assignments."""
        if self.max_workers > 1 and self._docs:     # parallel processing enabled
            # make assignments based on number of tokens per document
            self._workers_docs = greedy_partitioning({lbl: len(d) for lbl, d in self._docs.items()},
                                                     k=self.max_workers, return_only_labels=True)
        else:   # parallel processing disabled or no documents
            self._workers_docs = []

    def _serialize(self, deepcopy_attrs, store_nlp_instance_pointer):
        """
        Helper method to serialize this Corpus object to a dict. All SpaCy documents are serialized as
        `DocBin <https://spacy.io/api/docbin/>`_ object.

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

        # 2. spaCy data
        state_attrs['spacy_data'] = DocBin(attrs=list(set(self.STD_TOKEN_ATTRS) - {'whitespace'}),
                                           store_user_data=True,
                                           docs=self._docs.values()).to_bytes()

        if store_nlp_instance_pointer:
            state_attrs['spacy_instance'] = self.nlp
        else:
            state_attrs['spacy_instance'] = self.language_model

        return state_attrs

    @classmethod
    def _deserialize(cls, data: dict):
        """
        Helper method to deserialize a Corpus object from a dict. All SpaCy documents must be in
        ``data['spacy_data']`` as `DocBin <https://spacy.io/api/docbin/>`_ object.
        """
        # load documents using the DocBin data
        docs = DocBin().from_bytes(data['spacy_data'])

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
