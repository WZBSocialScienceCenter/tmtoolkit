import multiprocessing as mp
import string
from copy import deepcopy
from typing import Dict, Union, List, Optional, Any

import spacy
from spacy.tokens import Doc, DocBin
from loky import get_reusable_executor


from ._common import DEFAULT_LANGUAGE_MODELS, load_stopwords
from ..utils import greedy_partitioning


class Corpus:
    STD_TOKEN_ATTRS = ['whitespace', 'pos', 'lemma']

    def __init__(self, docs: Optional[Union[Dict[str, str], DocBin]] = None,
                 language: Optional[str] = None, language_model: Optional[str] = None,
                 spacy_instance: Optional[Any] = None,
                 spacy_disable: Optional[Union[list, tuple]] = ('parser', 'ner'),
                 spacy_opts: Optional[dict] = None,
                 stopwords: Optional[Union[list, tuple]] = None,
                 punctuation: Optional[Union[list, tuple]] = None,
                 max_workers: Optional[Union[int, float]] = None,
                 workers_timeout: int = 10):
        self.print_summary_default_max_tokens_string_length = 50
        self.print_summary_default_max_documents = 10

        if spacy_instance:
            self.nlp = spacy_instance
        else:
            if language is None and language_model is None:
                raise ValueError('either `language` or `language_model` must be given')

            if language_model is None:
                if not isinstance(language, str) or len(language) != 2:
                    raise ValueError('`language` must be a two-letter ISO 639-1 language code')

                if language not in DEFAULT_LANGUAGE_MODELS:
                    raise ValueError('language "%s" is not supported' % language)
                language_model = DEFAULT_LANGUAGE_MODELS[language] + '_sm'

            spacy_kwargs = dict(disable=spacy_disable)
            if spacy_opts:
                spacy_kwargs.update(spacy_opts)

            self.nlp = spacy.load(language_model, **spacy_kwargs)

        self.stopwords = stopwords or load_stopwords(self.language)
        self.punctuation = punctuation or (list(string.punctuation) + [' ', '\r', '\n', '\t'])
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

        self.max_workers = max_workers
        self.procexec = get_reusable_executor(max_workers=self.max_workers, timeout=workers_timeout) \
            if self.max_workers > 1 else None

        if docs is not None:
            if isinstance(docs, DocBin):
                self._docs = {d._.label: d for d in docs.get_docs(self.nlp.vocab)}
            else:
                self._tokenize(docs)

            self._update_workers_docs()

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'<Corpus [{self.n_docs} document{"s" if self.n_docs > 1 else ""} ({self.n_docs_masked} masked) / ' \
               f'language "{self.language}"]>'

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
        """
        if doc_label not in self.spacydocs_ignore_filter.keys():
            raise KeyError('document `%s` not found in corpus' % doc_label)
        return self.docs[doc_label]

    def __setitem__(self, doc_label: str, doc: Union[str, Doc]):
        """
        Dict method for inserting a new document or updating an existing document
        either as text or as spaCy Doc object.
        """
        from ._helpers import _init_spacy_doc

        if not isinstance(doc_label, str):
            raise KeyError('`doc_label` must be a string')

        if not isinstance(doc, (str, Doc)):
            raise ValueError('`doc_text` must be a string or spaCy Doc object')

        if isinstance(doc, str):
            doc = self.nlp(doc)

        _init_spacy_doc(doc, doc_label, additional_attrs=self._token_attrs_defaults)
        self._docs[doc_label] = doc

        self._update_workers_docs()

    def __delitem__(self, doc_label):
        """Dict method for removing a document with label `doc_label` via ``del corpus[<doc_label>]``."""
        if doc_label not in self.spacydocs_ignore_filter.keys():
            raise KeyError('document `%s` not found in corpus' % doc_label)
        del self._docs[doc_label]
        self._update_workers_docs()

    def __iter__(self):
        """Dict method for iterating through all unmasked documents."""
        return self.docs.__iter__()

    def __contains__(self, doc_label) -> bool:
        """Dict method for checking whether `doc_label` exists in this corpus."""
        return doc_label in self.spacydocs.keys()

    def __copy__(self):
        return self._deserialize(self._create_state_object(deepcopy_attrs=True))

    def __deepcopy__(self, memodict=None):
        return self.__copy__()

    def items(self):
        """Dict method to retrieve pairs of document labels and tokens of unmasked documents."""
        return self.docs.items()

    def keys(self):
        """Dict method to retrieve document labels of unmasked documents."""
        return self.spacydocs.keys()   # using "spacydocs" here is a bit faster b/c we don't call `doc_tokens`

    def values(self):
        """Dict method to retrieve document tokens."""
        return self.docs.values()

    def get(self, *args) -> List[str]:
        """
        Dict method to retrieve a specific document like ``corpus.get(<doc_label>, <default>)``.

        This method doesn't prevent you from retrieving a masked document.
        """
        return self.docs.get(*args)

    @property
    def docs_filtered(self) -> bool:
        for d in self._docs:
            if not d._.mask:
                return True
        return False

    @property
    def tokens_filtered(self) -> bool:
        return self._tokens_masked

    @property
    def is_filtered(self) -> bool:
        return self.docs_filtered or self.tokens_filtered

    @property
    def tokens_processed(self) -> bool:
        return self._tokens_processed

    @property
    def is_processed(self) -> bool:     # alias
        return self.tokens_processed

    @property
    def uses_unigrams(self) -> bool:
        return self._ngrams == 1

    @property
    def token_attrs(self) -> List[str]:
        return self.STD_TOKEN_ATTRS + list(self._token_attrs_defaults.keys())

    @property
    def custom_token_attrs_defaults(self) -> Dict[str, Any]:
        return self._token_attrs_defaults

    @property
    def doc_attrs(self) -> List[str]:
        return list(self._doc_attrs_defaults.keys())

    @property
    def doc_attrs_defaults(self) -> Dict[str, Any]:
        return self._doc_attrs_defaults

    @property
    def ngrams(self) -> int:
        return self._ngrams

    @property
    def ngrams_join_str(self) -> str:
        return self._ngrams_join_str

    @property
    def language(self) -> str:
        return self.nlp.lang

    @property
    def docs(self) -> Dict[str, List[str]]:
        from ._corpusfuncs import doc_tokens
        return doc_tokens(self._docs)

    @property
    def n_docs(self) -> int:
        return len(self)

    @property
    def n_docs_masked(self) -> int:
        return len(self.spacydocs_ignore_filter) - self.n_docs

    @property
    def ignore_doc_filter(self) -> bool:
        return self._ignore_doc_filter

    @ignore_doc_filter.setter
    def ignore_doc_filter(self, ignore: bool):
        self._ignore_doc_filter = ignore

    @property
    def spacydocs(self) -> Dict[str, Doc]:
        if self._ignore_doc_filter:
            return self.spacydocs_ignore_filter
        else:
            return {lbl: d for lbl, d in self._docs.items() if d._.mask}

    @property
    def spacydocs_ignore_filter(self) -> Dict[str, Doc]:
        return self._docs

    @spacydocs.setter
    def spacydocs(self, docs: Dict[str, Doc]):
        self._docs = docs
        self._update_workers_docs()

    @property
    def workers_docs(self) -> List[List[str]]:
        return self._workers_docs

    @property
    def max_workers(self):
        return self._n_max_workers

    @max_workers.setter
    def max_workers(self, max_workers):
        if max_workers is None:
            self._n_max_workers = 0
        else:
            if not isinstance(max_workers, (int, float)) or \
                    (isinstance(max_workers, float) and not 0 <= max_workers <= 1):
                raise ValueError('`max_workers` must be an integer, a float in [0, 1] or None')

            if isinstance(max_workers, float):
                self._n_max_workers = round(mp.cpu_count() * max_workers)
            else:
                if max_workers >= 0:
                   self._n_max_workers = max_workers if max_workers > 1 else 0
                else:
                    self._n_max_workers = mp.cpu_count() + max_workers

        self._update_workers_docs()

    def _tokenize(self, docs: Dict[str, str]):
        from ._helpers import _init_spacy_doc

        if self.max_workers > 1:
            tokenizerpipe = self.nlp.pipe(docs.values(), n_process=self.max_workers)
        else:
            tokenizerpipe = (self.nlp(d) for d in docs.values())

        for lbl, d in dict(zip(docs.keys(), tokenizerpipe)).items():
            _init_spacy_doc(d, lbl, additional_attrs=self._token_attrs_defaults)
            self._docs[lbl] = d

    def _update_workers_docs(self):
        if self.max_workers > 1 and self._docs:
            self._workers_docs = greedy_partitioning({lbl: len(d) for lbl, d in self._docs.items()},
                                                     k=self.max_workers, return_only_labels=True)
        else:
            self._workers_docs = []

    def _create_state_object(self, deepcopy_attrs):
        state_attrs = {'state': {}}
        attr_deny = {'nlp', 'procexec', 'spacydocs', 'workers_docs'}
        attr_acpt = {'_token_attrs_default', '_is_filtered', '_is_processed'}

        # 1. general object attributes
        for attr in dir(self):
            if attr not in attr_acpt and (attr.startswith('_') or attr.isupper() or attr in attr_deny):
                continue
            classattr = getattr(type(self), attr, None)
            if classattr is not None and (callable(classattr) or isinstance(classattr, property)):
                continue

            attr_obj = getattr(self, attr)
            if deepcopy_attrs:
                state_attrs['state'][attr] = deepcopy(attr_obj)
            else:
                state_attrs['state'][attr] = attr_obj

        state_attrs['language'] = self.language
        state_attrs['max_workers'] = self.max_workers
        state_attrs['workers_timeout'] = self.procexec._timeout if self.procexec else None

        # 2. spaCy data
        state_attrs['spacy_data'] = DocBin(attrs=list(set(self.STD_TOKEN_ATTRS) - {'whitespace'}),
                                           store_user_data=True,
                                           docs=self._docs.values()).to_bytes()

        return state_attrs

    @classmethod
    def _deserialize(cls, data: dict):
        docs = DocBin().from_bytes(data['spacy_data'])
        nlp = spacy.blank(data['language'])
        instance = cls(docs, spacy_instance=nlp,
                       max_workers=data['max_workers'],
                       workers_timeout=data['workers_timeout'])

        for attr, val in data['state'].items():
            setattr(instance, attr, val)

        return instance
