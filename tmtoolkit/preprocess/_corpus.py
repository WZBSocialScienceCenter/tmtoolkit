import os
import multiprocessing as mp
import string
import unicodedata
from copy import deepcopy, copy
from functools import partial, wraps
from inspect import signature
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Union, List, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import spacy
from spacy import Vocab
from spacy.tokens import Doc, DocBin
#from spacy.attrs import ORTH, SPACY, POS, LEMMA
from loky import get_reusable_executor

from ._common import DEFAULT_LANGUAGE_MODELS, LANGUAGE_LABELS, load_stopwords
from ..bow.dtm import create_sparse_dtm, dtm_to_dataframe, dtm_to_datatable
from ..utils import greedy_partitioning, merge_dicts, merge_counters, empty_chararray, as_chararray, flatten_list,\
    combine_sparse_matrices_columnwise, arr_replace, pickle_data, unpickle_file
from .._pd_dt_compat import USE_DT, FRAME_TYPE, pd_dt_frame, pd_dt_concat, pd_dt_sort, pd_dt_colnames, \
    pd_dt_frame_to_list

# Meta data on document level is stored as Doc extension.
# Custom meta data on token level is however *not* stored as Token extension, since this approach proved to be very
# slow. It is instead stored in the `user_data` dict of each Doc instance.
Doc.set_extension('label', default='', force=True)


class Corpus:
    STD_TOKEN_ATTRS = ['whitespace', 'pos', 'lemma']

    def __init__(self, docs: Optional[Union[Dict[str, str], DocBin]] = None,
                 language: Optional[str] = None, language_model: Optional[str] = None,
                 spacy_instance: Optional[object] = None,
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
        self._n_max_workers = 0
        self._docs = {}
        self._token_attrs = {}
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
        return f'<Corpus [{self.n_docs} document{"s" if self.n_docs > 1 else ""} / language "{self.language}"]>'

    def __len__(self) -> int:
        """
        Dict method to return number of documents.

        :return: number of documents
        """
        return len(self._docs)

    def __getitem__(self, doc_label) -> List[str]:
        """
        dict method for retrieving document with label `doc_label` via ``corpus[<doc_label>]``.
        """
        if doc_label not in self.docs:
            raise KeyError('document `%s` not found in corpus' % doc_label)
        return self.docs[doc_label]

    def __setitem__(self, doc_label: str, doc: Union[str, Doc]):
        """
        dict method for inserting a new document or updating an existing document
        either as text or as spaCy Doc object.
        """
        if not isinstance(doc_label, str):
            raise KeyError('`doc_label` must be a string')

        if not isinstance(doc, (str, Doc)):
            raise ValueError('`doc_text` must be a string or spaCy Doc object')

        if isinstance(doc, str):
            doc = self.nlp(doc)

        _init_spacy_doc(doc, doc_label, additional_attrs=self._token_attrs)
        self._docs[doc_label] = doc

        self._update_workers_docs()

    def __delitem__(self, doc_label):
        """dict method for removing a document with label `doc_label` via ``del corpus[<doc_label>]``."""
        if doc_label not in self.docs:
            raise KeyError('document `%s` not found in corpus' % doc_label)
        del self.docs[doc_label]

    def __iter__(self):
        """dict method for iterating through the document labels."""
        return self.docs.__iter__()

    def __contains__(self, doc_label) -> bool:
        """dict method for checking whether `doc_label` exists in this corpus."""
        return doc_label in self.keys()

    def __copy__(self):
        return self._deserialize(self._create_state_object(deepcopy_attrs=True))

    def __deepcopy__(self, memodict=None):
        return self.__copy__()

    def items(self):
        """dict method to retrieve pairs of document labels and texts."""
        return self.docs.items()

    def keys(self):
        """dict method to retrieve document labels."""
        return self.docs.keys()

    def values(self):
        """dict method to retrieve document texts."""
        return self.docs.values()

    def get(self, *args) -> List[str]:
        """dict method to retrieve a specific document like ``corpus.get(<doc_label>, <default>)``."""
        return self.docs.get(*args)

    @property
    def language(self) -> str:
        return self.nlp.lang

    @property
    def docs(self) -> Dict[str, List[str]]:
        return doc_tokens(self._docs)

    @property
    def n_docs(self) -> int:
        return len(self)

    @property
    def spacydocs(self) -> Dict[str, Doc]:
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
            self._n_max_workers = mp.cpu_count()
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
        if self.max_workers > 1:
            tokenizerpipe = self.nlp.pipe(docs.values(), n_process=self.max_workers)
        else:
            tokenizerpipe = (self.nlp(d) for d in docs.values())

        for lbl, d in dict(zip(docs.keys(), tokenizerpipe)).items():
            _init_spacy_doc(d, lbl, additional_attrs=self._token_attrs)
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
        attr_acpt = {'_token_attrs'}

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


#%% parallel execution helpers and other decorators

merge_dicts_sorted = partial(merge_dicts, sort_keys=True)

@dataclass
class ParallelTask:
    procexec: object
    workers_assignments: list
    data: dict


def _paralleltask(corpus: Corpus, tokens=None):
    return ParallelTask(corpus.procexec, corpus.workers_docs,
                        doc_tokens(corpus) if tokens is None else tokens)


def parallelexec(collect_fn):
    def deco_fn(fn):
        @wraps(fn)
        def inner_fn(docs_or_task, *args, **kwargs):
            if isinstance(docs_or_task, ParallelTask) and docs_or_task.procexec:
                print(f'{os.getpid()}: distributing function {fn} for {len(docs_or_task.data)} items to '
                      f'{len(docs_or_task.workers_assignments)} workers')
                if args:
                    fn_argnames = list(signature(fn).parameters.keys())
                    # first argument in `fn` is always the documents dict -> we skip this
                    if len(fn_argnames) <= len(args):
                        raise ValueError(f'function {fn} does not accept enough additional arguments')
                    kwargs.update({fn_argnames[i+1]: v for i, v in enumerate(args)})
                workers_data = [{lbl: docs_or_task.data[lbl] for lbl in itemlabels}
                                for itemlabels in docs_or_task.workers_assignments]
                res = docs_or_task.procexec.map(partial(fn, **kwargs), workers_data)
                if collect_fn:
                    return collect_fn(res)
                else:
                    return None
            else:
                print(f'{os.getpid()}: directly applying function {fn} to {len(docs_or_task.data)} items')
                return fn(docs_or_task.data, *args, **kwargs)

        return inner_fn

    return deco_fn


def corpus_func_copiable(fn):
    """
    Decorator for a Corpus function `fn` with an optional argument `inplace`. This decorator makes sure that if
    `fn` is called with `inplace=False`, the passed corpus will be copied before `fn` is applied to it. Then,
    the modified copy of corpus is returned. If `inplace=True`, `fn` is applied as usual.

    :param fn: Corpus function `fn` with an optional argument `inplace`
    :return: wrapper function of `fn`
    """
    @wraps(fn)
    def inner_fn(*args, **kwargs):
        assert isinstance(args[0], Corpus), 'first argument must be a Corpus object'

        if 'inplace' in kwargs:
            inplace = kwargs.pop('inplace')
        else:
            inplace = True

        # get Corpus object `corp`, optionally copy it
        if inplace:
            corp = args[0]
        else:
            corp = copy(args[0])  # makes a deepcopy

        # apply fn to `corp`, passing all other arguments
        fn(corp, *args[1:], **kwargs)
        return corp

    return inner_fn


#%% Corpus functions with readonly access to Corpus data


def doc_tokens(docs: Union[Corpus, Dict[str, Doc]],
               only_non_empty=False,
               tokens_as_hashes=False,
               with_metadata: Union[bool, list, tuple] = False,
               as_datatables=False, as_arrays=False, apply_filter=True) \
        -> Dict[str, Union[List[str], dict, FRAME_TYPE]]:
    # requesting the token mask disables the token filtering
    with_mask = isinstance(with_metadata, (list, tuple)) and 'mask' in with_metadata
    if with_mask:
        apply_filter = False

    if isinstance(docs, Corpus):
        docs = docs.spacydocs

    res = {}
    for lbl, d in docs.items():
        if only_non_empty and len(d) == 0:
            continue

        tok = _filtered_doc_tokens(d, tokens_as_hashes=tokens_as_hashes, apply_filter=apply_filter)

        if with_metadata is not False:
            resdoc = {'token': tok}

            std_attrs = Corpus.STD_TOKEN_ATTRS
            if isinstance(with_metadata, (list, tuple)):
                std_attrs = [k for k in std_attrs if k in with_metadata]

            for k in std_attrs:
                v = _filtered_doc_attr(d, k, apply_filter=apply_filter)
                if k == 'whitespace':
                    v = list(map(lambda ws: ws == ' ', v))
                resdoc[k] = v

            user_attrs = d.user_data.keys()
            if isinstance(with_metadata, (list, tuple)):
                user_attrs = [k for k in user_attrs if k in with_metadata]

            for k in user_attrs:
                if isinstance(k, str) and (k.startswith('meta_') or (with_mask and k == 'mask')):
                    resdoc[k] = _filtered_doc_attr(d, k, custom=True, apply_filter=apply_filter)
                    if not as_datatables and not as_arrays:
                        resdoc[k] = list(resdoc[k])
            res[lbl] = resdoc
        else:
            res[lbl] = tok

    if as_datatables:
        res = dict(zip(res.keys(), map(pd_dt_frame, res.values())))
    elif as_arrays:
        if with_metadata:
            res = dict(zip(res.keys(),
                           [dict(zip(d.keys(), map(np.array, d.values()))) for d in res.values()]))
        else:
            res = dict(zip(res.keys(), map(as_chararray, res.values())))

    return res


def doc_lengths(docs: Corpus) -> Dict[str, int]:
    """
    Return document length (number of tokens in doc.) for each document.

    :param docs: a Corpus object
    :return: dict of document lengths per document label
    """
    return {dl: np.sum(d.user_data['mask']) for dl, d in docs.spacydocs.items()}


def doc_token_lengths(docs: Corpus) -> Dict[str, List[int]]:
    """
    Return token lengths (number of characters of each token) for each document.

    :param docs: a Corpus object
    :return: dict with list of token lengths per document label
    """
    return {dl: list(map(len, tok)) for dl, tok in doc_tokens(docs).items()}


def n_tokens(docs: Corpus) -> int:
    return sum(doc_lengths(docs).values())


def n_chars(docs: Corpus) -> int:
    return sum(sum(n) for n in doc_token_lengths(docs).values())


def doc_labels(docs: Corpus) -> List[str]:
    return sorted(docs.keys())


def doc_texts(docs: Corpus, collapse: Optional[str] = None) -> Dict[str, str]:
    @parallelexec(collect_fn=merge_dicts_sorted)
    def _doc_texts(tokens):
        texts = {}
        for dl, dtok in tokens.items():
            if collapse is None:
                texts[dl] = ''
                for t, ws in zip(dtok['token'], dtok['whitespace']):
                    texts[dl] += t
                    if ws:
                        texts[dl] += ' '
            else:
                texts[dl] = collapse.join(dtok['token'])

        return texts

    return _doc_texts(_paralleltask(docs, doc_tokens(docs, with_metadata=True)))


def doc_frequencies(docs: Corpus, proportions=False) -> Dict[str, Union[int, float]]:
    """
    Document frequency per vocabulary token as dict with token to document frequency mapping.
    Document frequency is the measure of how often a token occurs *at least once* in a document.
    Example with absolute document frequencies:

    .. code-block:: text

        doc tokens
        --- ------
        A   z, z, w, x
        B   y, z, y
        C   z, z, y, z

        document frequency df(z) = 3  (occurs in all 3 documents)
        df(x) = df(w) = 1 (occurs only in A)
        df(y) = 2 (occurs in B and C)
        ...

    :param docs: list of string tokens or spaCy documents
    :param proportions: if True, normalize by number of documents to obtain proportions
    :return: dict mapping token to document frequency
    """
    @parallelexec(collect_fn=merge_counters)
    def _doc_frequencies(tokens, norm):
        doc_freqs = Counter()

        for dtok in tokens.values():
            for t in set(dtok):
                doc_freqs[t] += 1

        if norm != 1:
            doc_freqs = Counter({t: n/norm for t, n in doc_freqs.items()})

        return doc_freqs

    # still not sure if the version that uses hashes is faster:
    # res = _doc_frequencies(_paralleltask(docs, doc_tokens(docs, tokens_as_hashes=True)),
    #                        norm=len(docs) if proportions else 1)
    # return dict(zip(map(lambda h: docs.nlp.vocab.strings[h], res.keys()), res.values()))

    return _doc_frequencies(_paralleltask(docs), norm=len(docs) if proportions else 1)


def doc_vectors(docs: Corpus) -> Dict[str, np.ndarray]:    # TODO: from masked/processed documents?
    return {dl: d.vector for dl, d in docs.spacydocs.items()}


def token_vectors(docs: Corpus) -> Dict[str, np.ndarray]:  # TODO: from masked/processed documents? Generate lexemes?
    # uses spaCy documents -> would require to distribute documents via DocBin
    # (https://spacy.io/api/docbin) to parallelize
    return {dl: np.vstack([t.vector for t in d]) if len(d) > 0 else np.array([])
            for dl, d in docs.spacydocs.items()}


def vocabulary(docs: Union[Corpus, Dict[str, List[str]]], as_hashes=False, sort=False) -> Union[set, list]:
    if isinstance(docs, Corpus):
        tok = doc_tokens(docs, tokens_as_hashes=as_hashes).values()
    else:
        tok = docs.values()

    v = set(flatten_list(tok))

    if sort:
        return sorted(v)
    else:
        return v


def vocabulary_counts(docs: Corpus) -> Counter:
    """
    Return :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
    all documents.

    :param docs: list of string tokens or spaCy documents
    :return: :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
             all documents
    """
    @parallelexec(collect_fn=merge_counters)
    def _vocabulary_counts(tokens):
        return Counter(flatten_list(tokens.values()))

    return _vocabulary_counts(_paralleltask(docs))


def vocabulary_size(docs: Corpus) -> int:
    return len(vocabulary(docs))


def tokens_with_metadata(docs: Corpus) -> Dict[str, FRAME_TYPE]:
    return doc_tokens(docs, with_metadata=True, as_datatables=True)


def tokens_datatable(docs: Corpus, with_metadata: Union[bool, list, tuple, set] = True) -> FRAME_TYPE:
    @parallelexec(collect_fn=list)
    def _tokens_datatable(tokens):
        dfs = []
        for dl, df in tokens.items():
            n = df.shape[0]
            meta_df = pd_dt_frame({
                'doc': np.repeat(dl, n),
                'position': np.arange(n)
            })

            dfs.append(pd_dt_concat((meta_df, df), axis=1))
        return dfs

    tokens = doc_tokens(docs, only_non_empty=False, with_metadata=with_metadata or [], as_datatables=True)
    dfs = _tokens_datatable(_paralleltask(docs, tokens))

    if dfs:
        res = pd_dt_concat(dfs)
    else:
        res = pd_dt_frame({'doc': [], 'position': [], 'token': []})

    return pd_dt_sort(res, ['doc', 'position'])


def tokens_dataframe(docs: Corpus, with_metadata: Union[bool, list, tuple, set] = True) -> pd.DataFrame:
    # note that generating a datatable first and converting it to pandas is faster than generating a pandas data
    # frame right away

    df = tokens_datatable(docs, with_metadata=with_metadata)

    if USE_DT:
        df = df.to_pandas()

    return df.set_index(['doc', 'position'])


def tokens_with_pos_tags(docs: Corpus) -> Dict[str, FRAME_TYPE]:
    """
    Document tokens with POS tag as dict with mapping document label to datatable. The datatables have two
    columns, ``token`` and ``pos``.
    """
    return {dl: df[:, ['token', 'pos']] if USE_DT else df.loc[:, ['token', 'pos']]
            for dl, df in doc_tokens(docs, with_metadata=True, as_datatables=True).items()}


def corpus_summary(docs, max_documents=None, max_tokens_string_length=None) -> str:
    """
    Print a summary of this object, i.e. the first tokens of each document and some summary statistics.

    :param max_documents: maximum number of documents to print; `None` uses default value 10; set to -1 to
                          print *all* documents
    :param max_tokens_string_length: maximum string length of concatenated tokens for each document; `None` uses
                                     default value 50; set to -1 to print complete documents
    """

    if max_tokens_string_length is None:
        max_tokens_string_length = docs.print_summary_default_max_tokens_string_length
    if max_documents is None:
        max_documents = docs.print_summary_default_max_documents

    summary = f'Corpus with {len(docs)} document' \
              f'{"s" if len(docs) > 1 else ""} in {LANGUAGE_LABELS[docs.language].capitalize()}'

    texts = doc_texts(docs)
    dlengths = doc_lengths(docs)

    for dl, tokstr in texts.items():
        if max_tokens_string_length >= 0 and len(tokstr) > max_tokens_string_length:
            tokstr = tokstr[:max_tokens_string_length] + '...'

        summary += f'\n> {dl} ({dlengths[dl]} tokens): {tokstr}'

    if len(docs) > max_documents:
        summary += f'\n(and {len(docs) - max_documents} more documents)'

    summary += f'\ntotal number of tokens: {n_tokens(docs)} / vocabulary size: {vocabulary_size(docs)}'

    return summary


def print_summary(docs: Corpus, max_documents=None, max_tokens_string_length=None):
    print(corpus_summary(docs, max_documents=max_documents, max_tokens_string_length=max_tokens_string_length))


def dtm(docs: Corpus, as_datatable=False, as_dataframe=False, dtype=None)\
        -> Union[csr_matrix, FRAME_TYPE]:
    @parallelexec(collect_fn=list)
    def _sparse_dtms(docs):
        vocab = vocabulary(docs, sort=True)
        alloc_size = sum(len(set(dtok)) for dtok in docs.values())  # sum of *unique* tokens in each document

        return (create_sparse_dtm(vocab, docs.values(), alloc_size, vocab_is_sorted=True),
                docs.keys(),
                vocab)

    if len(docs) > 0:
        w_dtms, w_doc_labels, w_vocab = zip(*_sparse_dtms(_paralleltask(docs)))
        dtm, vocab, dtm_doc_labels = combine_sparse_matrices_columnwise(w_dtms, w_vocab, w_doc_labels,
                                                                        dtype=dtype)
        # sort according to document labels
        dtm = dtm[np.argsort(dtm_doc_labels), :]
        doc_labels = np.sort(dtm_doc_labels)
    else:
        dtm = csr_matrix((0, 0), dtype=dtype or int)   # empty sparse matrix
        vocab = empty_chararray()
        doc_labels = empty_chararray()

    if as_datatable:
        return dtm_to_datatable(dtm, doc_labels, vocab)
    elif as_dataframe:
        return dtm_to_dataframe(dtm, doc_labels, vocab)
    else:
        return dtm


def ngrams(docs: Corpus, n: int, join=True, join_str=' ') -> List[Union[List[str], str]]:
    """
    Generate and return n-grams of length `n`.

    :param docs: list of string tokens or spaCy documents
    :param n: length of n-grams, must be >= 2
    :param join: if True, join generated n-grams by string `join_str`
    :param join_str: string used for joining
    :return: list of n-grams; if `join` is True, the list contains strings of joined n-grams, otherwise the list
             contains lists of size `n` in turn containing the strings that make up the n-gram
    """
    @parallelexec(collect_fn=merge_dicts_sorted)
    def _ngrams(tokens, n, join, join_str):
        res = {}
        for dl, dt in tokens.items():
            if len(dt) == 0:
                ng = []
            else:
                if len(dt) < n:
                    ng = [dt]
                else:
                    ng = [[dt[i + j] for j in range(n)]
                          for i in range(len(dt) - n + 1)]

            if join:
                res[dl] = list(map(lambda x: join_str.join(x), ng))
            else:
                res[dl] = ng
        return res

    if n < 2:
        raise ValueError('`n` must be at least 2')

    return _ngrams(_paralleltask(docs), n, join, join_str)


#%% Corpus I/O


def save_corpus_to_picklefile(docs: Corpus, picklefile: str):
    pickle_data(serialize_corpus(docs, deepcopy_attrs=False), picklefile)


def load_corpus_from_picklefile(picklefile: str) -> Corpus:
    return deserialize_corpus(unpickle_file(picklefile))


def load_corpus_from_tokens(tokens: Dict[str, Union[list, tuple, Dict[str, List]]], **corpus_kwargs):
    if 'docs' in corpus_kwargs:
        raise ValueError('`docs` parameter is obsolete when initializing a Corpus with this function')
    corp = Corpus(**corpus_kwargs)

    # create SpaCy docs from tokens (with metadata)
    spacydocs = {}
    for label, tok in tokens.items():
        if isinstance(tok, (list, tuple)):                          # tokens alone (no metadata)
            doc = spacydoc_from_tokens(tok, label=label, vocab=corp.nlp.vocab)
        else:
            if isinstance(tok, FRAME_TYPE):  # each document is a datatable
                tok = {col: coldata for col, coldata in zip(pd_dt_colnames(tok), pd_dt_frame_to_list(tok))}
            elif not isinstance(tok, dict):
                raise ValueError(f'data for document `{label}` is of unknown type `{type(tok)}`')

            doc = spacydoc_from_tokens_with_metadata(tok, label=label, vocab=corp.nlp.vocab)

        spacydocs[label] = doc

    corp.spacydocs = spacydocs

    return corp


def load_corpus_from_tokens_datatable(tokens: FRAME_TYPE, **corpus_kwargs):
    if not USE_DT:
        raise RuntimeError('this function requires the package "datatable" to be installed')

    if 'docs' in corpus_kwargs:
        raise ValueError('`docs` parameter is obsolete when initializing a Corpus with this function')

    if {'doc', 'position', 'token'} & set(pd_dt_colnames(tokens)) != {'doc', 'position', 'token'}:
        raise ValueError('`tokens` must at least contain a columns "doc", "position" and "token"')

    import datatable as dt

    tokens_dict = {}
    for dl in dt.unique(tokens[:, dt.f.doc]).to_list()[0]:
        doc_df = tokens[dt.f.doc == dl, :]
        colnames = pd_dt_colnames(doc_df)
        colnames.pop(colnames.index('doc'))
        colnames.pop(colnames.index('position'))
        tokens_dict[dl] = doc_df[:, colnames]

    return load_corpus_from_tokens(tokens_dict, **corpus_kwargs)


def serialize_corpus(docs: Corpus, deepcopy_attrs=True):
    return docs._create_state_object(deepcopy_attrs=deepcopy_attrs)


def deserialize_corpus(serialized_corpus_data: dict):
    return Corpus._deserialize(serialized_corpus_data)


#%% Corpus functions that modify corpus data


@corpus_func_copiable
def add_metadata_per_token(docs: Corpus, /, key: str, data: dict, default=None, inplace=True):
    # convert data token string keys to token hashes
    data = {docs.nlp.vocab.strings[k]: v for k, v in data.items()}

    key = 'meta_' + key

    for d in docs.spacydocs.values():
        d.user_data[key] = np.array([data.get(hash, default) if mask else default
                                     for mask, hash in zip(d.user_data['mask'], d.user_data['processed'])])

    docs._token_attrs[key] = default


@corpus_func_copiable
def transform_tokens(docs: Corpus, /, func: Callable, inplace=True, **kwargs):
    vocab = vocabulary(docs, as_hashes=True)
    stringstore = docs.nlp.vocab.strings

    replace_from = []
    replace_to = []
    for t_hash in vocab:
        t_transformed = func(stringstore[t_hash], **kwargs)
        t_hash_transformed = stringstore[t_transformed]
        if t_hash != t_hash_transformed :
            stringstore.add(t_transformed)
            replace_from.append(t_hash)
            replace_to.append(t_hash_transformed)

    if replace_from:
        for d in docs.spacydocs.values():
            if d.user_data['processed'].flags.writeable:
                arr_replace(d.user_data['processed'], replace_from, replace_to, inplace=True)
            else:
                d.user_data['processed'] = arr_replace(d.user_data['processed'], replace_from, replace_to)


def to_lowercase(docs: Corpus, /, inplace=True):
    return transform_tokens(docs, str.lower, inplace=inplace)


def to_uppercase(docs: Corpus, /, inplace=True):
    return transform_tokens(docs, str.upper, inplace=inplace)


def remove_chars(docs: Corpus, /, chars: Iterable, inplace=True):
    del_chars = str.maketrans('', '', ''.join(chars))
    return transform_tokens(docs, lambda t: t.translate(del_chars), inplace=inplace)


def remove_punctuation(docs: Corpus, /, inplace=True):
    return remove_chars(docs, docs.punctuation, inplace=inplace)


def normalize_unicode(docs: Corpus, /, form: str = 'NFC', inplace=True):
    """
    Normalize unicode characters according to `form`.

    This function only *normalizes* unicode characters in the tokens of `docs` to the form
    specified by `form`. If you want to *simplify* the characters, i.e. remove diacritics,
    underlines and other marks, use :func:`simplify_unicode` instead.

    :param docs: a Corpus object
    :param form: normal form (see https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either original Corpus object `docs` or a modified copy of it
    """
    return transform_tokens(docs, lambda t: unicodedata.normalize(form, t), inplace=inplace)


def simplify_unicode(docs: Corpus, /, method: str = 'icu', inplace=True):
    """
    *Simplify* unicode characters in the tokens of `docs`, i.e. remove diacritics, underlines and
    other marks. Requires `PyICU <https://pypi.org/project/PyICU/>`_ to be installed when using
    `method="icu"`.

    :param docs: a Corpus object
    :param method: either `"icu"` which uses `PyICU <https://pypi.org/project/PyICU/>`_ for "proper"
                   simplification or "ascii" which tries to encode the characters as ASCII; the latter
                   is not recommended and will simply dismiss any characters that cannot be converted
                   to ASCII after decomposition
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either original Corpus object `docs` or a modified copy of it
    """

    method = method.lower()
    if method == 'icu':
        from icu import UnicodeString, Transliterator, UTransDirection

        def fn(t: str):
            u = UnicodeString(t)
            trans = Transliterator.createInstance("NFD; [:M:] Remove; NFC", UTransDirection.FORWARD)
            trans.transliterate(u)
            return str(u)

    elif method == 'ascii':
        def fn(t: str):
            return unicodedata.normalize('NFKD', t).encode('ASCII', 'ignore').decode('utf-8')
    else:
        raise ValueError('`method` must be either "icu" or "ascii"')

    return transform_tokens(docs, fn, inplace=inplace)


@corpus_func_copiable
def lemmatize(docs: Corpus, /, inplace=True):
    for d in docs.spacydocs.values():
        d.user_data['processed'] = np.fromiter((t.lemma for t in d), dtype='uint64', count=len(d))


@corpus_func_copiable
def filter_clean_tokens(docs: Corpus, /,
                        remove_punct: bool = True,
                        remove_stopwords: Union[bool, list, tuple, set] = True,
                        remove_empty: bool = True,
                        remove_shorter_than: Optional[int] = None,
                        remove_longer_than: Optional[int] = None,
                        remove_numbers: bool = False,
                        inplace=True):
    if remove_shorter_than is not None and remove_shorter_than < 0:
        raise ValueError('`remove_shorter_than` must be >= 0')
    if remove_longer_than is not None and remove_longer_than < 0:
        raise ValueError('`remove_longer_than` must be >= 0')

    tokens_to_remove = [''] if remove_empty else []

    if remove_stopwords is True:
        tokens_to_remove.extend(docs.stopwords)
    elif isinstance(remove_stopwords, (list, tuple, set)):
        tokens_to_remove.extend(remove_stopwords)

    if remove_punct is True:
        tokens_to_remove.extend(docs.punctuation)
    elif isinstance(remove_punct, (list, tuple, set)):
        tokens_to_remove.extend(remove_punct)

    # the "doc masks" list holds a binary array for each document where
    # `True` signals a token to be kept, `False` a token to be removed
    doc_masks = [np.repeat(True, n) for n in doc_lengths(docs).values()]

    # update remove mask for punctuation
    if remove_punct is True:
        doc_masks = [mask & ~doc.to_array('is_punct')[doc.user_data['mask']].astype(bool)
                     for mask, doc in zip(doc_masks, docs.spacydocs.values())]

    # update remove mask for tokens shorter/longer than a certain number of characters
    if remove_shorter_than is not None or remove_longer_than is not None:
        token_lengths = map(np.array, doc_token_lengths(docs).values())

        if remove_shorter_than is not None:
            doc_masks = [mask & (n >= remove_shorter_than) for mask, n in zip(doc_masks, token_lengths)]

        if remove_longer_than is not None:
            doc_masks = [mask & (n <= remove_longer_than) for mask, n in zip(doc_masks, token_lengths)]

    # update remove mask for numeric tokens
    if remove_numbers:
        doc_masks = [mask & ~doc.to_array('like_num')[doc.user_data['mask']].astype(bool)
                     for mask, doc in zip(doc_masks, docs.spacydocs.values())]

    # update remove mask for general list of tokens to be removed
    if tokens_to_remove:
        tokens_to_remove = set(tokens_to_remove)
        # this is actually much faster than using np.isin:
        doc_masks = [mask & np.array([t not in tokens_to_remove for t in doc], dtype=bool)
                     for mask, doc in zip(doc_masks, doc_tokens(docs).values())]

    # apply the mask
    _apply_matches_array(docs, doc_masks)


#%% common helper functions


def spacydoc_from_tokens_with_metadata(tokens_w_meta: Dict[str, List], label: str,
                                       vocab: Optional[Union[Vocab, List[str]]] = None):
    otherattrs = {}
    if 'pos' in tokens_w_meta:
        otherattrs['pos'] = tokens_w_meta['pos']
    if 'lemma' in tokens_w_meta:
        otherattrs['lemmas'] = tokens_w_meta['lemma']

    userdata = {k: v for k, v in tokens_w_meta.items() if k.startswith('meta_')}

    return spacydoc_from_tokens(tokens_w_meta['token'], label=label, vocab=vocab,  spaces=tokens_w_meta['whitespace'],
                                otherattrs=otherattrs, userdata=userdata)


def spacydoc_from_tokens(tokens: List[str], label: str,
                         vocab: Optional[Union[Vocab, List[str]]] = None,
                         spaces: Optional[List[bool]] = None,
                         mask: Optional[np.ndarray] = None,
                         otherattrs: Optional[Dict[str, List[str]]] = None,
                         userdata: Optional[Dict[str, np.ndarray]] = None):
    """
    Create a new spaCy ``Doc`` document with tokens `tokens`.
    """

    # spaCy doesn't accept empty tokens
    nonempty_tok = np.array([len(t) > 0 for t in tokens])
    has_nonempty = np.sum(nonempty_tok) < len(tokens)

    if has_nonempty:
        tokens = np.asarray(tokens)[nonempty_tok].tolist()

    if vocab is None:
        vocab = Vocab(strings=set(tokens))
    elif not isinstance(vocab, Vocab):
        vocab = Vocab(strings=vocab)

    if spaces is not None:
        if has_nonempty:
            spaces = np.asarray(spaces)[nonempty_tok].tolist()
        assert len(spaces) == len(tokens), '`tokens` and `spaces` must have same length'

    if mask is not None:
        if has_nonempty:
            mask = mask[nonempty_tok]
        assert len(mask) == len(tokens), '`tokens` and `mask` must have same length'

    for attrs in (otherattrs, userdata):
        if attrs is not None:
            if has_nonempty:
                for k in attrs.keys():
                    if isinstance(attrs[k], np.ndarray):
                        attrs[k] = attrs[k][nonempty_tok]
                    else:
                        attrs[k] = np.asarray(attrs[k])[nonempty_tok].tolist()

            which = 'otherattrs' if attrs == otherattrs else 'userdata'
            for k, v in attrs.items():
                assert len(v) == len(tokens), f'all attributes in `{which}` must have the same length as `tokens`; ' \
                                              f'this failed for attribute {k}'

    new_doc = Doc(vocab, words=tokens, spaces=spaces, **(otherattrs or {}))
    assert len(new_doc) == len(tokens), 'created Doc object must have same length as `tokens`'

    _init_spacy_doc(new_doc, label, mask=mask, additional_attrs=userdata)

    return new_doc


def _init_spacy_doc(doc: Doc, doc_label: str,
                    mask: Optional[np.ndarray] = None,
                    additional_attrs: Optional[Dict[str, Union[list, tuple, np.ndarray, int, float, str]]] = None):
    n = len(doc)
    doc._.label = doc_label
    if mask is None:
        doc.user_data['mask'] = np.repeat(True, n)
    else:
        doc.user_data['mask'] = mask

    doc.user_data['processed'] = np.fromiter((t.orth for t in doc), dtype='uint64', count=n)

    if additional_attrs:
        for k, default in additional_attrs.items():
            # default can be sequence (list, tuple or array) ...
            if isinstance(default, (list, tuple)):
                v = np.array(default)
            elif isinstance(default, np.ndarray):
                v = default
            else:  # ... or single value -> repeat this value to fill the array
                v = np.repeat(default, n)
            assert len(v) == n, 'user data array must have the same length as the tokens'
            doc.user_data[k] = v


def _filtered_doc_tokens(doc: Doc, tokens_as_hashes=False, apply_filter=True):
    hashes = doc.user_data['processed']

    if apply_filter:
        hashes = hashes[doc.user_data['mask']]

    if tokens_as_hashes:
        return hashes
    else:
        return list(map(lambda hash: doc.vocab.strings[hash], hashes))


def _filtered_doc_attr(doc: Doc, attr: str, custom=False, stringified=True, apply_filter=True):
    if custom:
        res = doc.user_data[attr]
        if apply_filter:
            return res[doc.user_data['mask']]
        else:
            return res
    else:
        if stringified:
            attr += '_'
        if apply_filter:
            return [getattr(t, attr) for t, m in zip(doc, doc.user_data['mask']) if m]
        else:
            return [getattr(t, attr) for t in doc]


def _apply_matches_array(docs: Corpus, matches: List[np.ndarray] = None, invert=False):
    if invert:
        matches = [~m for m in matches]

    assert len(matches) == len(docs), '`matches` and `docs` must have same length'

    # simply set the new filter mask to previously unfiltered elements; changes document masks in-place
    for mask, doc in zip(matches, docs.spacydocs.values()):
        assert len(mask) == sum(doc.user_data['mask'])
        doc.user_data['mask'][doc.user_data['mask']] = mask
