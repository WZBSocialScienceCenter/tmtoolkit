import os
import multiprocessing as mp
from functools import partial, wraps
from inspect import signature
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Union, List, Callable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import spacy
from spacy.tokens import Doc, Token, DocBin
#from spacy.attrs import ORTH, SPACY, POS, LEMMA
from loky import get_reusable_executor

from ..bow.dtm import create_sparse_dtm, dtm_to_dataframe, dtm_to_datatable
from ..utils import greedy_partitioning, merge_dicts, merge_counters, empty_chararray, as_chararray, flatten_list,\
    combine_sparse_matrices_columnwise, arr_replace
from .._pd_dt_compat import USE_DT, FRAME_TYPE, pd_dt_frame, pd_dt_concat, pd_dt_sort

from ._common import DEFAULT_LANGUAGE_MODELS, LANGUAGE_LABELS


Doc.set_extension('label', default='', force=True)


class Corpus:
    STD_TOKEN_ATTRS = ['whitespace', 'pos', 'lemma']

    def __init__(self, docs: Dict[str, str],
                 language: Optional[str] = None, language_model: Optional[str] = None,
                 spacy_instance=None, spacy_disable=('parser', 'ner'), spacy_opts=None,
                 max_workers: Union[int, float, None] = None,
                 workers_timeout=10):
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

        if max_workers is None:
            self.n_max_workers = mp.cpu_count()
        else:
            if not isinstance(max_workers, (int, float)) or \
                    (isinstance(max_workers, float) and not 0 <= max_workers <= 1):
                raise ValueError('`n_max_workers` must be an integer, a float in [0, 1] or None')

            if isinstance(max_workers, float):
                self.n_max_workers = round(mp.cpu_count() * max_workers)
            else:
                if max_workers > 0:
                   self.n_max_workers = max_workers
                else:
                    self.n_max_workers = mp.cpu_count() + max_workers

        self.procexec = get_reusable_executor(max_workers=self.n_max_workers, timeout=workers_timeout) \
            if self.n_max_workers > 1 else None

        self._docs = {}
        self._tokenize(docs)

        self.workers_docs = greedy_partitioning({lbl: len(d) for lbl, d in self._docs.items()},
                                                k=self.n_max_workers, return_only_labels=True)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'<Corpus [{self.n_docs} document{"s" if self.n_docs > 1 else ""}]>'

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

    # def __setitem__(self, doc_label: str, doc_text: Union[str, Doc]):
    #     """dict method for setting a document with label `doc_label` via ``corpus[<doc_label>] = <doc_text>``."""
    #     if not isinstance(doc_label, str):
    #         raise KeyError('`doc_label` must be a string')
    #
    #     if not isinstance(doc_text, str):
    #         raise ValueError('`doc_text` must be a string')
    #
    #     self.docs[doc_label] = doc_text

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
        return doc_label in self.docs

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

    def _tokenize(self, docs: Dict[str, str]):
        tokenizerpipe = self.nlp.pipe(docs.values(), n_process=self.n_max_workers)

        for lbl, d in dict(zip(docs.keys(), tokenizerpipe)).items():
            d._.label = lbl
            d.user_data['mask'] = np.repeat(True, len(d))
            d.user_data['processed'] = np.fromiter((t.orth for t in d), dtype='uint64', count=len(d))
            self._docs[lbl] = d


#%% decorators

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
                print(f'{os.getpid()}: distributing function {fn} for {len(docs_or_task.data)} items')
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
                print(f'{os.getpid()}: directly applying function {fn} to {len(docs_or_task)} items')
                return fn(docs_or_task, *args, **kwargs)

        return inner_fn

    return deco_fn


#%%


def add_metadata_per_token(docs: Corpus, key: str, data: dict, default=None):
    # convert data token string keys to token hashes
    data = {docs.nlp.vocab.strings[k]: v for k, v in data.items()}

    key = 'meta_' + key

    for d in docs.spacydocs.values():
        d.user_data[key] = np.array([data.get(hash, default) if mask else default
                                     for mask, hash in zip(d.user_data['mask'], d.user_data['processed'])])


def doc_tokens(docs: Union[Corpus, Dict[str, Doc]],
               only_non_empty=False, tokens_as_hashes=False,
               with_metadata=False, as_datatables=False, as_arrays=False) \
        -> Dict[str, Union[List[str], dict, FRAME_TYPE]]:
    if isinstance(docs, Corpus):
        docs = docs.spacydocs

    res = {}
    for lbl, d in docs.items():
        if only_non_empty and len(d) == 0:
            continue

        tok = _filtered_doc_tokens(d, tokens_as_hashes=tokens_as_hashes)

        if with_metadata:
            resdoc = {'token': tok}
            for k in Corpus.STD_TOKEN_ATTRS:
                v = _filtered_doc_attr(d, k)
                if k == 'whitespace':
                    v = list(map(lambda ws: ws == ' ', v))
                resdoc[k] = v
            for k in d.user_data.keys():
                if isinstance(k, str) and k.startswith('meta_'):
                    k_noprefix = k[5:]
                    assert k_noprefix not in resdoc
                    resdoc[k[5:]] = _filtered_doc_attr(d, k, custom=True)
                    if not as_datatables and not as_arrays:
                        resdoc[k[5:]] = list(resdoc[k[5:]])
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

    :param docs: list of string tokens or spaCy documents
    :return: list of document lengths
    """
    return {dl: np.sum(d.user_data['mask']) for dl, d in docs.spacydocs.items()}


def n_tokens(docs: Corpus) -> int:
    return sum(doc_lengths(docs).values())


def doc_labels(docs: Corpus) -> List[str]:
    return sorted(docs.keys())


def doc_texts(docs: Corpus) -> Dict[str, str]:
    @parallelexec(collect_fn=merge_dicts_sorted)
    def _doc_texts(tokens):
        texts = {}
        for dl, dtok in tokens.items():
            texts[dl] = ''
            for t, ws in zip(dtok['token'], dtok['whitespace']):
                texts[dl] += t
                if ws:
                    texts[dl] += ' '
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


def tokens_datatable(docs: Corpus) -> FRAME_TYPE:
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

    tokens = doc_tokens(docs, only_non_empty=False, with_metadata=True, as_datatables=True)
    dfs = _tokens_datatable(_paralleltask(docs, tokens))

    if dfs:
        res = pd_dt_concat(dfs)
    else:
        res = pd_dt_frame({'doc': [], 'position': [], 'token': []})

    return pd_dt_sort(res, ['doc', 'position'])


def tokens_dataframe(docs: Corpus) -> pd.DataFrame:
    # note that generating a datatable first and converting it to pandas is faster than generating a pandas data
    # frame right away

    df = tokens_datatable(docs)

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

#%% modifying Corpus functions

# def to_lowercase(docs: Corpus, inplace=True):
#     # vocab = vocabulary(docs)
#     stringstore = docs.nlp.vocab.strings
#     for d in docs.spacydocs.values():
#         for i, (thash, m) in enumerate(zip(d.user_data['processed'], d.user_data['mask'])):
#             if m:
#                 t_lwr = stringstore[thash].lower()
#                 d.user_data['processed'][i] = stringstore.add(t_lwr)


def transform_tokens(docs: Corpus, func: Callable, inplace=True, **kwargs):
    # TODO: inplement copy for "inplace=False"
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
            arr_replace(d.user_data['processed'], replace_from, replace_to, inplace=True)


def to_lowercase(docs: Corpus, inplace=True):
    return transform_tokens(docs, str.lower, inplace=inplace)


def to_uppercase(docs: Corpus, inplace=True):
    return transform_tokens(docs, str.upper, inplace=inplace)


#%%


def _filtered_doc_tokens(doc: Doc, tokens_as_hashes=False):
    hashes = doc.user_data['processed'][doc.user_data['mask']]
    if tokens_as_hashes:
        return hashes
    else:
        return list(map(lambda hash: doc.vocab.strings[hash], hashes))


def _filtered_doc_attr(doc: Doc, attr: str, custom=False, stringified=True):
    if custom:
        return doc.user_data[attr][doc.user_data['mask']]
    else:
        if stringified:
            attr += '_'
        return [getattr(t, attr) for t, m in zip(doc, doc.user_data['mask']) if m]
