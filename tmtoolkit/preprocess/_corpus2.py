import os
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from functools import partial, wraps
from inspect import signature
from dataclasses import dataclass
from typing import Dict, Union, List

import numpy as np
from scipy.sparse import csr_matrix
import spacy
from spacy.tokens import Doc, Token, DocBin
#from spacy.attrs import ORTH, SPACY, POS, LEMMA
from loky import get_reusable_executor

from ..bow.dtm import create_sparse_dtm, dtm_to_dataframe, dtm_to_datatable
from ..utils import greedy_partitioning, merge_dicts, empty_chararray, as_chararray, flatten_list,\
    combine_sparse_matrices_columnwise
from .._pd_dt_compat import USE_DT, pd_dt_frame, pd_dt_concat, pd_dt_sort

from ._common import DEFAULT_LANGUAGE_MODELS, LANGUAGE_LABELS


Doc.set_extension('label', default='', force=True)
Doc.set_extension('mask', default=True, force=True)
Token.set_extension('mask', default=True, force=True)
Token.set_extension('processed', default=0, force=True)
Doc.set_extension('token_metadata_keys', default=[], force=True)


class Corpus:
    STD_TOKEN_ATTRS = ['whitespace', 'pos', 'lemma']

    def __init__(self, docs, language=None, language_model=None,
                 spacy_instance=None, spacy_disable=('parser', 'ner'), spacy_opts=None,
                 n_max_workers=None, workers_timeout=10):
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

        if n_max_workers is None:
            self.n_max_workers = mp.cpu_count()
        else:
            if not isinstance(n_max_workers, int):
                raise ValueError('`n_max_workers` must be an integer or None')
            if n_max_workers > 0:
               self.n_max_workers = n_max_workers
            else:
                self.n_max_workers = mp.cpu_count() + n_max_workers

        self.procexec = get_reusable_executor(max_workers=self.n_max_workers, timeout=workers_timeout) \
            if self.n_max_workers > 1 else None

        if not isinstance(docs, dict):
            raise ValueError('`docs` must be a dictionary')
        elif len(docs) > 0 and not isinstance(next(iter(docs.values())), str):
            raise ValueError('`docs` values must be strings')

        self._docs = {}
        self._tokenize(docs)

        self.workers_docs = greedy_partitioning({lbl: len(d) for lbl, d in self._docs.items()},
                                                k=self.n_max_workers, return_only_labels=True)

    def __len__(self):
        """
        Dict method to return number of documents.

        :return: number of documents
        """
        return len(self._docs)

    def __getitem__(self, doc_label):
        """
        dict method for retrieving document with label `doc_label` via ``corpus[<doc_label>]``.
        """
        if doc_label not in self.docs:
            raise KeyError('document `%s` not found in corpus' % doc_label)
        return self.docs[doc_label]

    # def __setitem__(self, doc_label, doc_text):
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

    def __contains__(self, doc_label):
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

    def get(self, *args):
        """dict method to retrieve a specific document like ``corpus.get(<doc_label>, <default>)``."""
        return self.docs.get(*args)

    @property
    def language(self):
        return self.nlp.lang

    @property
    def docs(self):
        return doc_tokens(self._docs)

    @property
    def spacydocs(self):
        return self._docs

    def _tokenize(self, docs):
        tokenizerpipe = self.nlp.pipe(docs.values(), n_process=self.n_max_workers)

        for lbl, d in dict(zip(docs.keys(), tokenizerpipe)).items():
            d._.label = lbl
            self._docs[lbl] = d


#%% decorators

merge_dicts_sorted = partial(merge_dicts, sort_keys=True)

@dataclass
class ParallelTask:
    procexec: object
    workers_assignments: list
    data: dict


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
    Token.set_extension(key, default=default, force=True)

    for d in docs.spacydocs.values():
        if d._.mask:
            if key not in d._.token_metadata_keys:
                d._.token_metadata_keys.append(key)

            for t in d:
                if t._.mask:
                    setattr(t._, key, data.get(t._.processed or t.orth, default))


def doc_tokens(docs: Union[Corpus, Dict[str, Doc]],
               only_non_empty=False, with_metadata=False, as_datatables=False, as_arrays=False):
    if isinstance(docs, Corpus):
        docs = docs.spacydocs

    res = {}
    for lbl, d in docs.items():
        if not d._.mask or (only_non_empty and len(d) == 0):
            continue

        tok = _filtered_doc_tokens(d)

        if with_metadata:
            resdoc = {'token': tok}
            for k in Corpus.STD_TOKEN_ATTRS:
                v = _filtered_doc_attr(d, k)
                if k == 'whitespace':
                    v = list(map(lambda ws: ws == ' ', v))
                resdoc[k] = v
            for k in d._.token_metadata_keys:
                k_noprefix = k[5:]
                assert k_noprefix not in resdoc
                resdoc[k[5:]] = _filtered_doc_attr(d, k, custom=True)
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


def doc_lengths(docs: Corpus):
    """
    Return document length (number of tokens in doc.) for each document.

    :param docs: list of string tokens or spaCy documents
    :return: list of document lengths
    """
    return {dl: sum(t._.mask for t in dt) for dl, dt in docs.spacydocs.items()}


def n_tokens(docs: Corpus):
    return sum(doc_lengths(docs).values())


def doc_labels(docs: Corpus):
    return sorted(docs.keys())


def doc_texts(docs: Corpus):
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

    task = ParallelTask(docs.procexec, docs.workers_docs, doc_tokens(docs, with_metadata=True))
    return _doc_texts(task)


def vocabulary(docs: Union[Corpus, Dict[str, List[str]]], sort=False):
    if isinstance(docs, Corpus):
        tok = doc_tokens(docs).values()
    else:
        tok = docs.values()

    v = set(flatten_list(tok))

    if sort:
        return sorted(v)
    else:
        return v


def vocabulary_size(docs: Corpus):
    return len(vocabulary(docs))


def tokens_with_metadata(docs: Corpus):
    return doc_tokens(docs, with_metadata=True, as_datatables=True)


def tokens_datatable(docs: Corpus):
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
    task = ParallelTask(docs.procexec, docs.workers_docs, tokens)
    dfs = _tokens_datatable(task)

    if dfs:
        res = pd_dt_concat(dfs)
    else:
        res = pd_dt_frame({'doc': [], 'position': [], 'token': []})

    return pd_dt_sort(res, ['doc', 'position'])


def tokens_dataframe(docs: Corpus):
    # note that generating a datatable first and converting it to pandas is faster than generating a pandas data
    # frame right away

    df = tokens_datatable(docs)

    if USE_DT:
        df = df.to_pandas()

    return df.set_index(['doc', 'position'])


def tokens_with_pos_tags(docs: Corpus):
    """
    Document tokens with POS tag as dict with mapping document label to datatable. The datatables have two
    columns, ``token`` and ``pos``.
    """
    return {dl: df[:, ['token', 'pos']] if USE_DT else df.loc[:, ['token', 'pos']]
            for dl, df in doc_tokens(docs, with_metadata=True, as_datatables=True).items()}


def corpus_summary(docs, max_documents=None, max_tokens_string_length=None):
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


def print_summary(docs, max_documents=None, max_tokens_string_length=None):
    print(corpus_summary(docs, max_documents=max_documents, max_tokens_string_length=max_tokens_string_length))


def dtm(docs: Corpus, as_datatable=False, as_dataframe=False, dtype=None):
    @parallelexec(collect_fn=list)
    def _sparse_dtms(docs):
        vocab = vocabulary(docs, sort=True)
        alloc_size = sum(len(set(dtok)) for dtok in docs.values())  # sum of *unique* tokens in each document

        return (create_sparse_dtm(vocab, docs.values(), alloc_size, vocab_is_sorted=True),
                docs.keys(),
                vocab)

    if len(docs) > 0:
        tokens = doc_tokens(docs)
        task = ParallelTask(docs.procexec, docs.workers_docs, tokens)
        w_dtms, w_doc_labels, w_vocab = zip(*_sparse_dtms(task))
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


#%%

def _token_text(d: Doc, t: Token):
    return d.vocab.strings[t._.processed] if t._.processed else t.text


def _filtered_doc_tokens(doc: Doc):
    return [_token_text(doc, t) for t in doc if t._.mask]


def _filtered_doc_attr(doc: Doc, attr: str, custom=False, stringified=True):
    if custom:
        return [getattr(t._, attr) for t in doc if t._.mask]
    else:
        if stringified:
            attr += '_'
        return [getattr(t, attr) for t in doc if t._.mask]
