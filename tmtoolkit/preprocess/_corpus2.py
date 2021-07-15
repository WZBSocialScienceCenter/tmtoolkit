import os
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from functools import partial, wraps
from inspect import signature

import numpy as np
import spacy
from spacy.tokens import Doc, Token, DocBin
#from spacy.attrs import ORTH, SPACY, POS, LEMMA
from loky import get_reusable_executor
from typing import Dict, Union

from ..utils import greedy_partitioning, merge_dicts, empty_chararray, as_chararray
from .._pd_dt_compat import pd_dt_frame

from ._common import DEFAULT_LANGUAGE_MODELS


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
            self.language = self.nlp.lang
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
            self.language = language

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


#%%

merge_dicts_sorted = partial(merge_dicts, sort_keys=True)


# def parallelexec(collect_fn):
#     def deco_fn(fn):
#         @wraps(fn)
#         def inner_fn(docs, *args, **kwargs):
#             if isinstance(docs, Corpus) and docs.procexec:
#                 print(f'{os.getpid()}: distributing function {fn} for {len(docs)} docs')
#                 if args:
#                     fn_argnames = list(signature(fn).parameters.keys())
#                     # first argument in `fn` is always the documents dict -> we skip this
#                     if len(fn_argnames) <= len(args):
#                         raise ValueError(f'function {fn} does not accept enough additional arguments')
#                     kwargs.update({fn_argnames[i+1]: v for i, v in enumerate(args)})
#                 res = docs.procexec.map(partial(fn, **kwargs), docs._workers_handle)
#                 if collect_fn:
#                     return collect_fn(res)
#                 else:
#                     return None
#             else:
#                 print(f'{os.getpid()}: directly applying function {fn} to {len(docs)} docs')
#                 return fn(docs, *args, **kwargs)
#
#         return inner_fn
#
#     return deco_fn


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
