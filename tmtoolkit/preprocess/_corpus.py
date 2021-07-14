import os
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from functools import partial, wraps
from inspect import signature

import numpy as np
import spacy
from loky import get_reusable_executor

from ..utils import greedy_partitioning, merge_dicts

from ._common import DEFAULT_LANGUAGE_MODELS


class Corpus:
    STD_ATTRS = ['pos', 'lemma', 'whitespace']

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

        self._docs = None
        self._docs_meta = None
        self._vocab = None
        self._workers_data = None
        self._workers_handle = None
        self._metadata_attrs = {k: None for k in self.STD_ATTRS}  # metadata key -> default value

        self._tokenize(docs)

    def __len__(self):
        """
        Dict method to return number of documents.

        :return: number of documents
        """
        return len(self._docs_meta['label'])
    #
    # def __getitem__(self, doc_label):
    #     """
    #     dict method for retrieving document with label `doc_label` via ``corpus[<doc_label>]``.
    #     """
    #     if doc_label not in self.docs:
    #         raise KeyError('document `%s` not found in corpus' % doc_label)
    #     return self.docs[doc_label]

    # def __setitem__(self, doc_label, doc_text):
    #     """dict method for setting a document with label `doc_label` via ``corpus[<doc_label>] = <doc_text>``."""
    #     if not isinstance(doc_label, str):
    #         raise KeyError('`doc_label` must be a string')
    #
    #     if not isinstance(doc_text, str):
    #         raise ValueError('`doc_text` must be a string')
    #
    #     self.docs[doc_label] = doc_text

    # def __delitem__(self, doc_label):
    #     """dict method for removing a document with label `doc_label` via ``del corpus[<doc_label>]``."""
    #     if doc_label not in self.docs:
    #         raise KeyError('document `%s` not found in corpus' % doc_label)
    #     del self.docs[doc_label]

    # def __iter__(self):
    #     """dict method for iterating through the document labels."""
    #     return self.docs.__iter__()
    #
    # def __contains__(self, doc_label):
    #     """dict method for checking whether `doc_label` exists in this corpus."""
    #     return doc_label in self.docs
    #
    # def items(self):
    #     """dict method to retrieve pairs of document labels and texts."""
    #     return self.docs.items()
    #
    # def keys(self):
    #     """dict method to retrieve document labels."""
    #     return self.docs.keys()
    #
    # def values(self):
    #     """dict method to retrieve document texts."""
    #     return self.docs.values()
    #
    # def get(self, *args):
    #     """dict method to retrieve a specific document like ``corpus.get(<doc_label>, <default>)``."""
    #     return self.docs.get(*args)

    @property
    def docs(self):
        return doc_tokens(self)

    def _tokenize(self, docs):
        tokenizerpipe = self.nlp.pipe(docs.values(), n_process=self.n_max_workers)
        spacydocs = dict(zip(docs.keys(), tokenizerpipe))
        self._vocab = self.nlp.vocab

        self._docs_meta = {
            'label': list(docs.keys()),
            'mask': np.repeat(False, len(docs)),
        }

        docs_lengths = dict(zip(self._docs_meta['label'], [int(len(d)) for d in spacydocs.values()]))
        workers_doclabels = greedy_partitioning(docs_lengths, k=self.n_max_workers, return_only_labels=True)

        self._workers_data = []
        self._workers_handle = []
        for i, doclabels in enumerate(workers_doclabels):
            worker_doclengths = [docs_lengths[dl] for dl in doclabels]
            shape = (sum(worker_doclengths), len(self.STD_ATTRS) + 2)
            # allocate num. tokens * num. attributes per token * 8 bytes for uint64
            alloc_size = shape[0] * shape[1] * 8
            shm = SharedMemory(create=True, size=alloc_size)
            shmarr = np.ndarray(shape=shape, dtype='uint64', buffer=shm.buf)

            # tokens as IDs
            shmarr[:, 0] = np.concatenate([spacydocs[dl].to_array('ORTH') for dl in doclabels], dtype=shmarr.dtype)
            # mask
            shmarr[:, 1] = np.repeat(1, shape[0]).astype(dtype=shmarr.dtype)

            # other token "meta" data
            for j, attr in enumerate(self.STD_ATTRS, 2):
                if attr != 'whitespace':
                    shmarr[:, j] = np.concatenate([spacydocs[dl].to_array(attr.upper()) for dl in doclabels],
                                                  dtype=shmarr.dtype)
                else:
                    shmarr[:, j] = np.concatenate([[bool(t.whitespace_ != '') for t in spacydocs[dl]]
                                                   for dl in doclabels], dtype=shmarr.dtype)
            worker_doclengths_dict = dict(zip(doclabels, worker_doclengths))
            self._workers_data.append((worker_doclengths_dict, shm, shmarr))
            self._workers_handle.append((worker_doclengths_dict, self._vocab, shm.name, shmarr.shape, shmarr.dtype))


#%%

merge_dicts_sorted = partial(merge_dicts, sort_keys=True)


def parallelexec(collect_fn):
    def deco_fn(fn):
        @wraps(fn)
        def inner_fn(docs, *args, **kwargs):
            if isinstance(docs, Corpus) and docs.procexec:
                print(f'{os.getpid()}: distributing function {fn} for {len(docs)} docs')
                if args:
                    fn_argnames = list(signature(fn).parameters.keys())
                    # first argument in `fn` is always the documents dict -> we skip this
                    if len(fn_argnames) <= len(args):
                        raise ValueError(f'function {fn} does not accept enough additional arguments')
                    kwargs.update({fn_argnames[i+1]: v for i, v in enumerate(args)})
                res = docs.procexec.map(partial(fn, **kwargs), docs._workers_handle)
                if collect_fn:
                    return collect_fn(res)
                else:
                    return None
            else:
                print(f'{os.getpid()}: directly applying function {fn} to {len(docs)} docs')
                return fn(docs, *args, **kwargs)

        return inner_fn

    return deco_fn


#%%


@parallelexec(collect_fn=merge_dicts_sorted)
def doc_tokens(docs):
    doclengths, vocab, shmname, shape, dtype = docs
    shm = SharedMemory(shmname)
    arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)

    i = 0
    res = {}
    for dl, n in doclengths.items():
        res[dl] = [vocab[tid].text for tid in arr[i:(i+n), 0]]
        i += n

    return res
