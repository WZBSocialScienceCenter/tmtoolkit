"""
Internal module that implements functions that operate on :class:`~tmtoolkit.corpus.Corpus` objects.

The source is separated into sections using a ``#%% ...`` marker.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""
import logging
import operator
import os
import unicodedata
from copy import copy
from functools import partial, wraps
from inspect import signature
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Union, List, Callable, Optional, Any, Iterable, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from spacy.tokens import Doc

from ..bow.dtm import create_sparse_dtm, dtm_to_dataframe
from ..utils import merge_dicts, merge_counters, empty_chararray, as_chararray, \
    flatten_list, combine_sparse_matrices_columnwise, arr_replace, pickle_data, unpickle_file, merge_sets, \
    merge_lists_extend, merge_lists_append
from ..tokenseq import token_lengths, token_ngrams, token_match_multi_pattern, index_windows_around_matches, \
    token_match_subsequent, token_join_subsequent, npmi, token_collocations
from ..types import OrdCollection, UnordCollection, OrdStrCollection, UnordStrCollection, StrOrInt

from ._common import LANGUAGE_LABELS, simplified_pos
from ._corpus import Corpus
from ._helpers import _filtered_doc_token_attr, _filtered_doc_tokens, _corpus_from_tokens, \
    _ensure_writable_array, _check_filter_args, _token_pattern_matches, _match_against, _apply_matches_array


logger = logging.getLogger('tmtoolkit')
logger.addHandler(logging.NullHandler())


#%% parallel execution helpers and other decorators

merge_dicts_sorted = partial(merge_dicts, sort_keys=True)

@dataclass
class ParallelTask:
    """A parallel execution task for a loky reusable process executor."""
    # loky reusable process executor
    procexec: object
    # assignments of data chunks in `data` to workers; ``workers_assignments[i]`` contains list of keys in `data` which
    # worker ``i`` is assigned to work on
    workers_assignments: List[List[str]]
    # dict mapping data chunk key to data chunk
    data: dict


def _paralleltask(corpus: Corpus, tokens=None):
    """
    Helper function to generate a :class:`~ParallelTask` for the reusable process executor and the worker process
    assignments in the :class:`~tmtoolkit.corpus.Corpus` Corpus `corpus`. By default, use `corpus`' document tokens as
    data chunks, otherwise use `tokens`.
    """
    return ParallelTask(corpus.procexec, corpus.workers_docs,
                        doc_tokens(corpus) if tokens is None else tokens)


def parallelexec(collect_fn: Callable) -> Callable:
    """
    Decorator function for parallel processing. Using this decorator on a function `fn` will run this function in
    parallel, each parallel instance processing only a chunk of the whole data. After the results of all parallel
    instances were collected, they're merged to a single data object using `collect_fn`. Most Corpus functions will
    produce a dict (e.g. mapping document labels to some document-specific data), so :func:`tmtoolkit.utils.merge_dict`
    can be used as collection function.

    The function `fn` must accept a data chunk `data` *as first argument* which is always a dict and optionally
    additonal positional and/or keyword arguments.

    When a function `fn` is decorated with this decorator, you must create a :class:`~ParallelTask` object, e.g. with
    the :func:`~_paralleltask` helper, and call `fn` with this object.

    If a Corpus object and hence a :class:`~ParallelTask` object created from it has not enabled parallel processing,
    `fn` will be executed as usual in the main process.

    :param collect_fn: function to be called for combining the results from the parallel function executions; when
                       returning a dict, :func:`tmtoolkit.utils.merge_dict` can be used as collection function;
                       if this is None, simply always return None
    :return: wrapped function
    """
    def deco_fn(fn):
        @wraps(fn)
        def inner_fn(task: ParallelTask, *args, **kwargs):
            if task.procexec and len(task.data) > 1:   # parallel processing enabled and possibly useful
                logger.debug(f'{os.getpid()}: distributing function {fn} for {len(task.data)} items to '
                             f'{len(task.workers_assignments)} workers')
                if args:
                    # we have positional arguments -> map these to kwargs so that they don't overwrite the important
                    # first argument (the data chunk)
                    fn_argnames = list(signature(fn).parameters.keys())
                    # first argument in `fn` is always the data dict -> we skip this
                    if len(fn_argnames) <= len(args):
                        raise ValueError(f'function {fn} does not accept enough additional arguments')
                    kwargs.update({fn_argnames[i+1]: v for i, v in enumerate(args)})

                # generate a list where each item in the list represents the data chunk for the worker process
                workers_data = [{lbl: task.data[lbl] for lbl in itemlabels      # data chunks are dicts
                                 if lbl in task.data.keys()}
                                for itemlabels in task.workers_assignments]

                # execute `fn` in parallel, pass the worker data and additional keyword arguments
                res = task.procexec.map(partial(fn, **kwargs), workers_data)

                # combine the result
                if collect_fn:
                    return collect_fn(res)
                else:
                    return None
            else:               # parallel processing disabled
                logger.debug(f'{os.getpid()}: directly applying function {fn} to {len(task.data)} items')
                res = fn(task.data, *args, **kwargs)
                if collect_fn is merge_lists_append:
                    return [res]
                else:
                    return res

        return inner_fn

    return deco_fn


def corpus_func_copiable(fn: Callable) -> Callable:
    """
    Decorator for a Corpus function `fn` with an optional argument ``inplace``. This decorator makes sure that if
    `fn` is called with ``inplace=False``, the passed corpus will be copied before `fn` is applied to it. Then,
    the modified copy of corpus is returned. If ``inplace=True``, `fn` is applied as usual.

    If you decorate a Corpus function with this decorator, the first argument of the Corpus function should be
    defined as positional-only argument, i.e. ``def corpfunc(docs, /, some_arg, other_arg, ...): ...``.

    :param fn: Corpus function `fn` with an optional argument ``inplace``
    :return: wrapper function of `fn`
    """
    @wraps(fn)
    def inner_fn(*args, **kwargs):
        if not isinstance(args[0], Corpus):
            raise ValueError('first argument must be a Corpus object')

        if 'inplace' in kwargs:
            inplace = kwargs.pop('inplace')
        else:
            inplace = True

        # get Corpus object `corp`, optionally copy it
        if inplace:
            corp = args[0]
        else:
            corp = copy(args[0])   # copy of this Corpus, a new object with same data but the *same* SpaCy instance

        # apply fn to `corp`, passing all other arguments
        ret = fn(corp, *args[1:], **kwargs)
        if ret is None:         # most Corpus functions return None
            if inplace:         # no need to return Corpus since it was modified in-place
                return None
            else:               # return the modified copy
                return corp
        else:                   # for Corpus functions that return something
            if inplace:
                return ret
            else:
                return corp, ret    # always return the modified Corpus copy first

    return inner_fn


def corpus_func_filters_tokens(fn: Callable) -> Callable:
    """
    Decorator for a Corpus function `fn` that possibly filters tokens. Makes sure that the
    :attr:`~tmtoolkit.corpus.Corpus._tokens_masked` attribute is set whenever such a function is called.

    :param fn: Corpus function `fn` that filters tokens
    :return: wrapper function of `fn`
    """
    @wraps(fn)
    def inner_fn(*args, **kwargs):
        if not isinstance(args[0], Corpus):
            raise ValueError('first argument must be a Corpus object')

        corp = args[0]

        # apply fn to `corp`, passing all other arguments
        fn(corp, *args[1:], **kwargs)

        corp._tokens_masked = True

        return corp

    return inner_fn


def corpus_func_processes_tokens(fn):
    """
    Decorator for a Corpus function `fn` that possibly processes (transforms) tokens. Makes sure that the
    :attr:`~tmtoolkit.corpus.Corpus._tokens_processed` attribute is set whenever such a function is called.

    .. warning:: Use this decorator only for functions that accept an ``inplace`` argument.

    :param fn: Corpus function `fn` that processes (transforms) tokens
    :return: wrapper function of `fn`
    """
    @wraps(fn)
    def inner_fn(*args, **kwargs):
        if not isinstance(args[0], Corpus):
            raise ValueError('first argument must be a Corpus object')

        corp = args[0]

        # apply fn to `corp`, passing all other arguments
        res = fn(corp, *args[1:], **kwargs)

        if kwargs.get('inplace', True):
            corp._tokens_processed = True
        else:
            if isinstance(res, tuple):
                for r in res:
                    if isinstance(res, Corpus):
                        r._tokens_processed = True
            else:
                res._tokens_processed = True
        return res

    return inner_fn


#%% Corpus functions with readonly access to Corpus data


def doc_tokens(docs: Union[Corpus, Dict[str, Doc]],
               select: Optional[Union[str, UnordStrCollection]] = None,
               only_non_empty=False,
               tokens_as_hashes=False,
               with_attr: Union[bool, str, OrdStrCollection] = False,
               with_mask=False,
               with_spacy_tokens=False,
               as_tables=False,
               as_arrays=False,
               apply_document_filter=True,
               apply_token_filter=True,
               force_unigrams=False) \
        -> Union[
               # multiple documents
               Dict[str, Union[List[Union[str, int]],
                               np.ndarray,
                               Dict[str, Union[list, np.ndarray]],
                               pd.DataFrame]],
               # single document
               List[Union[str, int]],
               np.ndarray,
               Dict[str, Union[list, np.ndarray]],
               pd.DataFrame
           ]:
    """
    Retrieve documents' tokens from a Corpus or dict of SpaCy documents. Optionally also retrieve document and token
    attributes.

    :param docs: a Corpus object or a dict mapping document labels to SpaCy `Doc` objects
    :param select: if not None, this can be a single string or a sequence of strings specifying the documents to fetch;
                   if `select` is a string, retrieve only this specific document; if `select` is a list/tuple/set,
                   retrieve only the documents in this collection
    :param only_non_empty: if True, only return non-empty result documents
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
                             as from `SpaCy StringStore <https://spacy.io/api/stringstore/>`_
    :param with_attr: also return document and token attributes along with each token; if True, returns all default
                      attributes and custom defined attributes; if string, return this specific attribute; if list or
                      tuple, returns attributes specified in this sequence
    :param with_mask: if True, also return the document and token mask attributes; this disables the document or token
                      filtering (i.e. `apply_token_filter` and `apply_document_filter` are set to False)
    :param with_spacy_tokens: if True, also return a token attribute for the original SpaCy token string; the attribute
                              name will be "text"
    :param as_tables: return result as dataframe with tokens and document and token attributes in columns
    :param as_arrays: return result as NumPy arrays instead of lists
    :param apply_document_filter: if False, ignore document filter mask
    :param apply_token_filter: if False, ignore token filter mask
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :return: by default, a dict mapping document labels to document tokens data, which can be of different form,
             depending on the arguments passed to this function:
             (1) list of token strings or hash integers;
             (2) NumPy array of token strings or hash integers;
             (3) dict containing ``"token"`` key with values from (1) or (2) and document and token attributes with
                 their values as list or NumPy array;
             (4) dataframe with tokens and document and token attributes in columns;
             if `select` is a string not a dict of documents is returned, but a single document with one of the 4 forms
             described before
    """
    if select is None:
        select_docs = None
    else:
        if isinstance(select, str):
            select_docs = {select}
        else:
            select_docs = set(select)

    # prepare `with_attr_list`: a list that contains the document and token attributes to be fetched
    add_std_attrs = False
    with_attr_list = []
    if isinstance(with_attr, str):
        with_attr_list = [with_attr]        # populated by single string
    elif isinstance(with_attr, list):
        with_attr_list = with_attr.copy()   # list already given, copy it
    elif isinstance(with_attr, tuple):
        with_attr_list = list(with_attr)    # tuple given, convert to list
    elif with_attr is True:
        add_std_attrs = True                # True specified, means load standard attributes

    # add "text" attribute (original SpaCy token text) if requested by `with_spacy_tokens`
    if with_spacy_tokens and 'text' not in with_attr_list:
        if with_attr_list:
            with_attr_list = ['text'] + with_attr_list
        else:
            with_attr_list.append('text')

    # add document and/or token mask if requested by `with_mask`
    if with_mask:
        if with_attr_list:
            for attr in ('doc_mask', 'mask'):       # prevent duplicates in `with_attr_list`
                if attr not in with_attr_list:
                    with_attr_list.append(attr)
        else:
            with_attr_list.extend(['doc_mask', 'mask'])

    # if requested by `with_attr = True`, add standard token attributes
    if add_std_attrs:
        with_attr_list.extend(Corpus.STD_TOKEN_ATTRS)

    # set `with_mask` so that it reflects either the `with_mask` argument setting or a mask occurrence in the attr. list
    with_mask = with_mask or 'mask' in with_attr_list or 'doc_mask' in with_attr_list

    if with_mask:  # requesting the document and token mask disables the filtering
        apply_token_filter = 'mask' not in with_attr_list
        apply_document_filter = 'doc_mask' not in with_attr_list

    ng = 1      # ngram setting; default is unigram
    ng_join_str = None
    doc_attrs = {}
    custom_token_attrs_defaults = None   # set to None if docs is not a Corpus but a dict of SpaCy Docs

    if isinstance(docs, Corpus):    # if `docs` is a Corpus object, we can retrieve some additional information
        # get ngram setting
        if not force_unigrams:
            ng = docs.ngrams
            ng_join_str = docs.ngrams_join_str

        # get document attributes with default values
        if add_std_attrs or with_attr_list:
            doc_attrs = docs.doc_attrs_defaults.copy()

        # rely on custom token attrib. w/ defaults as reported from Corpus
        custom_token_attrs_defaults = docs.custom_token_attrs_defaults

        # if `docs` is a Corpus object, we obtain the SpaCy documents
        docs = docs.spacydocs_ignore_filter if with_mask else docs.spacydocs

    # subset documents
    if select_docs is not None:
        docs = {lbl: docs[lbl] for lbl in select_docs}

    # make sure `doc_attrs` contains only the attributes listed in `with_attr_list`; if `with_attr = True`, don't
    # filter the `doc_attrs`
    if with_attr_list and not add_std_attrs:
        doc_attrs = {k: doc_attrs[k] for k in with_attr_list if k in doc_attrs.keys()}

    # default setting for "mask" document attribute
    if 'doc_mask' in with_attr_list:
        doc_attrs['doc_mask'] = True
        with_attr_list.remove('doc_mask')

    res = {}
    for lbl, d in docs.items():     # iterate through SpaCy documents with label `lbl` and Doc objects `d`
        # skip this document if it is empty and `only_non_empty` is True
        # or if the document is masked and `apply_document_filter` is True
        if (only_non_empty and len(d) == 0) or (apply_document_filter and not d._.mask):
            continue

        # get the tokens of the document
        tok = _filtered_doc_tokens(d, tokens_as_hashes=tokens_as_hashes, apply_filter=apply_token_filter,
                                   as_array=as_arrays or as_tables)

        if ng > 1:  # no unigrams, transform to joined ngrams
            tok = token_ngrams(tok, n=ng, join=True, join_str=ng_join_str)

        if with_attr_list or doc_attrs:   # extract document and token attributes
            resdoc = {}      # result document

            # 1. document attributes
            for k, default in doc_attrs.items():
                if k in with_attr_list:
                    with_attr_list.remove(k)
                a = 'mask' if k == 'doc_mask' else k
                v = getattr(d._, a)
                if v is None:     # can't use default arg (third arg) in `getattr` b/c Doc extension *always* returns
                                  # a value; it will be None by Doc extension default
                    v = default
                resdoc[k] = [v] * len(tok) if as_tables else v

            # 2. always add tokens
            resdoc['token'] = tok

            # identify standard (SpaCy) token attributes
            all_user_attrs = d.user_data.keys() if custom_token_attrs_defaults is None \
                                                else custom_token_attrs_defaults.keys()
            all_user_attrs = [k for k in all_user_attrs if k not in {'processed', 'mask'}]
            if add_std_attrs and all_user_attrs:
                with_attr_list.extend(all_user_attrs)
            if 'mask' not in all_user_attrs:
                all_user_attrs.append('mask')
            spacy_attrs = [k for k in with_attr_list if k not in all_user_attrs]

            # 3. add standard token attributes to the result document
            for k in spacy_attrs:
                v = _filtered_doc_token_attr(d, k, custom=False, apply_filter=apply_token_filter)
                if k == 'whitespace':   # whitespace as boolean list
                    v = list(map(lambda ws: ws != '', v))
                if ng > 1:  # attributes are also joined as ngrams (transform to strings before)
                    v = token_ngrams(list(map(str, v)), n=ng, join=True, join_str=ng_join_str)
                resdoc[k] = v

            # identify user (custom) token attributes
            # if docs is not a Corpus but a dict of SpaCy Docs, use the keys in `user_data` as custom token attributes
            # -> risky since these `user_data` dict keys may differ between documents
            if add_std_attrs:
                user_attrs = [k for k in all_user_attrs if k in with_attr_list]
            else:
                user_attrs = [k for k in with_attr_list if k in all_user_attrs]

            if 'mask' in user_attrs:
                user_attrs = [k for k in user_attrs if k != 'mask'] + ['mask']   # order

            # 4. add custom token attributes to the result document
            for k in user_attrs:
                if isinstance(k, str):
                    default = None if custom_token_attrs_defaults is None else custom_token_attrs_defaults.get(k, None)
                    v = _filtered_doc_token_attr(d, k, default=default, custom=True, apply_filter=apply_token_filter)
                    if not as_tables and not as_arrays:
                        v = list(v)
                    if ng > 1:  # attributes are also joined as ngrams (transform to strings before)
                        v = token_ngrams(list(map(str, v)), n=ng, join=True, join_str=ng_join_str)
                    resdoc[k] = v
            res[lbl] = resdoc
        else:   # no attributes; result document is simply the (unigram / ngram) tokens
            if as_tables:
                res[lbl] = {'token': tok}
            else:
                res[lbl] = tok

    if as_tables:   # convert to dict of dataframe
        res = dict(zip(res.keys(), map(pd.DataFrame, res.values())))
    elif as_arrays and with_attr_list:     # convert to dict of arrays
        # nested: dict with attribute values
        res = dict(zip(res.keys(),
                       [{k: v if k == 'token' else np.array(v) for k, v in d.items()}
                        for d in res.values()]))

    if isinstance(select, str):     # return single document
        if only_non_empty and not res:
            raise ValueError(f'selected document "{select}" is empty but only non-empty documents should be retrieved')
        return res[select]
    else:
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
    return {lbl: token_lengths(tok) for lbl, tok in doc_tokens(docs).items()}


def doc_labels(docs: Corpus, sort=False) -> List[str]:
    """
    Return list of the documents' labels.

    :param docs: a Corpus object
    :param sort: if True, return as sorted list
    :return: list of the documents' labels
    """
    if sort:
        return sorted(docs.keys())
    else:
        return list(docs.keys())


def doc_texts(docs: Corpus, collapse: Optional[str] = None) -> Dict[str, str]:
    """
    Return reconstructed document text from documents in `docs`. By default, uses whitespace token attribute to collapse
    tokens to document text, otherwise custom `collapse` string.

    :param docs: a Corpus object
    :param collapse: if None, use whitespace token attribute for collapsing tokens, otherwise use custom string
    :return: dict with reconstructed document text per document label
    """
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

    return _doc_texts(_paralleltask(docs, doc_tokens(docs, with_attr=True)))


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

    :param docs: a :class:`Corpus` object
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

    # TODO: not sure if the version that uses hashes is faster
    # res = _doc_frequencies(_paralleltask(docs, doc_tokens(docs, tokens_as_hashes=True)),
    #                        norm=len(docs) if proportions else 1)
    # return dict(zip(map(lambda h: docs.nlp.vocab.strings[h], res.keys()), res.values()))

    return _doc_frequencies(_paralleltask(docs), norm=len(docs) if proportions else 1)


def doc_vectors(docs: Corpus, omit_empty=False) -> Dict[str, np.ndarray]:
    """
    Return a vector representation for each document in `docs`. The vector representation's size corresponds to the
    vector width of the language model that is used (usually 300).

    .. note:: The Corpus object `docs` must use a SpaCy language model with word vectors (i.e. an *_md* or *_lg* model).

    :param docs: a :class:`Corpus` object
    :param omit_empty: omit empty documents
    :return: dict mapping document label to vector representation of the document
    """

    if docs.nlp.meta.get('vectors', {}).get('width', 0) == 0:
        raise RuntimeError("Corpus object `docs` doesn't use a SpaCy language model with word vectors; you should "
                           "specify a different language model (i.e. an ..._md or ..._lg model) via "
                           "`language_model` parameter when initializing the Corpus object")

    if docs.is_processed or docs.tokens_filtered:
        raise RuntimeError('passed Corpus object `docs` contains filtered and/or processed tokens; '
                           'you need to apply `compact()` to this Corpus object before using this function')

    return {dl: d.vector for dl, d in docs.spacydocs.items() if not omit_empty or len(d) > 0}


def token_vectors(docs: Corpus, omit_oov=True) -> Dict[str, np.ndarray]:
    """
    Return a token vectors matrix for each document in `docs`. This matrix is of size *n* by *m* where *n* is
    the number of tokens in the document and *m* is the vector width of the language model that is used (usually 300).
    If `omit_oov` is True, *n* will be number of tokens in the document **for which there is a word vector** in
    used the language model.

    .. note:: The Corpus object `docs` must use a SpaCy language model with word vectors (i.e. an *_md* or *_lg* model).

    :param docs: a :class:`Corpus` object
    :param omit_oov: omit "out of vocabulary" tokens, i.e. tokens without a vector
    :return: dict mapping document label to token vectors matrix
    """
    if docs.nlp.meta.get('vectors', {}).get('width', 0) == 0:
        raise RuntimeError("Corpus object `docs` doesn't use a SpaCy language model with word vectors; you should "
                           "specify a different language model (i.e. an ..._md or ..._lg model) via "
                           "`language_model` parameter when initializing the Corpus object")

    if docs.is_processed or docs.tokens_filtered:   # tokens are processed and/or filtered
        res = {}
        vocab = docs.nlp.vocab
        # get token hashes
        for lbl, tok_hashes in doc_tokens(docs, tokens_as_hashes=True, force_unigrams=True).items():
            # get token type vector for hash from SpaCy Vocab
            tok_vecs = [vocab.get_vector(h) for h in tok_hashes if not omit_oov or vocab.has_vector(h)]
            res[lbl] = np.vstack(tok_vecs) if tok_vecs else np.array([], dtype='float32')
        return res
    else:   # fast track: directly get vector from tokens of SpaCy docs
        return {dl: np.vstack([t.vector for t in d if not (omit_oov and t.is_oov)])
                               if len(d) > 0 else np.array([], dtype='float32')
                for dl, d in docs.spacydocs.items()}


def vocabulary(docs: Union[Corpus, Dict[str, List[str]]], tokens_as_hashes=False, force_unigrams=False, sort=False)\
        -> Union[Set[StrOrInt], List[StrOrInt]]:
    """
    Return the vocabulary, i.e. the set or sorted list of unique token types, of a Corpus or a dict of token strings.

    :param docs: a :class:`Corpus` object or a dict of token strings
    :param tokens_as_hashes: use token hashes instead of token strings
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :param sort: if True, sort the vocabulary
    :return: set or, if `sort` is True, a sorted list of unique token types
    """
    if isinstance(docs, Corpus):
        tok = doc_tokens(docs, tokens_as_hashes=tokens_as_hashes, force_unigrams=force_unigrams).values()
    else:
        tok = docs.values()

    v = set(flatten_list(tok))

    if sort:
        return sorted(v)
    else:
        return v


def vocabulary_counts(docs: Corpus, tokens_as_hashes=False, force_unigrams=False) -> Counter:
    """
    Return :class:`collections.Counter` instance of vocabulary containing counts of occurrences of tokens across
    all documents.

    :param docs: a :class:`Corpus` object
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
                             as from `SpaCy StringStore <https://spacy.io/api/stringstore/>`_
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :return: :class:`collections.Counter` instance of vocabulary containing counts of occurrences of tokens across
             all documents
    """
    @parallelexec(collect_fn=merge_counters)
    def _vocabulary_counts(tokens):
        return Counter(flatten_list(tokens.values()))

    tok = doc_tokens(docs, tokens_as_hashes=tokens_as_hashes, force_unigrams=force_unigrams)

    return _vocabulary_counts(_paralleltask(docs, tok))


def vocabulary_size(docs: Union[Corpus, Dict[str, List[str]]], force_unigrams=False) -> int:
    """
    Return size of the vocabulary, i.e. number of unique token types in `docs`.

    :param docs: a :class:`Corpus` object or a dict of token strings
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :return: size of the vocabulary
    """
    return len(vocabulary(docs, tokens_as_hashes=True, force_unigrams=force_unigrams))


def tokens_with_attr(docs: Corpus, with_spacy_tokens=False, as_tables=False) \
        -> Dict[str, Union[dict, pd.DataFrame]]:
    """
    Returns tokens with document/token attributes. Shortcut for :func:`doc_tokens` with ``with_attr=True``.

    :param docs: a :class:`Corpus` object
    :param with_spacy_tokens: if True, also return a token attribute for the original SpaCy token string
    :param as_tables: return result as dataframe with tokens and document and token attributes in columns
    :return: dict mapping document label to dict or dataframe with tokens and attributes
    """
    return doc_tokens(docs, with_attr=True, with_spacy_tokens=with_spacy_tokens, as_tables=as_tables)


def tokens_table(docs: Corpus,
                 select: Optional[Union[str, UnordStrCollection]] = None,
                 tokens_as_hashes=False,
                 with_attr: Union[bool, OrdCollection] = True,
                 with_mask=False,
                 with_spacy_tokens=False,
                 apply_document_filter=True,
                 apply_token_filter=True,
                 force_unigrams=False) -> pd.DataFrame:
    """
    Generate a dataframe with tokens and document/token attributes. Result has columns "doc" (document label),
    "position" (token position in the document), "token" and optional columns for document/token attributes.

    :param docs: a :class:`Corpus` object
    :param select: if not None, this can be a single string or a sequence of strings specifying the documents to fetch
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
                             as from `SpaCy StringStore <https://spacy.io/api/stringstore/>`_
    :param with_attr: also return document and token attributes along with each token; if True, returns all default
                      attributes and custom defined attributes; if list or tuple, returns attributes specified in this
                      sequence
    :param with_mask: if True, also return the document and token mask attributes; this disables the document or token
                      filtering (i.e. `apply_token_filter` and `apply_document_filter` are set to False)
    :param with_spacy_tokens: if True, also return a token attribute for the original SpaCy token string
    :param apply_document_filter: if False, ignore document filter mask
    :param apply_token_filter: if False, ignore token filter mask
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :return: dataframe with tokens and document/token attributes
    """
    @parallelexec(collect_fn=merge_lists_extend)
    def _tokens_table(tokens):
        dfs = []
        for dl, df in tokens.items():
            n = df.shape[0]
            meta_df = pd.DataFrame({
                'doc': np.repeat(dl, n),
                'position': np.arange(n)
            })

            dfs.append(pd.concat((meta_df, df), axis=1))
        return dfs

    # get dict of dataframes
    tokens = doc_tokens(docs,
                        select={select} if isinstance(select, str) else select,
                        tokens_as_hashes=tokens_as_hashes,
                        only_non_empty=False,
                        with_attr=with_attr,
                        with_mask=with_mask,
                        with_spacy_tokens=with_spacy_tokens,
                        apply_document_filter=apply_document_filter,
                        apply_token_filter=apply_token_filter,
                        as_tables=True,
                        force_unigrams=force_unigrams)

    # transform in parallel
    dfs = _tokens_table(_paralleltask(docs, tokens))
    res = None

    if dfs:
        res = pd.concat(dfs, axis=0)

    if res is None or len(res) == 0:
        res = pd.DataFrame({'doc': [], 'position': [], 'token': []})

    return res.sort_values(['doc', 'position'])


def corpus_tokens_flattened(docs: Corpus, tokens_as_hashes=False, as_array=False, apply_document_filter=True,
                            apply_token_filter=True) -> Union[list, np.ndarray]:
    """
    Return tokens (or token hashes) from `docs` as flattened list, simply concatenating  all documents.

    :param docs: a Corpus object
    :param tokens_as_hashes: passed to :func:`doc_tokens`; if True, return token hashes instead of string tokens
    :param as_array: if True, return NumPy array instead of list
    :param apply_document_filter: passed to :func:`doc_tokens`
    :param apply_token_filter: passed to :func:`doc_tokens`
    :return: list or NumPy array (depending on `as_array`) of token strings or hashes (depending on `tokens_as_hashes`)
    """
    tok = doc_tokens(docs, only_non_empty=True, tokens_as_hashes=tokens_as_hashes, as_arrays=as_array,
                     apply_document_filter=apply_document_filter, apply_token_filter=apply_token_filter)

    if as_array:
        dtype = 'uint64' if tokens_as_hashes else 'str'
        if tok:
            return np.concatenate(list(tok.values()), dtype=dtype)
        else:
            return np.array([], dtype=dtype)
    else:
        return flatten_list(tok.values())


def corpus_num_tokens(docs: Corpus) -> int:
    """
    Return the number of tokens in a Corpus `docs`.

    :param docs: a Corpus object
    :return: number of tokens
    """
    return sum(doc_lengths(docs).values())


def corpus_num_chars(docs: Corpus) -> int:
    """
    Return the number of characters (excluding whitespace) in a Corpus `docs`.

    :param docs: a Corpus object
    :return: number of characters
    """
    return sum(sum(n) for n in doc_token_lengths(docs).values())


def corpus_collocations(docs: Corpus, threshold: Optional[float] = None,
                        min_count: int = 1, embed_tokens_min_docfreq: Optional[Union[int, float]] = None,
                        embed_tokens_set: Optional[UnordCollection] = None,
                        statistic: Callable = npmi, return_statistic=True, rank: Optional[str] = 'desc',
                        as_table=True, glue: str = ' ', **statistic_kwargs):
    """
    Identify token collocations in the corpus `docs`.

    .. seealso:: :func:`~tmtoolkit.tokenseq.token_collocations`

    :param docs: a Corpus object
    :param threshold: minimum statistic value for a collocation to enter the results; if None, results are not filtered
    :param min_count: ignore collocations with number of occurrences below this threshold
    :param embed_tokens_min_docfreq: dynamically generate the set of ``embed_tokens`` used when calling
                                     :func:`~tmtoolkit.tokenseq.token_collocations` by using a minimum document
                                     frequency (see :func:`~doc_frequencies`); if this is an integer, it is used as
                                     absolute count, if it is a float, it is used as proportion
    :param embed_tokens_set: tokens that, if occurring inside an n-gram, are not counted; see :func:`token_ngrams`
    :param statistic: function to calculate the statistic measure from the token counts; use one of the
                      ``[n]pmi[2,3]_from_counts`` functions provided in the :mod:`~tmtoolkit.tokenseq` module or provide
                      your own function which must accept parameters ``n_x, n_y, n_xy, n_total``; see
                      :func:`~tmtoolkit.tokenseq.pmi_from_counts` and :func:`~tmtoolkit.tokenseq.pmi`
                      for more information
    :param return_statistic: also return computed statistic
    :param rank: if not None, rank the results according to the computed statistic in ascending (``rank='asc'``) or
                 descending (``rank='desc'``) order
    :param as_table: return result as dataframe with columns "collocation" and optionally "statistic"
    :param glue: if not None, provide a string that is used to join the collocation tokens; must be set if
                 `as_table` is True
    :param statistic_kwargs: additional arguments passed to `statistic` function
    :return: if `as_table` is True, a dataframe with columns "collocation" and optionally "statistic";
             else same output as :func:`~tmtoolkit.tokenseq.token_collocations`, i.e. list of tuples
             ``(collocation tokens, score)`` if `return_statistic` is True, otherwise only a list of collocations
    """
    if as_table and glue is None:
        raise ValueError('`glue` cannot be None if `as_table` is True')

    tok = [corpus_tokens_flattened(docs)]    # TODO: use sentences
    vocab_counts = vocabulary_counts(docs)

    # generate ``embed_tokens`` set as used in :func:`~tmtookit.tokenseq.token_collocations`
    embed_tokens = _create_embed_tokens_for_collocations(docs, embed_tokens_min_docfreq, embed_tokens_set)

    # identify collocations
    colloc = token_collocations(tok, threshold=threshold, min_count=min_count, embed_tokens=embed_tokens,
                                vocab_counts=vocab_counts, statistic=statistic, return_statistic=return_statistic,
                                rank=rank, glue=glue, **statistic_kwargs)

    if as_table:
        if return_statistic:    # generate two columns: collocation and statistic
            if colloc:
                bg, stat = zip(*colloc)
            else:
                bg = []
                stat = []
            cols = {'collocation': bg, 'statistic': stat}
        else:                   # use only collocation column
            cols = {'collocation': colloc}
        return pd.DataFrame(cols)
    else:
        return colloc


def corpus_summary(docs: Corpus,
                   max_documents: Optional[int] = None,
                   max_tokens_string_length: Optional[int] = None) -> str:
    """
    Generate a summary of this object, i.e. the first tokens of each document and some summary statistics.

    :param docs: a Corpus object
    :param max_documents: maximum number of documents to print; ``None`` uses default value 10; set to -1 to
                          print *all* documents
    :param max_tokens_string_length: maximum string length of concatenated tokens for each document; ``None`` uses
                                     default value 50; set to -1 to print complete documents
    :return: summary as string
    """

    if max_tokens_string_length is None:
        max_tokens_string_length = docs.print_summary_default_max_tokens_string_length
    if max_documents is None:
        max_documents = docs.print_summary_default_max_documents

    if max_tokens_string_length < 0:
        raise ValueError('`max_tokens_string_length` must be non-negative')
    if max_documents < 0:
        raise ValueError('`max_documents` must be non-negative')

    summary = f'Corpus with {docs.n_docs} document' \
              f'{"s" if docs.n_docs > 1 else ""} ({docs.n_docs_masked} masked) in ' \
              f'{LANGUAGE_LABELS[docs.language].capitalize()}'

    texts = doc_texts(docs)
    dlengths = doc_lengths(docs)

    for i, (lbl, tokstr) in enumerate(texts.items()):
        tokstr = tokstr.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        if i >= max_documents:
            break
        if max_tokens_string_length >= 0 and len(tokstr) > max_tokens_string_length:
            tokstr = tokstr[:max_tokens_string_length] + '...'

        summary += f'\n> {lbl} ({dlengths[lbl]} tokens): {tokstr}'

    if len(docs) > max_documents:
        summary += f'\n(and {len(docs) - max_documents} more documents)'

    summary += f'\ntotal number of tokens: {corpus_num_tokens(docs)} / vocabulary size: {vocabulary_size(docs)}'

    states = [s for s in ('processed', 'filtered') if getattr(docs, 'tokens_' + s)]
    if docs.ngrams > 1:
        states.append(f'{docs.ngrams}-grams')
    if states:
        summary += '\ntokens are ' + (', '.join(states))

    return summary


def print_summary(docs: Corpus, max_documents=None, max_tokens_string_length=None):
    """
    Print a summary of this object, i.e. the first tokens of each document and some summary statistics.

    :param docs: a Corpus object
    :param max_documents: maximum number of documents to print; ``None`` uses default value 10; set to -1 to
                          print *all* documents
    :param max_tokens_string_length: maximum string length of concatenated tokens for each document; ``None`` uses
                                     default value 50; set to -1 to print complete documents
    """
    print(corpus_summary(docs, max_documents=max_documents, max_tokens_string_length=max_tokens_string_length))


def dtm(docs: Corpus, as_table=False, dtype=None, return_doc_labels=False, return_vocab=False) \
        -> Union[csr_matrix,
                 pd.DataFrame,
                 Tuple[Union[csr_matrix, pd.DataFrame], List[str]],
                 Tuple[Union[csr_matrix, pd.DataFrame], List[str], List[str]]]:
    """
    Generate and return a sparse document-term matrix (or alternatively a dataframe) of shape
    ``(n_docs, n_vocab)`` where ``n_docs`` is the number of documents and ``n_vocab`` is the vocabulary size.

    The rows of the matrix correspond to the *sorted* document labels, the columns of the matrix correspond to the
    *sorted* vocabulary of `docs`. Using `return_doc_labels` and/or `return_vocab`, you can additionally return these
    two lists.

    .. warning:: Setting `as_table` to True will return *dense* data, which means that it may require a lot of memory.

    :param docs: a Corpus object
    :param as_table: return result as dense pandas DataFrame
    :param dtype: use a specific matrix dtype; otherwise dtype will be uint32
    :param return_doc_labels: if True, additionally return sorted document labels that correspond to the rows of the
                              document-term matrix
    :param return_vocab: if True, additionally return the sorted vocabulary that corresponds to the columns of the
                         document-term matrix
    :return: document-term matrix as sparse matrix or dense dataframe; additionally sorted document labels and/or sorted
             vocabulary if `return_doc_labels` and/or `return_vocab` is True
    """
    @parallelexec(collect_fn=merge_lists_append)
    def _sparse_dtms(chunk):
        vocab = vocabulary(chunk, sort=True)
        alloc_size = sum(len(set(dtok)) for dtok in chunk.values())  # sum of *unique* tokens in each document

        return (create_sparse_dtm(vocab, chunk.values(), alloc_size, vocab_is_sorted=True, dtype=dtype),
                chunk.keys(),
                vocab)

    if len(docs) > 0:
        res = _sparse_dtms(_paralleltask(docs))
        w_dtms, w_doc_labels, w_vocab = zip(*res)
        dtm, vocab, dtm_doc_labels = combine_sparse_matrices_columnwise(w_dtms, w_vocab, w_doc_labels)
        # sort according to document labels
        dtm = dtm[np.argsort(dtm_doc_labels), :]
        doc_labels = np.sort(dtm_doc_labels)
    else:
        dtm = csr_matrix((0, 0), dtype=dtype or 'uint32')   # empty sparse matrix
        vocab = empty_chararray()
        doc_labels = empty_chararray()

    if as_table:
        mat = dtm_to_dataframe(dtm, doc_labels, vocab)
    else:
        mat = dtm

    if return_doc_labels and return_vocab:
        return mat, doc_labels.tolist(), vocab.tolist()
    elif return_doc_labels and not return_vocab:
        return mat, doc_labels.tolist()
    elif not return_doc_labels and return_vocab:
        return mat, vocab.tolist()
    else:
        return mat


def ngrams(docs: Corpus, n: int, join=True, join_str=' ') -> Dict[str, Union[List[str], str]]:
    """
    Generate and return n-grams of length `n`.

    :param docs: list of string tokens or spaCy documents
    :param n: length of n-grams, must be >= 2
    :param join: if True, join generated n-grams by string `join_str`
    :param join_str: string used for joining
    :return: dict mapping document label to document n-grams; if `join` is True, the list contains strings of
             joined n-grams, otherwise the list contains lists of size `n` in turn containing the strings that
             make up the n-gram
    """
    if n < 2:
        raise ValueError('`n` must be at least 2')

    @parallelexec(collect_fn=merge_dicts_sorted)
    def _ngrams(chunk):
        return {lbl: token_ngrams(dtok, n, join=join, join_str=join_str) for lbl, dtok in chunk.items()}

    return _ngrams(_paralleltask(docs))


def kwic(docs: Corpus, search_tokens: Any, context_size: Union[int, OrdCollection] = 2,
         by_attr: Optional[str] = None, match_type: str = 'exact', ignore_case=False, glob_method: str = 'match',
         inverse=False, with_attr: Union[bool, OrdCollection] = False, as_tables=False, only_non_empty=False,
         glue: Optional[str] = None, highlight_keyword: Optional[str] = None) \
        -> Dict[str, Union[list, pd.DataFrame]]:
    """
    Perform *keyword-in-context (KWIC)* search for `search_tokens`. Uses similar search parameters as
    :func:`filter_tokens`. Returns results as dict with document label to KWIC results mapping. For
    tabular output, use :func:`kwic_table`. You may also use `as_tables` which gives dataframes per document with
    columns ``doc`` (document label), ``context`` (document-specific context number), ``position`` (token position in
    document), ``token`` and further token attributes if specified via `with_attr`.

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param context_size: either scalar int or tuple/list (left, right) -- number of surrounding words in keyword
                         context; if scalar, then it is a symmetric surrounding, otherwise can be asymmetric
    :param by_attr: if not None, this should be an attribute name; this attribute data will then be
                    used for matching instead of the tokens in `docs`
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param with_attr: also return document and token attributes along with each token; if True, returns all default
                      attributes and custom defined attributes; if list or tuple, returns attributes specified in this
                      sequence
    :param as_tables: return result as dataframe with "doc" (document label) and "context" (context ID per document) and
                      optionally "position" (original token position in the document) if tokens are not glued via `glue`
                      parameter
    :param only_non_empty: if True, only return non-empty result documents
    :param glue: if not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: if not None, this must be a string which is used to indicate the start and end of the
                              matched keyword
    :return: dict with `document label -> kwic for document` mapping or a dataframe, depending on `as_tables`
    """
    if isinstance(context_size, int):
        context_size = (context_size, context_size)
    elif not isinstance(context_size, (list, tuple)):
        raise ValueError('`context_size` must be integer or list/tuple')

    if len(context_size) != 2:
        raise ValueError('`context_size` must be list/tuple of length 2')

    if any(s < 0 for s in context_size) or all(s == 0 for s in context_size):
        raise ValueError('`context_size` must contain non-negative values and at least one strictly positive value')

    if glue is not None and with_attr:
        raise ValueError('when `glue` given, `with_attr` must be False')

    try:
        matchdata = _match_against(docs.spacydocs, by_attr, default=docs.custom_token_attrs_defaults.get(by_attr, None))
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    if with_attr:
        docs_w_attr = doc_tokens(docs, with_attr=with_attr, as_arrays=True)
        matchattr = by_attr or 'token'
        prepared = {}
        for lbl, matchagainst in matchdata.items():
            if matchattr != 'token':
                d = {k: v for k, v in docs_w_attr[lbl].items() if k != 'token'}
            else:
                d = docs_w_attr[lbl]
            
            prepared[lbl] = merge_dicts((d, {'_matchagainst': matchagainst}))
    else:
        prepared = {k: {'_matchagainst': v} for k, v in matchdata.items()}

    kwicres = _build_kwic_parallel(_paralleltask(docs, prepared), search_tokens=search_tokens,
                                   context_size=context_size, by_attr=by_attr,
                                   match_type=match_type, ignore_case=ignore_case,
                                   glob_method=glob_method, inverse=inverse, highlight_keyword=highlight_keyword,
                                   with_window_indices=as_tables, only_token_masks=False)

    return _finalize_kwic_results(kwicres, only_non_empty=only_non_empty, glue=glue, as_tables=as_tables,
                                  matchattr=by_attr or 'token', with_attr=bool(with_attr))


def kwic_table(docs: Corpus, search_tokens: Any, context_size: Union[int, OrdCollection] = 2,
               by_attr: Optional[str] = None, match_type: str = 'exact', ignore_case=False, glob_method: str = 'match',
               inverse=False, with_attr: Union[bool, OrdCollection] = False, glue: str = ' ',
               highlight_keyword: Optional[str] = '*'):
    """
    Perform *keyword-in-context (KWIC)* search for `search_tokens` and return result as dataframe.

    If a `glue` string is given, a "short" dataframe will be generated with columns ``doc`` (document label),
    ``context`` (document-specific context number) and ``token`` (KWIC result) or, if `by_attr` is set, the specified
    token attribute as last column name.

    If a `glue` is None, a "long" dataframe will be generated with columns ``doc`` (document label),
    ``context`` (document-specific context number), ``position`` (token position in document), ``token`` and further
    token attributes if specified via `with_attr`.

    Uses similar search parameters as :func:`filter_tokens`.

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param context_size: either scalar int or tuple/list (left, right) -- number of surrounding words in keyword
                         context; if scalar, then it is a symmetric surrounding, otherwise can be asymmetric
    :param by_attr: if not None, this should be an attribute name; this attribute data will then be
                    used for matching instead of the tokens in `docs`
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param with_attr: also return document and token attributes along with each token; if True, returns all default
                      attributes and custom defined attributes; if list or tuple, returns attributes specified in this
                      sequence
    :param glue: if not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: if not None, this must be a string which is used to indicate the start and end of the
                              matched keyword
    :return: dataframe with columns ``doc`` (document label), ``context`` (document-specific context number)
             and ``kwic`` (KWIC result)
    """

    kwicres = kwic(docs, search_tokens=search_tokens, context_size=context_size, by_attr=by_attr, match_type=match_type,
                   ignore_case=ignore_case, glob_method=glob_method, inverse=inverse, with_attr=with_attr,
                   as_tables=True, only_non_empty=True, glue=glue, highlight_keyword=highlight_keyword)

    if kwicres:
        kwic_df = pd.concat(kwicres.values(), axis=0)
        if glue is None:
            return kwic_df.sort_values(['doc', 'context', 'position'])
        else:
            return kwic_df.sort_values(['doc', 'context'])
    else:
        matchattr = by_attr or 'token'
        cols = ['doc', 'context']

        if glue is None:
            cols.append('position')
        cols.append(matchattr)

        if with_attr is True:
            cols.extend([a for a in Corpus.STD_TOKEN_ATTRS if a != by_attr])
        elif isinstance(with_attr, list):
            cols.extend([a for a in with_attr if a != by_attr])
        if isinstance(with_attr, str) and with_attr != by_attr:
            cols.append(with_attr)

        return pd.DataFrame(dict(zip(cols, [[] for _ in range(len(cols))])))


#%% Corpus I/O


def save_corpus_to_picklefile(docs: Corpus, picklefile: str):
    """
    Serialize Corpus `docs` and save to Python pickle file `picklefile`.

    .. seealso:: Use :func:`load_corpus_from_picklefile` to load the Corpus object from a pickle file.

    :param docs: a Corpus object
    :param picklefile: path to pickle file
    """
    pickle_data(serialize_corpus(docs, deepcopy_attrs=False), picklefile)


def load_corpus_from_picklefile(picklefile: str) -> Corpus:
    """
    Load and deserialize a stored Corpus object from the Python pickle file `picklefile`.

    .. seealso:: Use :func:`save_corpus_to_picklefile` to save a Corpus object to a pickle file.

    :param picklefile: path to pickle file
    :return: a Corpus object
    """
    return deserialize_corpus(unpickle_file(picklefile))


def load_corpus_from_tokens(tokens: Dict[str, Union[OrdCollection, Dict[str, List]]],
                            doc_attr_names: Optional[UnordStrCollection] = None,
                            token_attr_names: Optional[UnordStrCollection] = None,
                            **corpus_opt) -> Corpus:
    """
    Create a :class:`~tmtoolkit.corpus.Corpus` object from a dict of tokens (optionally along with document/token
    attributes) as may be returned from :func:`doc_tokens`.

    :param tokens: dict mapping document labels to tokens (optionally along with document/token attributes)
    :param doc_attr_names: names of document attributes
    :param token_attr_names: names of token attributes
    :param corpus_opt: arguments passed to :meth:`~tmtoolkit.corpus.Corpus.__init__`; shall not contain ``docs``
                       argument; at least ``language``, ``language_model`` or ``spacy_instance`` should be given
    :return: a Corpus object
    """
    if 'docs' in corpus_opt:
        raise ValueError('`docs` parameter is obsolete when initializing a Corpus with this function')

    corp = Corpus(**corpus_opt)
    _corpus_from_tokens(corp, tokens, doc_attr_names=doc_attr_names, token_attr_names=token_attr_names)

    return corp


def load_corpus_from_tokens_table(tokens: pd.DataFrame, **corpus_kwargs):
    """
    Create a :class:`~tmtoolkit.corpus.Corpus` object from a dataframe as may be returned from :func:`tokens_table`.

    :param tokens: a dataframe with tokens, optionally along with document/token attributes
    :param corpus_kwargs: arguments passed to :meth:`~tmtoolkit.corpus.Corpus.__init__`; shall not contain ``docs``
                          argument
    :return: a Corpus object
    """
    if 'docs' in corpus_kwargs:
        raise ValueError('`docs` parameter is obsolete when initializing a Corpus with this function')

    if {'doc', 'position', 'token'} & set(tokens.columns) != {'doc', 'position', 'token'}:
        raise ValueError('`tokens` must at least contain a columns "doc", "position" and "token"')

    tokens_dict = {}
    doc_attr_names = set()
    token_attr_names = set()
    for lbl in tokens['doc'].unique():      # TODO: could make this faster
        doc_df = tokens.loc[tokens['doc'] == lbl, :]

        colnames = doc_df.columns.tolist()
        colnames.pop(colnames.index('doc'))
        colnames.pop(colnames.index('position'))

        doc_attr_names.update(colnames[:colnames.index('token')])
        token_attr_names.update(colnames[colnames.index('token')+1:])

        tokens_dict[lbl] = doc_df.loc[:, colnames]

    return load_corpus_from_tokens(tokens_dict,
                                   doc_attr_names=list(doc_attr_names),
                                   token_attr_names=list(token_attr_names.difference(Corpus.STD_TOKEN_ATTRS)),
                                   **corpus_kwargs)


def serialize_corpus(docs: Corpus, deepcopy_attrs=True):
    """
    Serialize a Corpus object to a dict. The inverse operation is implemented in :func:`deserialize_corpus`.

    :param docs: a Corpus object
    :param deepcopy_attrs: apply *deep* copy to all attributes
    :return: Corpus data serialized as dict
    """
    return docs._serialize(deepcopy_attrs=deepcopy_attrs, store_nlp_instance_pointer=False)


def deserialize_corpus(serialized_corpus_data: dict):
    """
    Deserialize a Corpus object from a dict. The inverse operation is implemented in :func:`serialize_corpus`.

    :param serialized_corpus_data: Corpus data serialized as dict
    :return: a Corpus object
    """
    return Corpus._deserialize(serialized_corpus_data)


#%% Corpus functions that modify corpus data: document / token attribute handling


@corpus_func_copiable
def set_document_attr(docs: Corpus, /, attrname: str, data: Dict[str, Any], default=None, inplace=True):
    """
    Set a document attribute named `attrname` for documents in Corpus object `docs`. If the attribute
    already exists, it will be overwritten.

    .. seealso:: See `~tmtoolkit.corpus.remove_document_attr` to remove a document attribute.

    :param docs: a Corpus object
    :param attrname: name of the document attribute
    :param data: dict that maps document labels to document attribute value
    :param default: default document attribute value
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if attrname in docs.token_attrs + ['processed']:
        raise ValueError(f'attribute name "{attrname}" is already used as token attribute')

    if not Doc.has_extension(attrname):
        # setting default to None here always, since a default on `Doc` is not Corpus-specific but "global";
        # Corpus-specific default is set via `Corpus._doc_attrs_defaults`
        Doc.set_extension(attrname, default=None, force=True)

    for lbl, val in data.items():
        if lbl not in docs.spacydocs_ignore_filter.keys():
            raise ValueError(f'document "{lbl}" does not exist in Corpus object `docs`')

        setattr(docs.spacydocs_ignore_filter[lbl]._, attrname, val)

    if attrname not in {'label', 'mask'}:               # set Corpus-specific default
        docs._doc_attrs_defaults[attrname] = default


@corpus_func_copiable
def remove_document_attr(docs: Corpus, /, attrname: str, inplace=True):
    """
    Remove a document attribute with name `attrname` from the Corpus object `docs`.

    .. seealso:: See `~tmtoolkit.corpus.set_document_attr` to set a document attribute.

    :param docs: a Corpus object
    :param attrname: name of the document attribute
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if attrname not in docs.doc_attrs:
        raise ValueError(f'attribute name "{attrname}" is not registered as document attribute')

    for d in docs.spacydocs_ignore_filter.values():
        try:
            setattr(d._, attrname, None)
        except AttributeError: pass

    # note: we only remove the Corpus-specific custom document attribute, not the global SpaCy `Doc` attribute,
    # since this might still be in use with other Corpus objects
    del docs._doc_attrs_defaults[attrname]


@corpus_func_copiable
def set_token_attr(docs: Corpus, /, attrname: str, data: Dict[str, Any], default=None, per_token_occurrence=True,
                   inplace=True):
    """
    Set a token attribute named `attrname` for all tokens in all documents in Corpus object `docs`. If the attribute
    already exists, it will be overwritten.

    There are two ways of assigning token attributes which are determined by the argument `per_token_occurrence`. If
    `per_token_occurrence` is True, then `data` is a dict that maps token occurrences (or "word types") to attribute
    values, i.e. ``{'foo': True}`` will assign the attribute value ``True`` to every occurrence of the token ``"foo"``.
    If `per_token_occurrence` is False, then `data` is a dict that maps document labels to token attributes. In this
    case the token attributes must be a list, tuple or NumPy array with a length according to the number of (unmasked)
    tokens.

    .. seealso:: See `~tmtoolkit.corpus.remove_token_attr` to remove a token attribute.

    :param docs: a Corpus object
    :param attrname: name of the token attribute
    :param data: depends on `per_token_occurrence` –
    :param per_token_occurrence: determines how `data` is interpreted when assigning token attributes
    :param default: default token attribute value
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if attrname in docs.STD_TOKEN_ATTRS + ['mask', 'processed']:
        raise ValueError(f'cannot set attribute with protected name "{attrname}"')

    if attrname in docs.doc_attrs:
        raise ValueError(f'attribute name "{attrname}" is already used as document attribute')

    if per_token_occurrence:
        # convert data token string keys to token hashes
        data = {docs.nlp.vocab.strings[k]: v for k, v in data.items()}

    for lbl, d in docs.spacydocs.items():
        if per_token_occurrence:
            # match token occurrence with token's attribute value from `data`
            d.user_data[attrname] = np.array([data.get(hash, default) if mask else default
                                              for mask, hash in zip(d.user_data['mask'], d.user_data['processed'])])
        else:
            # set the token attributes for the whole document
            n_tok = len(d.user_data['mask'])
            n_filt = np.sum(d.user_data['mask'])

            if lbl not in data.keys():   # if not attribute data for this document, repeat default values
                attrvalues = np.repeat(default, n_tok)
            else:
                attrvalues = data[lbl]

            # convert to array
            if isinstance(attrvalues, (list, tuple)):
                attrvalues = np.array(attrvalues)
            elif not isinstance(attrvalues, np.ndarray):
                raise ValueError(f'token attributes for document "{lbl}" are neither tuple, list nor NumPy array')

            # if token attributes are only given for unmasked tokens, fill the gaps with default values
            if n_filt != n_tok and len(attrvalues) == n_filt:
                tmp = np.repeat(default, n_tok)
                if np.issubdtype(tmp.dtype, str):
                    tmp = tmp.astype('<U%d' % max(max(len(s) for s in attrvalues) if len(attrvalues) > 0 else 1, 1))
                tmp[d.user_data['mask']] = attrvalues
                attrvalues = tmp

            if len(attrvalues) != n_tok:
                raise ValueError(f'number of token attributes for document "{lbl}" do not match the number of tokens')

            # set the token attributes
            d.user_data[attrname] = attrvalues

    docs._token_attrs_defaults[attrname] = default


@corpus_func_copiable
def remove_token_attr(docs: Corpus, /, attrname: str, inplace=True):
    """
    Remove a token attribute with name `attrname` from the Corpus object `docs`.

    .. seealso:: See `~tmtoolkit.corpus.set_token_attr` to set a token attribute.

    :param docs: a Corpus object
    :param attrname: name of the token attribute
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if attrname not in docs.custom_token_attrs_defaults.keys():
        raise ValueError(f'attribute name "{attrname}" is not registered as custom token attribute')

    # remove respective user data in each document
    for d in docs.spacydocs_ignore_filter.values():
        try:
            del d.user_data[attrname]
        except KeyError: pass

    # remove custom token attributes entry
    del docs._token_attrs_defaults[attrname]


#%% Corpus functions that modify corpus data: token transformations


@corpus_func_copiable
@corpus_func_processes_tokens
def transform_tokens(docs: Corpus, /, func: Callable, inplace=True, **kwargs):
    """
    Transform tokens in all documents by applying function `func` to each document's tokens individually.

    :param docs: a Corpus object
    :param func: a function to apply to all documents' tokens; it must accept a single token string and vice-versa
                 return single token string
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :param kwargs: additional arguments passed to `func`
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    # get unique token types as hashes
    vocab = vocabulary(docs, tokens_as_hashes=True, force_unigrams=True)
    stringstore = docs.nlp.vocab.strings

    # construct two lists of same length:
    replace_from = []   # original token hash
    replace_to = []     # new token hash for transformed tokens
    for t_hash in vocab:    # iterate through token type hashes
        # get string representation for hash and transform it
        t_transformed = func(stringstore[t_hash], **kwargs)
        # get hash for transformed token type string
        t_hash_transformed = stringstore[t_transformed]
        # if hashes differ (i.e. transformation changed the string), record the hashes
        if t_hash != t_hash_transformed :
            stringstore.add(t_transformed)
            replace_from.append(t_hash)
            replace_to.append(t_hash_transformed)

    # replace the hashes in the documents
    if replace_from:
        for d in docs.spacydocs.values():
            if d.user_data['processed'].flags.writeable:
                arr_replace(d.user_data['processed'], replace_from, replace_to, inplace=True)
            else:
                d.user_data['processed'] = arr_replace(d.user_data['processed'], replace_from, replace_to)


def to_lowercase(docs: Corpus, /, inplace=True):
    """
    Convert all tokens to lower-case form.

    :param docs: a Corpus object
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return transform_tokens(docs, str.lower, inplace=inplace)


def to_uppercase(docs: Corpus, /, inplace=True):
    """
    Convert all tokens to upper-case form.

    :param docs: a Corpus object
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return transform_tokens(docs, str.upper, inplace=inplace)


def remove_chars(docs: Corpus, /, chars: Iterable[str], inplace=True):
    """
    Remove all characters listed in `chars` from all tokens.

    :param docs: a Corpus object
    :param chars: list of characters to remove; each element in the list should be a single character
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    del_chars = str.maketrans('', '', ''.join(chars))
    return transform_tokens(docs, lambda t: t.translate(del_chars), inplace=inplace)


def remove_punctuation(docs: Corpus, /, inplace=True):
    """
    Removes punctuation characters *in* tokens, i.e. ``['a', '.', 'f;o;o']`` becomes ``['a', '', 'foo']``.

    If you want to remove punctuation *tokens*, use :func:`~filter_clean_tokens`.

    :param docs: a Corpus object
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return remove_chars(docs, docs.punctuation, inplace=inplace)


def normalize_unicode(docs: Corpus, /, form: str = 'NFC', inplace=True):
    """
    Normalize unicode characters according to `form`.

    This function only *normalizes* unicode characters in the tokens of `docs` to the form
    specified by `form`. If you want to *simplify* the characters, i.e. remove diacritics,
    underlines and other marks, use :func:`~simplify_unicode` instead.

    :param docs: a Corpus object
    :param form: normal form (see https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return transform_tokens(docs, lambda t: unicodedata.normalize(form, t), inplace=inplace)


def simplify_unicode(docs: Corpus, /, method: str = 'icu', inplace=True):
    """
    *Simplify* unicode characters in the tokens of `docs`, i.e. remove diacritics, underlines and
    other marks. Requires `PyICU <https://pypi.org/project/PyICU/>`_ to be installed when using
    ``method="icu"``.

    :param docs: a Corpus object
    :param method: either ``"icu"`` which uses `PyICU <https://pypi.org/project/PyICU/>`_ for "proper"
                   simplification or ``"ascii"`` which tries to encode the characters as ASCII; the latter
                   is not recommended and will simply dismiss any characters that cannot be converted
                   to ASCII after decomposition
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """

    method = method.lower()
    if method == 'icu':
        try:
            from icu import UnicodeString, Transliterator, UTransDirection
        except ImportError:
            raise RuntimeError('package PyICU (https://pypi.org/project/PyICU/) must be installed to use this method')

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
@corpus_func_processes_tokens
def lemmatize(docs: Corpus, /, inplace=True):
    """
    Lemmatize tokens, i.e. set the lemmata as tokens so that all further processing will happen
    using the lemmatized tokens.

    :param docs: a Corpus object
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    for d in docs.spacydocs.values():
        d.user_data['processed'] = np.fromiter((t.lemma for t in d), dtype='uint64', count=len(d))


@corpus_func_copiable
@corpus_func_processes_tokens
def join_collocations_by_patterns(docs: Corpus, /, patterns: OrdStrCollection, glue: str = '_',
                                  match_type: str = 'exact', ignore_case=False, glob_method: str = 'match',
                                  return_joint_tokens=False, inplace=True):
    """
    Match *N* *subsequent* tokens to the *N* patterns in `patterns` using match options like in :func:`filter_tokens`.
    Join the matched tokens by glue string `glue` and mask the original tokens that this new joint token was
    generated from.

    .. warning:: For each of the joint subsequent tokens, only the token attributes of the first token in the sequence
                 will be retained. All further tokens will be masked. For example: In a document with tokens
                 ``["a", "hello", "world", "example"]`` where we join ``"hello", "world"``, the resulting document will
                 be ``["a", "hello_world", "example"]`` and only the token attributes (lemma, POS tag, etc. and custom
                 attributes) for ``"hello"`` will be retained and assigned to "hello_world".

    :param docs: a Corpus object
    :param patterns: a sequence of search patterns as excepted by `filter_tokens`
    :param glue: string used for joining the matched subsequent tokens
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is ``'glob'``, use either ``'search'`` or ``'match'`` as glob method
                        (has similar implications as Python's ``re.search`` vs. ``re.match``)
    :param return_joint_tokens: also return set of joint collocations
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object; if
             `return_joint_tokens` is True, return set of joint collocations instead (if `inplace` is True) or
             additionally in tuple ``(modified Corpus copy, set of joint collocations)`` (if `inplace` is False)
    """
    if not isinstance(patterns, (list, tuple)) or len(patterns) < 2:
        raise ValueError('`patterns` must be a list or tuple containing at least two elements')

    @parallelexec(merge_dicts)
    def _join_colloc(chunk: Dict[str, List[str]]):
        res = {}
        for lbl, tok in chunk.items():
            # get the subsequent matches as boolean mask arrays
            matches = token_match_subsequent(patterns, tok, match_type=match_type, ignore_case=ignore_case,
                                             glob_method=glob_method)

            # join the matched subsequent tokens; `return_mask=True` makes sure that we only get the newly
            # generated joint tokens together with an array to mask all but the first token of the subsequent tokens
            res[lbl] = token_join_subsequent(tok, matches, glue=glue, return_mask=True)

        return res

    joint_colloc = _join_colloc(_paralleltask(docs))
    joint_tokens = _apply_collocations(docs, joint_colloc,
                                       tokens_as_hashes=False, glue=None, return_joint_tokens=return_joint_tokens)

    if return_joint_tokens:
        return joint_tokens


@corpus_func_copiable
@corpus_func_processes_tokens
def join_collocations_by_statistic(docs: Corpus, /, threshold: float, glue: str = '_', min_count: int = 1,
                                   embed_tokens_min_docfreq: Optional[Union[int, float]] = None,
                                   embed_tokens_set: Optional[UnordCollection] = None,
                                   statistic: Callable = npmi, return_joint_tokens=False, inplace=True,
                                   **statistic_kwargs):
    """
    Join subsequent tokens by token collocation statistic as can be computed by :func:`corpus_collocations`.

    :param docs: a Corpus object
    :param threshold: minimum statistic value for a collocation to enter the results
    :param glue: string used for joining the subsequent tokens
    :param min_count: ignore collocations with number of occurrences below this threshold
    :param embed_tokens_min_docfreq: dynamically generate the set of ``embed_tokens`` used when calling
                                     :func:`~tmtoolkit.tokenseq.token_collocations` by using a minimum document
                                     frequency (see :func:`~doc_frequencies`); if this is an integer, it is used as
                                     absolute count, if it is a float, it is used as proportion
    :param embed_tokens_set: tokens that, if occurring inside an n-gram, are not counted; see :func:`token_ngrams`
    :param statistic: function to calculate the statistic measure from the token counts; use one of the
                      ``[n]pmi[2,3]_from_counts`` functions provided in the :mod:`~tmtoolkit.tokenseq` module or provide
                      your own function which must accept parameters ``n_x, n_y, n_xy, n_total``; see
                      :func:`~tmtoolkit.tokenseq.pmi_from_counts` and :func:`~tmtoolkit.tokenseq.pmi`
                      for more information
    :param return_joint_tokens: also return set of joint collocations
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :param statistic_kwargs: additional arguments passed to `statistic` function
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object; if
             `return_joint_tokens` is True, return set of joint collocations instead (if `inplace` is True) or
             additionally in tuple ``(modified Corpus copy, set of joint collocations)`` (if `inplace` is False)
    """
    if not isinstance(glue, str):
        raise ValueError('`glue` must be a string')

    @parallelexec(merge_dicts)
    def _join_colloc(chunk: Dict[str, List[str]], colloc):
        res = {}
        for lbl, tok in chunk.items():
            # get the subsequent matches of the collocation token hashes as boolean mask arrays
            matches = []
            for hashes in colloc:
                matches.extend(token_match_subsequent(hashes, tok, match_type='exact'))

            # join the matched subsequent tokens; `return_mask=True` makes sure that we only get the newly
            # generated joint tokens together with an array to mask all but the first token of the subsequent tokens
            # `glue=None` makes sure that the token hashes are not joint
            res[lbl] = token_join_subsequent(tok, matches, glue=None,  tokens_dtype='uint64', return_mask=True)

        return res

    # get tokens as hashes
    tok = doc_tokens(docs, tokens_as_hashes=True)
    tok_flat = [flatten_list(tok.values())]   # TODO: use sentences
    vocab_counts = vocabulary_counts(docs, tokens_as_hashes=True)

    # # generate ``embed_tokens`` set as used in :func:`~tmtookit.tokenseq.token_collocations`
    embed_tokens = _create_embed_tokens_for_collocations(docs, embed_tokens_min_docfreq, embed_tokens_set)

    # identify collocations
    colloc = token_collocations(tok_flat, threshold=threshold, min_count=min_count, embed_tokens=embed_tokens,
                                vocab_counts=vocab_counts, statistic=statistic, return_statistic=False,
                                rank=None, **statistic_kwargs)

    # join collocations
    joint_colloc = _join_colloc(_paralleltask(docs, tok), colloc=colloc)
    joint_tokens = _apply_collocations(docs, joint_colloc,
                                       tokens_as_hashes=True, glue=glue, return_joint_tokens=return_joint_tokens)

    if return_joint_tokens:
        return joint_tokens


#%% Corpus functions that modify corpus data: filtering / KWIC

@corpus_func_copiable
def reset_filter(docs: Corpus, /, which: str = 'all', inplace=True):
    """
    Reset the token- and/or document-level filters on Corpus `docs`.

    :param docs: a Corpus object
    :param which: either ``'documents'`` which resets the document-level filter,
                  or ``'tokens'`` which resets the token-level filter, or ``'all'`` (default)
                  which resets both
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if which in {'all', 'documents'}:
        for d in docs.spacydocs_ignore_filter.values():
            d._.mask = True

    if which in {'all', 'tokens'}:
        for d in docs.spacydocs.values():
            d.user_data['mask'] = np.repeat(True, len(d.user_data['mask']))
        docs._tokens_masked = False


@corpus_func_copiable
@corpus_func_filters_tokens
def filter_tokens_by_mask(docs: Corpus, /, mask: Dict[str, Union[List[bool], np.ndarray]],
                          replace=False, inverse=False, inplace=True):
    """
    Filter tokens according to a boolean mask specified by `mask`.

    .. seealso:: :func:`remove_tokens_by_mask`

    :param docs: a Corpus object
    :param mask: dict mapping document label to boolean list or NumPy array where ``False`` means "masked" and
                 ``True`` means "unmasked" for the respective token; if `replace` is True, the length of the mask
                 must equal the number of *all* tokens (masked and unmasked) in the document; if `replace` is False, the
                 length of the mask must equal the number of *unmasked* tokens in the document since only for those the
                 new mask is set
    :param replace: if True, replace the whole document mask array, otherwise set the mask only for the items of the
                    unmasked tokens
    :param inverse: inverse the truth values in the mask arrays
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """

    for lbl, m in mask.items():
        if lbl not in docs.keys():
            raise ValueError(f'document "{lbl}" does not exist in Corpus object `docs` or is masked - '
                             f'cannot set token mask')

        if not isinstance(m, np.ndarray):
            m = np.array(m, dtype=bool)

        if inverse:
            m = ~m

        d = docs.spacydocs[lbl]

        if replace:
            if len(d.user_data['mask']) != len(m):
                raise ValueError(f'length of provided mask for document "{lbl}" does not match the existing mask\'s '
                                 f'length')

            # replace with the whole new mask
            d.user_data['mask'] = m
        else:
            if np.sum(d.user_data['mask']) != len(m):
                raise ValueError(f'length of provided mask for document "{lbl}" does not match the number of '
                                 f'unfiltered tokens')

            # set the new mask items
            d.user_data['mask'] = _ensure_writable_array(d.user_data['mask'])
            d.user_data['mask'][d.user_data['mask']] = m


def remove_tokens_by_mask(docs: Corpus, /, mask: Dict[str, Union[List[bool], np.ndarray]],
                          replace=False, inplace=True):
    """
    Remove tokens according to a boolean mask specified by `mask`.

    .. seealso:: :func:`filter_tokens_by_mask`

    :param docs: a Corpus object
    :param mask: dict mapping document label to boolean list or NumPy array where ``True`` means "masked" and
                 ``False`` means "unmasked" for the respective token; if `replace` is True, the length of the mask
                 must equal the number of *all* tokens (masked and unmasked) in the document; if `replace` is False, the
                 length of the mask must equal the number of *unmasked* tokens in the document since only for those the
                 new mask is set
    :param replace: if True, replace the whole document mask array, otherwise set the mask only for the items of the
                    unmasked tokens
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_tokens_by_mask(docs, mask=mask, inverse=True, replace=replace, inplace=inplace)


def filter_tokens(docs: Corpus, /, search_tokens: Any, by_attr: Optional[str] = None,
                  match_type: str = 'exact', ignore_case=False,
                  glob_method: str = 'match', inverse=False, inplace=True):
    """
    Filter tokens according to search pattern(s) `search_tokens` and several matching options. Only those tokens
    are retained that match the search criteria unless you set ``inverse=True``, which will *remove* all tokens
    that match the search criteria (which is the same as calling :func:`remove_tokens`).

    .. note:: Tokens will only be *masked* (hidden) with a filter when using this function. You can reset the filter
              using :func:`reset_filter` or permanently remove masked tokens using :func:`compact`.

    .. seealso:: :func:`remove_tokens` and :func:`~tmtoolkit.preprocess.token_match`

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param by_attr: if not None, this should be an attribute name; this attribute data will then be
                    used for matching instead of the tokens in `docs`
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is ``'glob'``, use either ``'search'`` or ``'match'`` as glob method
                        (has similar implications as Python's ``re.search`` vs. ``re.match``)
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    _check_filter_args(match_type=match_type, glob_method=glob_method)

    @parallelexec(collect_fn=merge_dicts)
    def _filter_tokens(chunk):
        return _token_pattern_matches(chunk, search_tokens, match_type=match_type,
                                      ignore_case=ignore_case, glob_method=glob_method)

    try:
        matchdata = _match_against(docs.spacydocs, by_attr, default=docs.custom_token_attrs_defaults.get(by_attr, None))
        masks = _filter_tokens(_paralleltask(docs, matchdata))
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    return filter_tokens_by_mask(docs, masks, inverse=inverse)


def remove_tokens(docs: Corpus, /, search_tokens: Any, by_attr: Optional[str] = None,
                  match_type: str = 'exact', ignore_case=False,
                  glob_method: str = 'match', inplace=True):
    """
    This is a shortcut for the :func:`filter_tokens` method with ``inverse=True``, i.e. *remove* all tokens that match
    the search criteria).

    .. note:: Tokens will only be *masked* (hidden) with a filter when using this function. You can reset the filter
          using :func:`reset_filter` or permanently remove masked tokens using :func:`compact`.

    .. seealso:: :func:`filter_tokens` and :func:`~tmtoolkit.preprocess.token_match`

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param by_attr: if not None, this should be an attribute name; this attribute data will then be
                    used for matching instead of the tokens in `docs`
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is ``'glob'``, use either ``'search'`` or ``'match'`` as glob method
                        (has similar implications as Python's ``re.search`` vs. ``re.match``)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_tokens(docs, search_tokens=search_tokens, match_type=match_type,
                         ignore_case=ignore_case, glob_method=glob_method,
                         by_attr=by_attr, inverse=True)


def filter_for_pos(docs: Corpus, /, search_tokens: Any, simplify_pos=True, tagset:str = 'ud',
                   inverse=False, inplace=True):
    @parallelexec(collect_fn=merge_dicts)
    def _filter_pos(chunk):
        if simplify_pos:
            chunk = {lbl: list(map(lambda x: simplified_pos(x, tagset=tagset), tok_pos))
                     for lbl, tok_pos in chunk.items()}

        return _token_pattern_matches(chunk, search_tokens)

    matchdata = _match_against(docs.spacydocs, 'pos')
    masks = _filter_pos(_paralleltask(docs, matchdata))

    return filter_tokens_by_mask(docs, masks, inverse=inverse)


def filter_tokens_by_doc_frequency(docs: Corpus, /, which: str, df_threshold: Union[int, float], proportions=False,
                                   return_filtered_tokens=False, inverse=False, inplace=True):
    """
    Filter tokens according to their document frequency.

    :param docs: a Corpus object
    :param which: which threshold comparison to use: either ``'common'``, ``'>'``, ``'>='`` which means that tokens
                  with higher document freq. than (or equal to) `df_threshold` will be kept;
                  or ``'uncommon'``, ``'<'``, ``'<='`` which means that tokens with lower document freq. than
                  (or equal to) `df_threshold` will be kept
    :param df_threshold: document frequency threshold value
    :param proportions: if True, document frequency threshold is given in proportions rather than absolute counts
    :param return_filtered_tokens: if True, additionally return set of filtered token types
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: depending on `return_filtered_tokens` and `inplace`: if both are True, returns only filtered token types;
             if `return_filtered_tokens` is True and `inplace` is False, returns tuple with modified copy of `docs` and
             filtered token types; if `return_filtered_tokens` is False returns either original Corpus object `docs` or
             a modified copy of it
    """
    if proportions:
        if not 0 <= df_threshold <= 1:
            raise ValueError('`df_threshold` must be in range [0, 1]')
    else:
        n_docs = len(docs)
        if not 0 <= df_threshold <= n_docs:
            raise ValueError(f'`df_threshold` must be in range [0, {n_docs}]')

    comp = _comparison_operator_from_str(which, common_alias=True)

    toks = doc_tokens(docs)
    doc_freqs = doc_frequencies(docs, proportions=proportions)
    mask = {lbl: [comp(doc_freqs[t], df_threshold) for t in dtok] for lbl, dtok in toks.items()}

    filt_tok = set()
    if return_filtered_tokens:
        filt_tok = set(t for t, f in doc_freqs.items() if comp(f, df_threshold))

    res = filter_tokens_by_mask(docs, mask=mask, inverse=inverse, inplace=inplace)
    if return_filtered_tokens:
        if inplace:
            return filt_tok
        else:
            return res, filt_tok
    else:
        return res


def remove_common_tokens(docs: Corpus, /, df_threshold: Union[int, float] = 0.95, proportions=True, inplace=True):
    """
    Shortcut for :func:`filter_tokens_by_doc_frequency` for removing tokens *above* a certain  document frequency.

    :param docs: a Corpus object
    :param df_threshold: document frequency threshold value
    :param proportions: if True, document frequency threshold is given in proportions rather than absolute counts
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_tokens_by_doc_frequency(docs, which='common', df_threshold=df_threshold, proportions=proportions,
                                          inverse=True, inplace=inplace)


def remove_uncommon_tokens(docs: Corpus, /, df_threshold: Union[int, float] = 0.05, proportions=True, inplace=True):
    """
    Shortcut for :func:`filter_tokens_by_doc_frequency` for removing tokens *below* a certain  document frequency.

    :param docs: a Corpus object
    :param df_threshold: document frequency threshold value
    :param proportions: if True, document frequency threshold is given in proportions rather than absolute counts
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_tokens_by_doc_frequency(docs, which='uncommon', df_threshold=df_threshold, proportions=proportions,
                                          inverse=True, inplace=inplace)


@corpus_func_copiable
def filter_documents(docs: Corpus, /, search_tokens: Any, by_attr: Optional[str] = None,
                     matches_threshold: int = 1, match_type: str = 'exact', ignore_case=False,
                     glob_method: str = 'match', inverse_result=False, inverse_matches=False, inplace=True):
    """
    This function is similar to :func:`filter_tokens` but applies at document level. For each document, the number of
    matches is counted. If it is at least `matches_threshold` the document is retained, otherwise it is removed.
    If `inverse_result` is True, then documents that meet the threshold are *masked*.

    .. note:: Documents will only be *masked* (hidden) with a filter when using this function. You can reset the filter
              using :func:`reset_filter` or permanently remove masked documents using :func:`compact`.

    .. seealso:: :func:`remove_documents`

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param by_attr: if not None, this should be an attribute name; this attribute data will then be
                    used for matching instead of the tokens in `docs`
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is ``'glob'``, use either ``'search'`` or ``'match'`` as glob method
                        (has similar implications as Python's ``re.search`` vs. ``re.match``)
    :param inverse_result: inverse the threshold comparison result
    :param inverse_matches: inverse the match results for filtering
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    _check_filter_args(match_type=match_type, glob_method=glob_method)

    @parallelexec(collect_fn=merge_sets)
    def _filter_documents(chunk):
        matches = _token_pattern_matches(chunk, search_tokens, match_type=match_type,
                                         ignore_case=ignore_case, glob_method=glob_method)
        rm_docs = set()
        for lbl, m in matches.items():
            if inverse_matches:
                m = ~m

            thresh_met = np.sum(m) >= matches_threshold
            if inverse_result:
                thresh_met = not thresh_met
            if not thresh_met:
                rm_docs.add(lbl)

        return rm_docs

    try:
        matchdata = _match_against(docs.spacydocs, by_attr, default=docs.custom_token_attrs_defaults.get(by_attr, None))
        remove = _filter_documents(_paralleltask(docs, matchdata))
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    return filter_documents_by_mask(docs, mask=dict(zip(remove, [False] * len(remove))))


def remove_documents(docs: Corpus, /, search_tokens: Any, by_attr: Optional[str] = None,
                     matches_threshold: int = 1, match_type: str = 'exact', ignore_case=False,
                     glob_method: str = 'match', inverse_matches=False, inplace=True):
    """
    This is a shortcut for the :func:`filter_documents` function with ``inverse_result=True``, i.e. *remove* all
    documents that meet the token matching threshold.

    .. note:: Documents will only be *masked* (hidden) with a filter when using this function. You can reset the filter
              using :func:`reset_filter` or permanently remove masked documents using :func:`compact`.

    .. seealso:: :func:`filter_documents`

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param by_attr: if not None, this should be an attribute name; this attribute data will then be
                    used for matching instead of the tokens in `docs`
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is ``'glob'``, use either ``'search'`` or ``'match'`` as glob method
                        (has similar implications as Python's ``re.search`` vs. ``re.match``)
    :param inverse_matches: inverse the match results for filtering
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_documents(docs, search_tokens=search_tokens, by_attr=by_attr, matches_threshold=matches_threshold,
                            match_type=match_type, ignore_case=ignore_case, glob_method=glob_method,
                            inverse_matches=inverse_matches, inverse_result=True)


def filter_documents_by_mask(docs: Corpus, /, mask: Dict[str, List[bool]], inverse=False, inplace=True):
    """
    Filter documents by setting a mask.

    .. seealso:: :func:`remove_documents_by_mask`

    :param docs: a Corpus object
    :param mask: dict that maps document labels to document attribute value
    :param inverse: inverse the mask
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if inverse:
        mask = {lbl: list(~np.array(m)) for lbl, m in mask.items()}

    return set_document_attr(docs, 'mask', data=mask)


def remove_documents_by_mask(docs: Corpus, /, mask: Dict[str, List[bool]], inplace=True):
    """
    This is a shortcut for the :func:`filter_documents_by_mask` function with ``inverse_result=True``, i.e. *remove* all
    documents where the mask is set to True.

    .. seealso:: :func:`filter_documents_by_mask`

    :param docs: a Corpus object
    :param mask: dict that maps document labels to document attribute value
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_documents_by_mask(docs, mask=mask, inverse=True, inplace=inplace)


def filter_documents_by_docattr(docs: Corpus, /, search_tokens: Any, by_attr: str,
                                match_type: str = 'exact', ignore_case=False, glob_method: str = 'match',
                                inverse=False, inplace=True):
    """
    Filter documents by a document attribute `by_attr`.

    .. seealso:: :func:`remove_documents_by_docattr`

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param by_attr: document attribute name used for filtering
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is ``'glob'``, use either ``'search'`` or ``'match'`` as glob method
                        (has similar implications as Python's ``re.search`` vs. ``re.match``)
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    _check_filter_args(match_type=match_type, glob_method=glob_method)

    if by_attr == 'label':
        attr_values = doc_labels(docs)
    else:
        if by_attr not in docs.doc_attrs_defaults:
            raise ValueError(f'document attribute "{by_attr}" not defined in Corpus `docs`')

        default = docs.doc_attrs_defaults[by_attr]
        attr_values = []
        for d in docs.spacydocs.values():
            v = getattr(d._, by_attr)
            if v is None:   # can't use default arg (third arg) in `getattr` b/c Doc extension *always* returns
                            # a value; it will be None by Doc extension default
                v = default
            attr_values.append(v)

    matches = token_match_multi_pattern(search_tokens, attr_values, match_type=match_type,
                                        ignore_case=ignore_case, glob_method=glob_method)
    return filter_documents_by_mask(docs, mask=dict(zip(docs.keys(), matches)), inverse=inverse, inplace=inplace)


def remove_documents_by_docattr(docs: Corpus, /, search_tokens: Any, by_attr: str,
                                match_type: str = 'exact', ignore_case=False, glob_method: str = 'match', inplace=True):
    """
    This is a shortcut for the :func:`filter_documents_by_docattr` function with ``inverse=True``, i.e. *remove* all
    documents that meet the document attribute matching criteria.

    .. seealso:: :func:`filter_documents_by_docattr`

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param by_attr: document attribute name used for filtering
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is ``'glob'``, use either ``'search'`` or ``'match'`` as glob method
                        (has similar implications as Python's ``re.search`` vs. ``re.match``)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_documents_by_docattr(docs, search_tokens=search_tokens, by_attr=by_attr, match_type=match_type,
                                       ignore_case=ignore_case, glob_method=glob_method, inverse=True, inplace=inplace)


def filter_documents_by_label(docs: Corpus, /, search_tokens: Any, match_type: str = 'exact',
                             ignore_case=False, glob_method: str = 'match', inverse=False, inplace=True):
    """
    Filter documents by document label.

    .. seealso:: :func:`remove_documents_by_label`

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is ``'glob'``, use either ``'search'`` or ``'match'`` as glob method
                        (has similar implications as Python's ``re.search`` vs. ``re.match``)
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_documents_by_docattr(docs, search_tokens=search_tokens, by_attr='label', match_type=match_type,
                                       ignore_case=ignore_case, glob_method=glob_method, inverse=inverse,
                                       inplace=inplace)


def remove_documents_by_label(docs: Corpus, /, search_tokens: Any, match_type: str = 'exact',
                              ignore_case=False, glob_method: str = 'match', inplace=True):
    """
    Shortcut for :func:`filter_documents_by_label` with ``inverse=True``, i.e. *remove* all
    documents that meet the document label matching criteria.

    .. seealso:: :func:`filter_documents_by_label`

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is ``'glob'``, use either ``'search'`` or ``'match'`` as glob method
                        (has similar implications as Python's ``re.search`` vs. ``re.match``)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_documents_by_label(docs, search_tokens=search_tokens, match_type=match_type,
                                     ignore_case=ignore_case, glob_method=glob_method, inverse=True,
                                     inplace=inplace)


def filter_documents_by_length(docs: Corpus, /, relation: str, threshold: int, inverse=False, inplace=True):
    """
    Filter documents in `docs` by length, i.e. number of tokens.

    .. seealso:: :func:`remove_documents_by_length`

    :param docs: a Corpus object
    :param relation: comparison operator as string; must be one of ``'<', '<=', '==', '>=', '>'``
    :param threshold: document length threshold in number of documents
    :param inverse: inverse the mask
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if threshold < 0:
        raise ValueError("`threshold` cannot be negative")

    comp = _comparison_operator_from_str(relation, equal=True, whicharg='relation')
    mask = {lbl: comp(n, threshold) for lbl, n in doc_lengths(docs).items()}

    return filter_documents_by_mask(docs, mask=mask, inverse=inverse, inplace=inplace)


def remove_documents_by_length(docs: Corpus, /, relation: str, threshold: int, inplace=True):
    """
    Shortcut for :func:`filter_documents_by_length` with ``inverse=True``, i.e. *remove* all
    documents that meet the length criterion.

    .. seealso:: :func:`filter_documents_by_length`

    :param docs: a Corpus object
    :param relation: comparison operator as string; must be one of ``'<', '<=', '==', '>=', '>'``
    :param threshold: document length threshold in number of documents
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_documents_by_length(docs, relation=relation, threshold=threshold, inverse=True, inplace=inplace)


@corpus_func_copiable
@corpus_func_filters_tokens
def filter_clean_tokens(docs: Corpus, /,
                        remove_punct: bool = True,
                        remove_stopwords: Union[bool, Iterable[str]] = True,
                        remove_empty: bool = True,
                        remove_shorter_than: Optional[int] = None,
                        remove_longer_than: Optional[int] = None,
                        remove_numbers: bool = False,
                        inplace=True):
    """
    Filter tokens in `docs` to retain only a certain, configurable subset of token.

    :param docs: a Corpus object
    :param remove_punct: remove all tokens that are considered to be punctuation (``"."``, ``","``, ``";"`` etc.)
                         according to the ``is_punct`` attribute of the
                         `SpaCy Token <https://spacy.io/api/token#attributes>`_
    :param remove_stopwords: remove all tokens that are considered to be stopwords; if True, remove tokens according to
                             the ``is_stop`` attribute of the `SpaCy Token <https://spacy.io/api/token#attributes>`_;
                             if `remove_stopwords` is a set/tuple/list it defines the stopword list
    :param remove_empty: remove all empty string ``""`` tokens
    :param remove_shorter_than: remove all tokens shorter than this length
    :param remove_longer_than: remove all tokens longer than this length
    :param remove_numbers: remove all tokens that are "numeric" according to the ``like_num`` attribute of the
                           `SpaCy Token <https://spacy.io/api/token#attributes>`_
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if not remove_punct and not remove_stopwords and not remove_empty and \
            remove_shorter_than is None and remove_longer_than is None and \
            not remove_numbers:
        # nothing to do
        return

    # check parameters
    if remove_shorter_than is not None and remove_shorter_than < 0:
        raise ValueError('`remove_shorter_than` must be >= 0')
    if remove_longer_than is not None and remove_longer_than < 0:
        raise ValueError('`remove_longer_than` must be >= 0')

    # function for parallel filtering: accepts a chunk of documents as dict
    # doc. label -> doc. data and returns a dict doc. label -> doc. filter mask
    @parallelexec(collect_fn=merge_dicts)
    def _filter_clean_tokens(chunk, tokens_to_remove):
        # the "doc masks" list holds a boolean array for each document where
        # `True` signals a token to be kept, `False` a token to be removed
        doc_masks = [np.repeat(True, doc['doc_length']) for doc in chunk.values()]

        if remove_punct:
            doc_masks = [mask & ~doc['is_punct'][doc['mask']].astype(bool)
                         for mask, doc in zip(doc_masks, chunk.values())]

        if remove_shorter_than is not None:
            doc_masks = [mask & (n >= remove_shorter_than)
                         for mask, n in zip(doc_masks, (doc['token_lengths'] for doc in chunk.values()))]

        if remove_longer_than is not None:
            doc_masks = [mask & (n <= remove_longer_than)
                         for mask, n in zip(doc_masks, (doc['token_lengths'] for doc in chunk.values()))]

        if remove_numbers:
            doc_masks = [mask & ~doc['like_num'][doc['mask']].astype(bool)
                         for mask, doc in zip(doc_masks, chunk.values())]

        if remove_stopwords is True:
            doc_masks = [mask & ~doc['is_stop'][doc['mask']].astype(bool)
                         for mask, doc in zip(doc_masks, chunk.values())]

        if tokens_to_remove:
            doc_masks = [mask & np.array([t not in tokens_to_remove for t in doc], dtype=bool)
                         for mask, doc in zip(doc_masks, (doc['tokens'] for doc in chunk.values()))]

        return dict(zip(chunk.keys(), doc_masks))

    # add stopwords
    if isinstance(remove_stopwords, (list, tuple, set)):
        tokens_to_remove = remove_stopwords
    else:
        tokens_to_remove = []

    # data preparation for parallel processing: create a dict `docs_data` with
    # doc. label -> doc. data that contains all necessary information for filtering
    # the document, depending on the filtering options
    docs_data = {}
    lengths = doc_lengths(docs)

    if tokens_to_remove:
        tokens = doc_tokens(docs, force_unigrams=True)
    else:
        tokens = None

    if remove_empty and not remove_shorter_than:
        remove_shorter_than = 1

    if remove_shorter_than is not None or remove_longer_than is not None:
        token_lengths = doc_token_lengths(docs)
    else:
        token_lengths = None

    for lbl, d in docs.spacydocs.items():
        d_data = {}
        d_data['doc_length'] = lengths[lbl]
        d_data['mask'] = d.user_data['mask']

        if remove_punct:
            d_data['is_punct'] = d.to_array('is_punct')

        if remove_numbers:
            d_data['like_num'] = d.to_array('like_num')

        if remove_stopwords is True:
            d_data['is_stop'] = d.to_array('is_stop')

        if token_lengths is not None:
            d_data['token_lengths'] = np.array(token_lengths[lbl], dtype='uint16')

        if tokens is not None:
            d_data['tokens'] = tokens[lbl]

        docs_data[lbl] = d_data

    # run filtering in parallel
    new_masks = _filter_clean_tokens(_paralleltask(docs, docs_data),
                                     tokens_to_remove=set(tokens_to_remove))

    # apply the mask
    _apply_matches_array(docs, new_masks)


def filter_tokens_with_kwic(docs: Corpus, /, search_tokens: Any, context_size: Union[int, OrdCollection] = 2,
                            by_attr: Optional[str] = None, match_type: str = 'exact', ignore_case=False,
                            glob_method: str = 'match', inverse=False, inplace=True):
    """
    Filter tokens in `docs` according to Keywords-in-Context (KWIC) context window of size `context_size` around
    `search_tokens`. Uses similar search parameters as :func:`filter_tokens`. Use :func:`kwic` or :func:`kwic_table`
    if you want to retrieve KWIC results without filtering the corpus.

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param context_size: either scalar int or tuple/list (left, right) -- number of surrounding words in keyword
                         context; if scalar, then it is a symmetric surrounding, otherwise can be asymmetric
    :param by_attr: if not None, this should be an attribute name; this attribute data will then be
                    used for matching instead of the tokens in `docs`
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if isinstance(context_size, int):
        context_size = (context_size, context_size)
    elif not isinstance(context_size, (list, tuple)):
        raise ValueError('`context_size` must be integer or list/tuple')

    if len(context_size) != 2:
        raise ValueError('`context_size` must be list/tuple of length 2')

    try:
        matchdata = _match_against(docs.spacydocs, by_attr, default=docs.custom_token_attrs_defaults.get(by_attr, None))
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    matches = _build_kwic_parallel(_paralleltask(docs, matchdata), search_tokens=search_tokens,
                                   context_size=context_size, by_attr=by_attr,
                                   match_type=match_type, ignore_case=ignore_case,
                                   glob_method=glob_method, inverse=inverse, only_token_masks=True)
    return filter_tokens_by_mask(docs, matches)


@corpus_func_copiable
def compact(docs: Corpus, /, which: str = 'all', inplace=True):
    """
    Set processed tokens as "original" document tokens and permanently apply the current filters to `doc` by removing
    the masked tokens and/or documents. Frees memory.

    :param docs: a Corpus object
    :param which: specify to permanently apply filters to tokens (``which = 'tokens'``),
                  documents (``which = 'documents'``) or both  (``which = 'all'``),
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    tok = doc_tokens(docs, with_attr=True, force_unigrams=True,
                     apply_token_filter=which in {'all', 'tokens'},
                     apply_document_filter=which in {'all', 'documents'})
    _corpus_from_tokens(docs, tok,
                        doc_attr_names=docs.doc_attrs,
                        token_attr_names=list(docs.custom_token_attrs_defaults.keys()))   # re-create spacy docs
    if which != 'documents':
        docs._tokens_masked = False
    docs._tokens_processed = False


#%% Corpus functions that modify corpus data: other


@corpus_func_copiable
def ngramify(docs: Corpus, /, n: int, join_str=' ', inplace=True):
    """
    Set the Corpus `docs` to handle tokens as n-grams.

    :param docs: a Corpus object
    :param n: size of the n-grams to generate
    :param join_str: string to join n-grams
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if n < 1:
        raise ValueError('`n` must be greater or equal 1')

    docs._ngrams = n
    docs._ngrams_join_str = join_str


#%% KWIC helpers

@parallelexec(collect_fn=merge_dicts)
def _build_kwic_parallel(docs, search_tokens, context_size, by_attr, match_type, ignore_case, glob_method,
                         inverse=False, highlight_keyword=None, with_window_indices=None, only_token_masks=False):
    """Parallel KWIC processing for a chunk of documents in `docs`."""
    # find matches for search criteria -> list of NumPy boolean mask arrays
    if only_token_masks:
        matchagainst = docs
    else:
        matchagainst = {lbl: d['_matchagainst'] for lbl, d in docs.items()}

    # keyword matches
    matches = _token_pattern_matches(matchagainst, search_tokens,
                                     match_type=match_type, ignore_case=ignore_case, glob_method=glob_method)

    if not only_token_masks and inverse:
        matches = {lbl: ~m for lbl, m in matches.items()}

    # build "context windows"
    left, right = context_size
    matchattr = by_attr or 'token'

    kwic_res = {}   # maps document labels to context windows
    for lbl, mask in matches.items():
        ind = np.where(mask)[0]

        # indices around each keyword match
        ind_windows = index_windows_around_matches(mask, left, right,
                                                   flatten=only_token_masks, remove_overlaps=True)

        if only_token_masks:    # return only boolean mask of matched token windows per document
            assert ind_windows.ndim == 1
            assert len(ind) <= len(ind_windows)

            # from indices back to boolean mask; this only works with remove_overlaps=True
            win_mask = np.repeat(False, len(mask))
            win_mask[ind_windows] = True

            if inverse:
                win_mask = ~win_mask

            kwic_res[lbl] = win_mask
        else:                   # return list of token windows per keyword match per document
            docdata = docs[lbl]
            tok_arr = docdata.pop('_matchagainst')

            if not isinstance(tok_arr, np.ndarray) or not np.issubdtype(tok_arr.dtype, str):
                assert isinstance(tok_arr, (list, tuple, np.ndarray))
                tok_arr = as_chararray(tok_arr)

            assert len(ind) == len(ind_windows)

            windows_in_doc = []     # context windows
            for match_ind, win in zip(ind, ind_windows):  # win is an array of indices into dtok_arr
                tok_win = tok_arr[win].tolist()     # context window for this match

                if highlight_keyword is not None:     # add "highlight" around matched keyword, e.g. to form "*keyword*"
                    highlight_mask = win == match_ind
                    assert np.sum(highlight_mask) == 1
                    highlight_ind = np.where(highlight_mask)[0][0]
                    tok_win[highlight_ind] = highlight_keyword + tok_win[highlight_ind] + highlight_keyword

                # add matched attribute window (usually tokens)
                win_res = {matchattr: tok_win}

                # optionally add indices
                if with_window_indices:
                    win_res['index'] = win

                # optionally add windows for other attributes
                for attr_key, attr_vals in docdata.items():
                    if attr_key != matchattr:
                        win_res[attr_key] = attr_vals[win].tolist()

                windows_in_doc.append(win_res)

            kwic_res[lbl] = windows_in_doc

    assert len(kwic_res) == len(docs)

    return kwic_res


def _finalize_kwic_results(kwic_results, only_non_empty, glue, as_tables, matchattr, with_attr):
    """
    Helper function to finalize raw KWIC results coming from `_build_kwic_parallel()`: Filter results,
    "glue" (join) tokens, transform to dataframe, return or dismiss attributes.
    """
    if only_non_empty:      # remove documents with no matches and hence no KWIC results
        kwic_results = {dl: windows for dl, windows in kwic_results.items() if len(windows) > 0}

    if glue is not None:    # join tokens in context windows
        kwic_results = {lbl: [glue.join(win[matchattr]) for win in windows] for lbl, windows in kwic_results.items()}

    if as_tables:     # convert to dataframes
        dfs = {}    # dataframe for each result
        for lbl, windows in kwic_results.items():
            if glue is not None:  # every "window" in windows is a concatenated string; there are no further attributes
                if windows:
                    dfs[lbl] = pd.DataFrame({'doc': np.repeat(lbl, len(windows)),
                                             'context': np.arange(len(windows)),
                                             matchattr: windows})
                elif not only_non_empty:
                    dfs[lbl] = pd.DataFrame({'doc': [], 'context': [], matchattr: []})
            else:               # every "window" in windows is a KWIC context window...
                win_dfs = []
                for i_win, win in enumerate(windows):
                    if isinstance(win, list):   # ... with separate tokens as list or a dict of tokens and attributes
                        win = {matchattr: win}

                    n_tok = len(win[matchattr])
                    df_windata = [np.repeat(lbl, n_tok),
                                  np.repeat(i_win, n_tok),
                                  win['index'],
                                  win[matchattr]]

                    if with_attr:
                        meta_cols = [col for col in win.keys() if col not in {matchattr, 'index'}]
                        df_windata.extend([win[col] for col in meta_cols])
                    else:
                        meta_cols = []

                    df_cols = ['doc', 'context', 'position', matchattr] + meta_cols
                    win_dfs.append(pd.DataFrame(dict(zip(df_cols, df_windata))))

                if win_dfs:
                    dfs[lbl] = pd.concat(win_dfs, axis=0)
                elif not only_non_empty:
                    dfs[lbl] = pd.DataFrame(dict(zip(['doc', 'context', 'position', matchattr],
                                                     [[] for _ in range(4)])))

        return dfs

    if not with_attr and glue is None:     # dismiss attributes
        return {lbl: [win[matchattr] for win in windows] for lbl, windows in kwic_results.items()}
    else:
        return kwic_results


def _create_embed_tokens_for_collocations(docs: Corpus, embed_tokens_min_docfreq, embed_tokens_set):
    """
    Helper function to generate ``embed_tokens`` set as used in :func:`~tmtookit.tokenseq.token_collocations`.

    If given, use `embed_tokens_min_docfreq` to populate ``embed_tokens`` using a minimum document frequency for
    token types in `docs`. Additionally use fixed set of tokens in `embed_tokens_set`.
    """
    if embed_tokens_min_docfreq is not None:
        if not isinstance(embed_tokens_min_docfreq, (float, int)):
            raise ValueError('`embed_tokens_min_docfreq` must be either None, a float or an integer')

        df_prop = isinstance(embed_tokens_min_docfreq, float)
        if df_prop and not 0.0 <= embed_tokens_min_docfreq <= 1.0:
            raise ValueError('if `embed_tokens_min_docfreq` is given as float, it must be a proportion in the '
                             'interval [0, 1]')
        elif not df_prop and embed_tokens_min_docfreq < 1:
            raise ValueError('if `embed_tokens_min_docfreq` is given as integer, it must be strictly positive')

        # get token types with document frequencies and filter them
        token_df = doc_frequencies(docs, proportions=df_prop)
        embed_tokens = {t for t, df in token_df.items() if df >= embed_tokens_min_docfreq}
        if embed_tokens_set:  # additionally use fixed set of tokens
            embed_tokens.update(embed_tokens_set)
        return embed_tokens
    else:
        return embed_tokens_set     # solely use fixed set of tokens


def _apply_collocations(docs: Corpus, joint_colloc: Dict[str, tuple],
                        tokens_as_hashes: bool, glue: Optional[str], return_joint_tokens: bool):
    """
    Helper function to apply collocations from `joint_colloc` to documents in `docs`. `joint_colloc` maps document label
    to a tuple containing new (joint) tokens and a mask as provided by :func:`~tmtookit.tokenseq.token_join_subsequent`
    with parameter ``return_mask=True``. The tokens can be given as strings or as hashes (integers).
    """
    stringstore = docs.nlp.vocab.strings
    if return_joint_tokens:
        joint_tokens = set()
    for lbl, (new_tok, mask) in joint_colloc.items():
        if new_tok:
            d = docs.spacydocs[lbl]   # get spacy document for that label

            # get unmasked token hashes of that document
            d.user_data['processed'] = _ensure_writable_array(d.user_data['processed'])
            tok_hashes = d.user_data['processed'][d.user_data['mask']]     # unmasked token hashes

            # get new tokens as strings
            if tokens_as_hashes:
                # the tokens in the collocations are hashes:
                # 1. get the token type strings for each collocation token hash `t` from the StringStore
                # 2. join the token type strings with `glue`
                new_tok_strs = [glue.join(stringstore[t] for t in colloc) for colloc in new_tok]
            else:
                # the tokens in the collocations are strings: use them as-is
                new_tok_strs = new_tok

            if return_joint_tokens:
                joint_tokens.update(new_tok_strs)

            # this doesn't work since slicing is copying:
            # d.user_data['processed'][d.user_data['mask']][mask > 1] = [stringstore.add(t) for t in new_tok]
            # so we have to take the long (and slow) route

            # add the strings as new token types to the StringStore and save the hashes to the array
            tok_hashes[mask > 1] = [stringstore.add(t) for t in new_tok_strs]   # replace with hashes of new tokens
            d.user_data['processed'][d.user_data['mask']] = tok_hashes     # copy back to original array

            # store the new mask
            d.user_data['mask'] = _ensure_writable_array(d.user_data['mask'])
            d.user_data['mask'][d.user_data['mask']] = mask > 0

    if return_joint_tokens:
        return joint_tokens


def _comparison_operator_from_str(which: str, common_alias=False, equal=True, whicharg: str = 'which') -> Callable:
    """
    Helper function to return the appropriate comparison operator function from a specifier string like ">=" or "<".
    """
    op_table = {
        '>': operator.gt,
        '>=': operator.ge,
        '<=': operator.le,
        '<': operator.lt,
    }

    if common_alias:
        op_table.update({
            'common': operator.ge,
            'uncommon': operator.le,
        })

    if equal:
        op_table['=='] = operator.eq

    if which not in op_table.keys():
        raise ValueError(f"`{whicharg}` must be one of {', '.join(op_table.keys())}")

    return op_table[which]
