"""
Internal module that implements functions that operate on :class:`~tmtoolkit.corpus.Corpus` objects.

The source is separated into sections using a ``#%% ...`` marker.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""
import logging
import operator
import os
import random
import re
import unicodedata
from collections import defaultdict
from copy import copy
from functools import partial, wraps
from glob import glob
from inspect import signature
from dataclasses import dataclass
from tempfile import mkdtemp
from typing import Dict, Union, List, Callable, Optional, Any, Iterable, Set, Tuple, Sequence, Collection, TypeVar, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd
from bidict import bidict
from scipy.sparse import csr_matrix
from spacy.strings import hash_string
from spacy.tokens import Doc
from loky import ProcessPoolExecutor

from ._document import document_token_attr, document_from_attrs, Document
from ..bow.dtm import create_sparse_dtm, dtm_to_dataframe
from ..utils import merge_dicts, empty_chararray, as_chararray, \
    flatten_list, combine_sparse_matrices_columnwise, pickle_data, unpickle_file, merge_sets, \
    path_split, read_text_file, linebreaks_win2unix, sample_dict, dict2df
from ..tokenseq import token_lengths, token_ngrams, token_match_multi_pattern, index_windows_around_matches, \
    token_match_subsequent, token_join_subsequent, npmi, token_collocations, numbertoken_to_magnitude, token_match, \
    collapse_tokens, simplify_unicode_chars
from ..types import Proportion, StrOrInt

from ._common import DATAPATH, LANGUAGE_LABELS, TOKENMAT_ATTRS, simplified_pos
from ._corpus import Corpus


TOKINDEX = 1
PTTRN_WS = re.compile(r'^\s+$')

CorpusFunc = TypeVar('CorpusFunc', bound=Callable[..., Any])

logger = logging.getLogger('tmtoolkit')


#%% parallel execution helpers and other decorators

merge_dicts_sorted = partial(merge_dicts, sort_keys=True)
merge_dicts_safe = partial(merge_dicts, safe=True)


@dataclass
class ParallelTask:
    """A parallel execution task for a loky reusable process executor."""
    # loky reusable process executor
    procexec: ProcessPoolExecutor
    # assignments of data chunks in `data` to workers; ``workers_assignments[i]`` contains list of keys in `data` which
    # worker ``i`` is assigned to work on
    workers_assignments: List[List[str]]
    # dict mapping data chunk key to data chunk
    data: dict


def _paralleltask(corpus: Corpus, tokens: Dict[str, Any]) -> ParallelTask:
    """
    Helper function to generate a :class:`~ParallelTask` for the reusable process executor and the worker process
    assignments in the :class:`~tmtoolkit.corpus.Corpus` Corpus `corpus`. By default, use `corpus`' document tokens as
    data chunks, otherwise use `tokens`.
    """
    return ParallelTask(corpus.procexec, corpus.workers_docs, tokens)


def parallelexec(collect_fn: Callable) -> Callable[[CorpusFunc], Callable]:
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
    def deco_fn(fn: CorpusFunc) -> CorpusFunc:
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
                if collect_fn is list:
                    return [res]
                else:
                    return res

        return cast(CorpusFunc, inner_fn)

    return deco_fn


def corpus_func_inplace_opt(fn: Callable) -> Callable:
    """
    Decorator for a Corpus function `fn` with an optional argument ``inplace``. This decorator makes sure that if
    `fn` is called with ``inplace=False``, the passed corpus will be copied before `fn` is applied to it. Then,
    the modified copy of corpus is returned. If ``inplace: bool = True``, `fn` is applied as usual.

    If you decorate a Corpus function with this decorator, the first argument of the Corpus function should be
    defined as positional-only argument, i.e. ``def corpfunc(docs, /, some_arg, other_arg, ...): ...``.

    :param fn: Corpus function `fn` with an optional argument ``inplace``
    :return: wrapper function of `fn`
    """
    @wraps(fn)
    def inner_fn(*args, **kwargs) -> Union[None, Corpus, Tuple[Corpus, Any]]:
        if not isinstance(args[0], Corpus):
            raise ValueError('first argument must be a Corpus object')

        if 'inplace' in kwargs:
            inplace = kwargs.pop('inplace')
        else:
            inplace = True

        # get Corpus object `corp`, optionally copy it
        if inplace:
            logger.debug(f'applying function {str(fn)} to {str(args[0])} inplace')
            corp = args[0]
        else:
            logger.debug(f'applying function {str(fn)} to a copy of {str(args[0])}')
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


def tabular_result_option(key: str, value: str) -> Callable:
    def deco_fn(fn):
        @wraps(fn)
        def inner_fn(*args, **kwargs):
            if not isinstance(args[0], Corpus):
                raise ValueError('first argument must be a Corpus object')

            if 'as_table' in kwargs:
                as_table = kwargs.pop('as_table')
            else:
                as_table = False

            ret = fn(*args, **kwargs)

            if as_table:
                if not isinstance(ret, dict):
                    raise ValueError('result must be a dictionary')
                if as_table is True:
                    sort = None
                else:  # as_table is string
                    sort = as_table
                return dict2df(ret, key, value, sort=sort)
            else:
                return ret

        return inner_fn

    return deco_fn



def corpus_func_update_bimaps(which_attrs: Union[str, Optional[Collection[str]]] = None) -> Callable:
    def deco_fn(fn):
        @wraps(fn)
        def inner_fn(*args, **kwargs):
            if not isinstance(args[0], Corpus):
                raise ValueError('first argument must be a Corpus object')

            ret = fn(*args, **kwargs)

            if isinstance(ret, Corpus):
                corp = ret
            elif isinstance(ret, tuple):
                if not ret or not isinstance(ret[0], Corpus):
                    raise ValueError('first return value must be a Corpus object')
                corp = ret[0]
            else:   # return type is None or something else -> we assume `fn` was called with `inplace: bool = True`
                corp = args[0]

            corp._update_bimaps(which_attrs=which_attrs)

            return ret

        return inner_fn

    return deco_fn


#%% Corpus functions with readonly access to Corpus data


def doc_tokens(docs: Corpus,
               select: Optional[Union[str, Collection[str]]] = None,
               sentences: bool = False,
               only_non_empty: bool = False,
               tokens_as_hashes: bool = False,
               with_attr: Union[bool, str, Sequence[str]] = False,
               as_tables: bool = False,
               as_arrays: bool = False,
               force_unigrams: bool = False) \
        -> Union[
               # multiple documents
               Dict[str, Union[List[StrOrInt],        # tokens
                               List[List[StrOrInt]],  # sentences with tokens
                               np.ndarray,                   # tokens
                               List[np.ndarray],             # sentences with tokens
                               Dict[str, Union[list, np.ndarray]],  # tokens with attributes
                               List[Dict[str, Union[list, np.ndarray]]],  # sentences with tokens with attributes
                               pd.DataFrame]],
               # single document
               List[StrOrInt],        # plain tokens
               List[List[StrOrInt]],  # sentences with plain tokens
               np.ndarray,                   # plain tokens
               List[np.ndarray],             # sentences with plain tokens
               Dict[str, Union[list, np.ndarray]],          # tokens with attributes
               List[Dict[str, Union[list, np.ndarray]]],    # sentences with tokens with attributes
               pd.DataFrame
           ]:
    """
    Retrieve documents' tokens from a Corpus or dict of SpaCy documents. Optionally also retrieve document and token
    attributes.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying the documents to fetch
    :param sentences: divide results into sentences; if True, each document will consist of a list of sentences which in
                      turn contain a list or array of tokens
    :param only_non_empty: if True, only return non-empty result documents
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
    :param with_attr: also return document and token attributes along with each token; if True, returns all default
                      attributes and custom defined attributes; if string, return this specific attribute; if sequence,
                      returns attributes specified in this sequence
    :param as_tables: return result as dataframe with tokens and document and token attributes in columns
    :param as_arrays: return result as NumPy arrays instead of lists
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :return: by default, a dict mapping document labels to document tokens data, which can be of different form,
             depending on the arguments passed to this function:
             (1) list of token strings or hash integers;
             (2) NumPy array of token strings or hash integers;
             (3) dict containing ``"token"`` key with values from (1) or (2) and document and token attributes with
                 their values as list or NumPy array;
             (4) dataframe with tokens and document and token attributes in columns;
             if `select` is a string not a dict of documents is returned, but a single document with one of the 4 forms
             described before; if `sentences` is True, another list level representing sentences is added
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
    elif isinstance(with_attr, bool):
        add_std_attrs = with_attr           # True specified, means load standard attributes
    else:
        raise ValueError(f'cannot handle argument `with_attr` of type "{type(with_attr)}"')

    # if requested by `with_attr = True`, add standard token attributes
    if add_std_attrs:
        with_attr_list.extend(docs.spacy_token_attrs)

    # get ngram setting
    if force_unigrams:
        ng = 1  # ngram setting; default is unigram
        ng_join_str = None
    else:
        ng = docs.ngrams
        ng_join_str = docs.ngrams_join_str

    # get document attributes with default values
    if add_std_attrs or with_attr_list:
        exclude_doc_attrs = {'has_sents'} if as_tables else {'has_sents', 'label'}
        doc_attrs = {k: v for k, v in docs.doc_attrs_defaults.items() if k not in exclude_doc_attrs}
        # rely on custom token attrib. w/ defaults as reported from Corpus
        custom_token_attrs_defaults = docs.custom_token_attrs_defaults
        if add_std_attrs:
            with_attr_list.extend(custom_token_attrs_defaults.keys())
    else:
        doc_attrs = {'label': ''} if as_tables else {}
        custom_token_attrs_defaults = {}

    # subset documents
    if select_docs is not None:
        docs = {lbl: docs[lbl] for lbl in select_docs}

    # make sure `doc_attrs` contains only the attributes listed in `with_attr_list`; if `with_attr = True`, don't
    # filter the `doc_attrs`
    if with_attr_list and not add_std_attrs:
        doc_attrs = {k: doc_attrs[k] for k in with_attr_list + ['label'] if k in doc_attrs.keys()}

    token_attrs = [k for k in with_attr_list if k not in set(doc_attrs.keys()) | {'label', 'has_sents'}]

    res = {}
    for lbl, d in docs.items():     # iterate through corpus with label `lbl` and Document objects `d`
        # skip this document if it is empty and `only_non_empty` is True
        n_tok = len(d)
        if only_non_empty and len(d) == 0:
            if select_docs is not None:
                raise ValueError(f'document "{lbl}" is an empty selected document but only non-empty documents should '
                                 f'be retrieved')
            continue

        token_base_attr = ['token']
        if as_tables and sentences and d.has_sents:   # add sentence numbers column
            token_base_attr = ['sent'] + token_base_attr

        attr_values = {}   # maps document or token attribute name to values

        # get token attributes (incl. tokens themselves)
        tok_attr_values = document_token_attr(d, attr=token_base_attr + token_attrs,
                                              default=custom_token_attrs_defaults,
                                              sentences=sentences and not as_tables,
                                              ngrams=ng,
                                              ngrams_join=ng_join_str,
                                              as_hashes=tokens_as_hashes,
                                              as_array=as_arrays or as_tables)

        # get document attributes
        if doc_attrs:
            # for tables, repeat the value to match the number of tokens, otherwise a document attrib. is a scalar value
            attr_values = {attr: np.repeat(d.doc_attrs.get(attr, default), n_tok) if as_tables
                           else d.doc_attrs.get(attr, default)
                           for attr, default in doc_attrs.items()}

        if attr_values:
            # add token attributes to document attributes (doc. attrib. come first in dict/dataframe)
            attr_values.update(tok_attr_values)
        else:
            attr_values = tok_attr_values

        if as_tables or with_attr_list:
            res[lbl] = attr_values
        else:  # tokens alone requested
            res[lbl] = attr_values['token']

    if as_tables:   # convert to dict of dataframes
        res = {lbl: pd.DataFrame(doc_data) for lbl, doc_data in res.items()}

    if isinstance(select, str):     # return single document
        return res[select]
    else:
        return res


@tabular_result_option('doc', 'length')
def doc_lengths(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None,
                as_table: Union[bool, str] = False) -> Union[Dict[str, int], pd.DataFrame]:
    """
    Return document length (number of tokens in doc.) for each document.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param as_table: if True, return result as dataframe; if a string, sort dataframe by this column; if string prefixed
                     with "-", sort by this column in descending order
    :return: dict of document lengths per document label or dataframe if `as_table` is active
    """

    select = _single_str_to_set(select, check_docs=docs)

    if select is None:
        return {lbl: len(d) for lbl, d in docs.items()}
    else:
        return {lbl: len(d) for lbl, d in docs.items() if lbl in select}


def doc_token_lengths(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None) -> Dict[str, List[int]]:
    """
    Return token lengths (number of characters of each token) for each document.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :return: dict with list of token lengths per document label
    """

    # get token lengths for each token in the vocabulary
    vocab_hashes = docs.bimaps['token'].keys()
    vocab_tokens = docs.bimaps['token'].values()
    # maps token hash to token length
    vocab_lengths = dict(zip(vocab_hashes, token_lengths(vocab_tokens)))

    select = _single_str_to_set(select, check_docs=docs)

    res = {}
    for lbl, d in docs.items():
        if select is None or lbl in select:
            tok = d.tokenmat[:, TOKINDEX]
            # lookup token length by hash
            res[lbl] = [vocab_lengths[h] for h in tok]

    return res


@tabular_result_option('doc', 'num_sents')
def doc_num_sents(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None,
                  as_table: Union[bool, str] = False) -> Union[Dict[str, int], pd.DataFrame]:
    """
    Return number of sentences for each document.

    .. note:: This number may be unreliable after filtering tokens in the corpus, since a filter may remove
              the starting tokens of sentences.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param as_table: if True, return result as dataframe; if a string, sort dataframe by this column; if string prefixed
                     with "-", sort by this column in descending order
    :return: dict with number of sentences per document label or dataframe if `as_table` is active
    """

    select = _single_str_to_set(select, check_docs=docs)
    res = {}

    try:
        for lbl, d in docs.items():
            # using max() here to make sure that each non-empty document has at least one sentence (the "sent_start"
            # array may not report any sentence starts after filtering which may otherwise result in non-empty documents
            # being reported as containing no sentences)
            if select is None or lbl in select:
                res[lbl] = max(int(np.sum(document_token_attr(d, attr='sent_start', as_array=True))),
                               1 if len(d) > 0 else 0)
        return res
    except KeyError:
        raise RuntimeError('sentence borders not set; Corpus documents probably not parsed with sentence recognition')


def doc_sent_lengths(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None) -> Dict[str, List[int]]:
    """
    Return sentence lengths (number of tokens of each sentence) for each document.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :return: dict with list of sentence lengths per document label
    """

    select = _single_str_to_set(select, check_docs=docs)
    res = {}

    for lbl, d in docs.items():
        if select is None or lbl in select:
            n = len(d)
            if n == 0:   # no tokens -> no sentences
                res[lbl] = []
            else:
                # get starting indices of sentences; make sure that at least the very first token is a starting index
                sent_start_ind = np.flatnonzero(document_token_attr(d, 'sent_start', as_array=True)).tolist() or [0]
                sent_start_ind.append(n)  # add the number of tokens as virtual next sentence starting index
                # calculate the lengths between the starting indices to get the sentence lengths
                res[lbl] = np.diff(sent_start_ind).tolist()

    return res


def doc_labels(docs: Corpus, sort: bool = False) -> List[str]:
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


def doc_labels_sample(docs: Corpus, n: int) -> Set[str]:
    """
    Generate random sample of document labels from `docs` with sample size `n`.

    :param docs: a Corpus object
    :param n: sample size; must be in interval ``[0, len(docs)]``
    :return: set of sampled document labels
    """
    if logger.isEnabledFor(logging.INFO):
        logger.info(f'sampling {n} documents out of {len(docs)} in the corpus')
    return set(random.sample(doc_labels(docs), n))


@tabular_result_option('doc', 'text')
def doc_texts(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None, collapse: Optional[str] = None,
              as_table: Union[bool, str] = False) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Return reconstructed document text from documents in `docs`. By default, uses whitespace token attribute to collapse
    tokens to document text, otherwise custom `collapse` string.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying the documents to fetch
    :param collapse: if None, use whitespace token attribute for collapsing tokens, otherwise use custom string
    :param as_table: if True, return result as dataframe; if a string, sort dataframe by this column; if string prefixed
                     with "-", sort by this column in descending order
    :return: dict with reconstructed document text per document label or dataframe if `as_table` is active
    """
    @parallelexec(collect_fn=merge_dicts)
    def _doc_texts(tokens, collapse):
        texts = {}
        for lbl, tok in tokens.items():
            if collapse is None:
                texts[lbl] = collapse_tokens(tok['token'], tok['whitespace'])
            else:
                texts[lbl] = collapse_tokens(tok, collapse)

        return texts

    select = _single_str_to_set(select)   # force doc_tokens output as dict

    if collapse is None:
        tokdata = doc_tokens(docs, select=select, with_attr='whitespace')
    else:
        tokdata = doc_tokens(docs, select=select)

    return _doc_texts(_paralleltask(docs, tokdata), collapse=collapse)


@tabular_result_option('token', 'freq')
def doc_frequencies(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None,
                    tokens_as_hashes: bool = False, force_unigrams: bool = False,
                    proportions: Proportion = Proportion.NO,
                    as_table: Union[bool, str] = False) \
        -> Union[Dict[StrOrInt, Union[int, float]], pd.DataFrame]:
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
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :param proportions: one of :attr:`~tmtoolkit.types.Proportion`: ``NO (0)`` – return counts; ``YES (1)`` – return
                        proportions; ``LOG (2)`` – return log10 of proportions
    :param as_table: if True, return result as dataframe; if a string, sort dataframe by this column; if string prefixed
                     with "-", sort by this column in descending order
    :return: dict mapping token to document frequency or dataframe if `as_table` is active
    """
    result_uses_hashes = docs.ngrams == 1 or force_unigrams

    if not result_uses_hashes and tokens_as_hashes:
        raise ValueError('supplied `docs` Corpus object uses n-grams; `tokens_as_hashes` must be False in that case')

    select = _single_str_to_set(select)  # force doc_tokens output as dict
    tokens = doc_tokens(docs, select=select, tokens_as_hashes=result_uses_hashes, force_unigrams=force_unigrams)

    if not tokens:   # empty corpus -> no doc. frequencies (prevent log(0) domain error)
        return {}

    # the following is faster than using `Counter`
    hashes = np.array(flatten_list(set(dtok) for dtok in tokens.values()),  # count *unique* occurrences per document
                      dtype='uint64' if result_uses_hashes else 'str')
    hashes, counts = np.unique(hashes, return_counts=True)

    if proportions == Proportion.YES:
        counts = counts / len(tokens)
    elif proportions == Proportion.LOG:
        counts = np.log10(counts) - np.log10(len(tokens))

    if tokens_as_hashes or not result_uses_hashes:
        return dict(zip(hashes, counts))
    else:
        return {docs.bimaps['token'][h]: n for h, n in zip(hashes, counts)}


def doc_vectors(docs: Union[Corpus, Dict[str, Doc]], select: Optional[Union[str, Collection[str]]] = None,
                collapse: Optional[str] = None, omit_empty: bool = False) -> Dict[str, np.ndarray]:
    """
    Return a vector representation for each document in `docs`. The vector representation's size corresponds to the
    vector width of the language model that is used (usually 300).

    .. note:: `docs` can be either a :class:`Corpus` object or dict of SpaCy Doc objects. If it is a Corpus object,
              it must use a SpaCy language model with word vectors (i.e. an *_md* or *_lg* model).
              If the corpus was transformed, especially if tokens were removed, then you should set `collapse` to `" "`.
              Otherwise tokens may be joint because of missing whitespace between them.

    :param docs: a :class:`Corpus` object or dict mapping document labels to SpaCy Doc objects
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param collapse: if None, use whitespace token attribute for collapsing tokens, otherwise use custom string
    :param omit_empty: omit empty documents
    :return: dict mapping document label to vector representation of the document
    """
    spacydocs = _spacydocs_for_vectors(docs, select=select, collapse=collapse)

    return {lbl: d.vector for lbl, d in spacydocs.items() if not omit_empty or len(d) > 0}


def token_vectors(docs: Union[Corpus, Dict[str, Doc]], select: Optional[Union[str, Collection[str]]] = None,
                  collapse: Optional[str] = None, omit_oov: bool = True) -> Dict[str, np.ndarray]:
    """
    Return a token vectors matrix for each document in `docs`. This matrix is of size *n* by *m* where *n* is
    the number of tokens in the document and *m* is the vector width of the language model that is used (usually 300).
    If `omit_oov` is True, *n* will be number of tokens in the document **for which there is a word vector** in
    used the language model.

    .. note:: `docs` can be either a :class:`Corpus` object or dict of SpaCy Doc objects. If it is a Corpus object,
              it must use a SpaCy language model with word vectors (i.e. an *_md* or *_lg* model).
              If the corpus was transformed, especially if tokens were removed, then you should set `collapse` to `" "`.
              Otherwise tokens may be joint because of missing whitespace between them.

    :param docs: a :class:`Corpus` object or dict mapping document labels to SpaCy Doc objects
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param collapse: if None, use whitespace token attribute for collapsing tokens, otherwise use custom string
    :param omit_oov: omit "out of vocabulary" tokens, i.e. tokens without a vector
    :return: dict mapping document label to token vectors matrix
    """
    spacydocs = _spacydocs_for_vectors(docs, select=select, collapse=collapse)

    return {lbl: np.vstack([t.vector for t in d if not (omit_oov and t.is_oov)])
                            if len(d) > 0 else np.array([], dtype='float32')
            for lbl, d in spacydocs.items()}


def spacydocs(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None, collapse: Optional[str] = None) \
        -> Dict[str, Doc]:
    """
    Generate `SpaCy Doc <https://spacy.io/api/doc/>`_ objects from current corpus.

    .. note:: If the corpus was transformed, especially if tokens were removed, then you should set `collapse` to `" "`.
              Otherwise tokens may be joint because of missing whitespace between them.

    :param docs: a :class:`Corpus` object or a dict of token strings
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param collapse: if None, use whitespace token attribute for collapsing tokens, otherwise use custom string
    :return: dict mapping document labels to `SpaCy Doc <https://spacy.io/api/doc/>`_ objects
    """
    # set document extensions for document attributes
    for attr, default in docs.doc_attrs_defaults.items():
        Doc.set_extension(attr, default=default, force=True)

    # generate texts
    logger.debug('generating document texts')
    txts = doc_texts(docs, select=select, collapse=collapse)

    # set up pipe
    logger.debug('generating SpaCy documents from Corpus instance documents')
    pipe = docs._nlppipe(txts.values())
    sp_docs = {}

    # iterate through SpaCy documents
    for lbl, sp_d in zip(txts.keys(), pipe):
        # take over document attributes from corresponding Document object
        for attr, val in docs[lbl].doc_attrs.items():
            setattr(sp_d._, attr, val)

        assert sp_d._.label == lbl, f'document label "{lbl}" must match SpaCy document attribute'
        assert lbl not in sp_docs, f'document label "{lbl}" must be unique'
        sp_docs[lbl] = sp_d

    return sp_docs


def vocabulary(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None, tokens_as_hashes: bool = False,
               force_unigrams: bool = False, sort: bool = False, convert_uint64hashes: bool = True) \
        -> Union[Set[StrOrInt], List[StrOrInt]]:
    """
    Return the vocabulary, i.e. the set or sorted list of unique token types, of a Corpus or a dict of token strings.

    :param docs: a :class:`Corpus` object or a dict of token strings
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param tokens_as_hashes: use token hashes instead of token strings
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :param sort: if True, sort the vocabulary
    :param convert_uint64hashes: if True, convert NumPy ``uint64`` hashes to Python ``int`` types (only is effective if
                                 `tokens_as_hashes` is True)
    :return: set or, if `sort` is True, a sorted list of unique token types
    """
    if select is not None or (not force_unigrams and docs.ngrams > 1):
        if isinstance(select, str):   # force doc_tokens output as dict
            select = [select]
        logger.debug("generating vocabulary from documents' tokens")
        v = flatten_list(doc_tokens(docs, select=select, tokens_as_hashes=tokens_as_hashes).values())
    else:
        logger.debug("generating vocabulary from tokens bimap")
        if tokens_as_hashes:
            v = docs.bimaps['token'].keys()
        else:
            v = docs.bimaps['token'].values()

    v = set(v)

    if sort:
        if tokens_as_hashes and convert_uint64hashes:
            v = map(int, v)
        return sorted(v)
    else:
        if tokens_as_hashes and convert_uint64hashes:
            return set(map(int, v))
        else:
            return v


@tabular_result_option(key='token', value='freq')
def vocabulary_counts(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None,
                      proportions: Proportion = Proportion.NO,
                      tokens_as_hashes: bool = False, force_unigrams: bool = False,
                      convert_uint64hashes: bool = True, as_table: Union[bool, str] = False) \
        -> Union[Dict[StrOrInt, Union[int, float]], pd.DataFrame]:
    """
    Return a dict mapping the tokens in the vocabulary to their respective number of occurrences across all or selected
    documents.

    :param docs: a :class:`Corpus` object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param proportions: one of :attr:`~tmtoolkit.types.Proportion`: ``NO (0)`` – return counts; ``YES (1)`` – return
                        proportions; ``LOG (2)`` – return log10 of proportions
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :param convert_uint64hashes: if True, convert NumPy ``uint64`` hashes to Python ``int`` types (only is effective if
                                 `tokens_as_hashes` is True)
    :param as_table: if True, return result as dataframe; if a string, sort dataframe by this column; if string prefixed
                     with "-", sort by this column in descending order
    :return: dict mapping the tokens in the vocabulary to their respective counts or dataframe if `as_table` is active
    """
    result_uses_hashes = docs.ngrams == 1 or force_unigrams

    if not result_uses_hashes and tokens_as_hashes:
        raise ValueError('supplied `docs` Corpus object uses n-grams; `tokens_as_hashes` must be False in that case')

    if isinstance(select, str):   # force doc_tokens output as dict
        select = [select]
    tok = doc_tokens(docs, select=select, tokens_as_hashes=result_uses_hashes, force_unigrams=force_unigrams)

    if not tok:  # shortcut
        return {}

    # the following is faster than using `Counter`
    hashes = np.array(flatten_list(tok.values()), dtype='uint64' if result_uses_hashes else 'str')
    hashes, counts = np.unique(hashes, return_counts=True)

    if proportions == Proportion.YES:
        counts = counts / np.sum(counts)
    elif proportions == Proportion.LOG:
        counts = np.log10(counts) - np.log10(np.sum(counts))

    if tokens_as_hashes or not result_uses_hashes:
        if tokens_as_hashes and convert_uint64hashes:
            hashes = hashes.tolist()
        return dict(zip(hashes, counts))
    else:
        return {docs.bimaps['token'][h]: n for h, n in zip(hashes, counts)}


def vocabulary_size(docs: Union[Corpus, Dict[str, List[str]]], select: Optional[Union[str, Collection[str]]] = None,
                    force_unigrams: bool = False) -> int:
    """
    Return size of the vocabulary, i.e. number of unique token types in `docs` (or a subset via `select`).

    :param docs: a :class:`Corpus` object or a dict of token strings
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :return: size of the vocabulary
    """
    return len(vocabulary(docs, select=select, tokens_as_hashes=True, force_unigrams=force_unigrams,
                          convert_uint64hashes=False))


def tokens_table(docs: Corpus,
                 select: Optional[Union[str, Collection[str]]] = None,
                 sentences: bool = False,
                 tokens_as_hashes: bool = False,
                 with_attr: Union[bool, str, Sequence[str]] = True,
                 force_unigrams: bool = False) -> pd.DataFrame:
    """
    Generate a dataframe with tokens and document/token attributes. Result has columns "doc" (document label),
    "position" (token position in the document, starting at zero), "token" and optional columns for
    document/token attributes.

    :param docs: a :class:`Corpus` object
    :param select: if not None, this can be a single string or a sequence of strings specifying the documents to fetch
    :param sentences: if True, list sentence index (starting at zero) per token in `sent` column
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
    :param with_attr: also return document and token attributes along with each token; if True, returns all default
                      attributes and custom defined attributes; if sequence, returns attributes specified in this
                      sequence
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :return: dataframe with tokens and document/token attributes
    """
    @parallelexec(collect_fn=list)
    def _tokens_table(chunks):
        # store data for dataframe as dict mapping columns to values
        col_data = defaultdict(list)

        # iterate through document dicts per parallel processing chunk
        for lbl, d in chunks.items():
            n = None

            if not isinstance(d, dict):
                d = {'token': d}

            # make sure tokens are retrieved first in order to get `n`
            attrs = ['token'] + list(set(d.keys()) - {'token'})

            if 'label' not in d:   # make sure to set document label
                d['label'] = lbl
                attrs.append('label')

            # iterate through attributes
            for a in attrs:
                val = d[a]
                if isinstance(val, np.ndarray):   # token attrib.
                    if n is None:
                        n = len(val)
                        col_data['position'].append(np.arange(n))
                    col_data[a].append(val)
                else:                             # document attrib.
                    if n is None:
                        raise ValueError('number of tokens must be determined before')
                    col_data[a].append(np.repeat(val, n))

        # construct dataframe for all data passed to this worker process
        return pd.DataFrame({col: np.concatenate(vals) for col, vals in col_data.items()})

    # get dict of dataframes
    if with_attr is True:
        with_attr = list(docs.spacy_token_attrs)
        if sentences:
            with_attr.append('sent')
        with_attr.extend(docs.doc_attrs)
        with_attr.extend(docs.custom_token_attrs_defaults.keys())
    elif with_attr is False:
        with_attr = []
    elif isinstance(with_attr, str):
        with_attr = [with_attr]
    elif isinstance(with_attr, list):
        with_attr = with_attr.copy()
    elif isinstance(with_attr, tuple):
        with_attr = list(with_attr)
    else:
        raise ValueError(f'cannot handle argument `with_attr` of type "{type(with_attr)}"')

    if sentences and 'sent' not in with_attr:
        with_attr.append('sent')

    logger.debug('getting tokens')
    doc_tok = doc_tokens(docs,
                         select={select} if isinstance(select, str) else select,
                         sentences=False,
                         tokens_as_hashes=tokens_as_hashes,
                         only_non_empty=True,
                         with_attr=with_attr,
                         as_arrays=True,
                         force_unigrams=force_unigrams)

    logger.debug('generating result')
    if doc_tok:
        dfs = _tokens_table(_paralleltask(docs, doc_tok))
        if len(dfs) == 1:
            res = dfs[0]
        else:
            # concatenate the dataframes from the worker processes
            res = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        # empty corpus
        logger.debug('corpus is empty')
        cols = ['label']
        if sentences:
            cols.append('sent')
        cols.extend(['position', 'token'])
        if isinstance(with_attr, (list, tuple, set)):
            cols.extend([a for a in with_attr if a not in cols])

        res = pd.DataFrame({c: [] for c in cols})

    if sentences:
        first_cols = ['doc', 'sent', 'position', 'token']
    else:
        first_cols = ['doc', 'position', 'token']

    cols = first_cols + sorted(c for c in res.columns if c not in first_cols + ['label'])
    return res.sort_values(['label', 'position'])\
        .rename(columns={'label': 'doc'})\
        .reindex(columns=cols)\
        .reset_index(drop=True)


def corpus_tokens_flattened(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None,
                            sentences: bool = False, tokens_as_hashes: bool = False,
                            as_array: bool = False, force_unigrams: bool = False) -> Union[list, np.ndarray]:
    """
    Return tokens (or token hashes) from `docs` as flattened list, simply concatenating  all documents.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param sentences: divide results into sentences; if True, the result will consist of a list of sentences
    :param tokens_as_hashes: passed to :func:`doc_tokens`; if True, return token hashes instead of string tokens
    :param as_array: if True, return NumPy array instead of list
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :return: list or NumPy array (depending on `as_array`) of token strings or hashes (depending on `tokens_as_hashes`);
             if `sentences` is True, the result is a list of sentences that in turn are token lists/arrays
    """

    if isinstance(select, str):  # force doc_tokens output as dict
        select = [select]

    tok = doc_tokens(docs, select=select, sentences=sentences, only_non_empty=True,
                     tokens_as_hashes=tokens_as_hashes, as_arrays=as_array, force_unigrams=force_unigrams)

    dtype = 'uint64' if tokens_as_hashes else 'str'
    if as_array and not sentences:
        if tok:
            return np.concatenate(list(tok.values()), dtype=dtype)
        else:
            return np.array([], dtype=dtype)
    else:
        res = flatten_list(tok.values())

        if res or not sentences:
            return res
        else:
            if as_array:
                return [np.array([], dtype=dtype)]
            else:
                return [[]]


def corpus_num_tokens(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None) -> int:
    """
    Return the number of tokens in a Corpus `docs`.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :return: number of tokens
    """
    return sum(doc_lengths(docs, select=select).values())


def corpus_num_chars(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None) -> int:
    """
    Return the number of characters (excluding whitespace) in a Corpus `docs`.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :return: number of characters
    """
    return sum(sum(n) for n in doc_token_lengths(docs, select=select).values())


def corpus_unique_chars(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None) -> Set[str]:
    """
    Return the set of characters used in a Corpus `docs`.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :return: set of characters
    """
    vocab = vocabulary(docs, select=select)

    chars = set()
    for t in vocab:
        chars.update(set(t))

    return chars


def corpus_collocations(docs: Corpus,
                        select: Optional[Union[str, Collection[str]]] = None,
                        threshold: Optional[float] = None,
                        min_count: int = 1,
                        embed_tokens_min_docfreq: Optional[Union[int, float]] = None,
                        embed_tokens_set: Optional[Set] = None,
                        statistic: Callable = npmi,
                        return_statistic: bool = True,
                        rank: Optional[str] = 'desc',
                        as_table: bool = True,
                        glue: str = ' ',
                        **statistic_kwargs) \
        -> Union[pd.DataFrame, List[Union[tuple, str]]]:
    """
    Identify token collocations in the corpus `docs`. Collocations are tokens that occur together in a series
    frequently (i.e. more than would be expected by chance).

    .. seealso:: :func:`~tmtoolkit.tokenseq.token_collocations`

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param threshold: minimum statistic value for a collocation to enter the results; if None, results are not filtered
    :param min_count: ignore collocations with number of occurrences below this threshold
    :param embed_tokens_min_docfreq: dynamically generate the set of ``embed_tokens`` used when calling
                                     :func:`~tmtoolkit.tokenseq.token_collocations` by using a minimum document
                                     frequency (see :func:`~doc_frequencies`); if this is an integer, it is used as
                                     absolute count, if it is a float, it is used as proportion
    :param embed_tokens_set: tokens that, if occurring inside an n-gram, are not counted; see :func:`token_ngrams`
    :param statistic: function to calculate the statistic measure from the token counts; use one of the
                      ``[n]pmi[2,3]`` functions provided in the :mod:`~tmtoolkit.tokenseq` module or provide
                      your own function which must accept parameters ``n_x, n_y, n_xy, n_total``; see
                      :func:`~tmtoolkit.tokenseq.pmi` for more information
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

    if docs.ngrams > 1:
        raise ValueError(f'this function is only applicable to Corpus objects with unigrams, but `docs` has '
                         f'docs.ngrams set to {docs.ngrams}')

    logger.debug('getting flattened tokens')
    tok = corpus_tokens_flattened(docs, select=select, sentences=True, tokens_as_hashes=True, as_array=True)
    logger.debug('getting vocabulary counts')
    vocab_counts = vocabulary_counts(docs, tokens_as_hashes=True)

    # generate ``embed_tokens`` set as used in :func:`~tmtookit.tokenseq.token_collocations`
    logger.debug('creating embed tokens')
    embed_tokens = _create_embed_tokens_for_collocations(docs, embed_tokens_min_docfreq, embed_tokens_set,
                                                         tokens_as_hashes=True)

    # identify collocations
    logger.debug('identifying collocations')
    colloc = token_collocations(tok, threshold=threshold, min_count=min_count, embed_tokens=embed_tokens,
                                vocab_counts=vocab_counts, statistic=statistic, return_statistic=return_statistic,
                                rank=rank, glue=glue, tokens_as_hashes=True, hashes2tokens=docs.bimaps['token'],
                                **statistic_kwargs)

    logger.debug('generating result')
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
                   select: Optional[Union[str, Collection[str]]] = None,
                   max_documents: Optional[int] = None,
                   max_tokens_string_length: Optional[int] = None) -> str:
    """
    Generate a summary of this object, i.e. the first tokens of each document and some summary statistics.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param max_documents: maximum number of documents to print; ``None`` uses default value 10; set to -1 to
                          print *all* documents; this setting is disabled in `select` is not None
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

    n_docs = len(docs)
    summary = f'Corpus with {n_docs} document' \
              f'{"s" if n_docs > 1 else ""} in ' \
              f'{LANGUAGE_LABELS[docs.language].capitalize()}'

    select = _single_str_to_set(select, check_docs=docs)

    if select is not None:
        summary += f' ({len(select)} document{"s" if len(select) > 1 else ""} selected for display)'

    logger.info('generating document texts')
    texts = doc_texts(docs, select=select, collapse=' ')
    dlengths = doc_lengths(docs, select=select)

    for i, (lbl, tokstr) in enumerate(texts.items()):
        tokstr = tokstr.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        if select is None and i >= max_documents:
            break
        if max_tokens_string_length >= 0 and len(tokstr) > max_tokens_string_length:
            tokstr = tokstr[:max_tokens_string_length] + '...'

        summary += f'\n> {lbl} ({dlengths[lbl]} tokens): {tokstr}'

    if select is None and len(docs) > max_documents:
        summary += f'\n(and {len(docs) - max_documents} more documents)'

    summary += f'\ntotal number of tokens: {corpus_num_tokens(docs)} / vocabulary size: {vocabulary_size(docs)}'

    return summary


def print_summary(docs: Corpus,
                  select: Optional[Union[str, Collection[str]]] = None,
                  max_documents: Optional[int] = None,
                  max_tokens_string_length: Optional[int] = None) -> None:
    """
    Print a summary of this object, i.e. the first tokens of each document and some summary statistics.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param max_documents: maximum number of documents to print; ``None`` uses default value 10; set to -1 to
                          print *all* documents; this setting is disabled in `select` is not None
    :param max_tokens_string_length: maximum string length of concatenated tokens for each document; ``None`` uses
                                     default value 50; set to -1 to print complete documents
    """
    print(corpus_summary(docs, select=select, max_documents=max_documents,
                         max_tokens_string_length=max_tokens_string_length))


def dtm(docs: Corpus, select: Optional[Union[str, Collection[str]]] = None, as_table: bool = False,
        dtype: Optional[Union[str, np.dtype]] = None, return_doc_labels: bool = False, return_vocab: bool = False) \
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
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param as_table: return result as dense pandas DataFrame
    :param dtype: use a specific matrix dtype; otherwise dtype will be uint32
    :param return_doc_labels: if True, additionally return sorted document labels that correspond to the rows of the
                              document-term matrix
    :param return_vocab: if True, additionally return the sorted vocabulary that corresponds to the columns of the
                         document-term matrix
    :return: document-term matrix as sparse matrix or dense dataframe; additionally sorted document labels and/or sorted
             vocabulary if `return_doc_labels` and/or `return_vocab` is True
    """
    @parallelexec(collect_fn=list)
    def _sparse_dtms(chunk):
        vocab = sorted(set(flatten_list(chunk.values())))
        alloc_size = sum(len(set(dtok)) for dtok in chunk.values())  # sum of *unique* tokens in each document

        return (create_sparse_dtm(vocab, chunk.values(), alloc_size, vocab_is_sorted=True, dtype=dtype),
                chunk.keys(),
                vocab)

    select = _single_str_to_set(select)
    logger.debug('getting tokens')
    tokens = doc_tokens(docs, select=select)

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'generating sparse DTM with {len(tokens)} documents and '
                    f'vocab size {len(set(flatten_list(tokens.values())))}')

    if len(tokens) > 0:
        logger.debug('generating sparse DTM')
        res = _sparse_dtms(_paralleltask(docs, tokens=tokens))
        w_dtms, w_doc_labels, w_vocab = zip(*res)
        dtm, vocab, dtm_doc_labels = combine_sparse_matrices_columnwise(w_dtms, w_vocab, w_doc_labels)
        # sort according to document labels
        dtm = dtm[np.argsort(dtm_doc_labels), :]
        doc_labels = np.sort(dtm_doc_labels)
    else:
        logger.debug('empty corpus')
        dtm = csr_matrix((0, 0), dtype=dtype or 'int32')   # empty sparse matrix
        vocab = empty_chararray()
        doc_labels = empty_chararray()

    logger.debug('generating result')
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


def ngrams(docs: Corpus, n: int, select: Optional[Union[str, Collection[str]]] = None, join: bool = True,
           join_str: str = ' ') -> Dict[str, Union[List[str], str]]:
    """
    Generate and return n-grams of length `n`.

    :param docs: a Corpus object
    :param n: length of n-grams, must be >= 2
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
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

    select = _single_str_to_set(select)
    logger.debug('getting tokens')
    tokens = doc_tokens(docs, select=select)
    logger.debug(f'generating {n}-grams')
    return _ngrams(_paralleltask(docs, tokens=tokens))


def kwic(docs: Corpus, search_tokens: Any, context_size: Union[int, Tuple[int, int], List[int]] = 2,
         select: Optional[Union[str, Collection[str]]] = None,
         by_attr: Optional[str] = None, match_type: str = 'exact', ignore_case: bool = False,
         glob_method: str = 'match', inverse: bool = False, with_attr: Union[bool, str, Sequence[str]] = False,
         as_tables: bool = False, only_non_empty: bool = False, glue: Optional[str] = None,
         highlight_keyword: Optional[str] = None) \
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
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
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
                      attributes and custom defined attributes; if sequence, returns attributes specified in this
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

    by_attr = by_attr or 'token'
    select = _single_str_to_set(select, check_docs=docs)

    logger.debug('getting data to match against')

    try:
        matchdata = _match_against(docs, by_attr, select=select,
                                   default=docs.custom_token_attrs_defaults.get(by_attr, None))
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    if with_attr:
        logger.debug('getting tokens')
        docs_w_attr = doc_tokens(docs, select=select, with_attr=with_attr, as_arrays=True)
        prepared = {}
        for lbl, matchagainst in matchdata.items():
            if by_attr != 'token':
                d = {k: v for k, v in docs_w_attr[lbl].items() if k != 'token'}
            else:
                d = docs_w_attr[lbl]
            
            prepared[lbl] = merge_dicts((d, {'_matchagainst': matchagainst}))
    else:
        prepared = {k: {'_matchagainst': v} for k, v in matchdata.items()}

    logger.debug('generating KWIC')
    kwicres = _build_kwic_parallel(_paralleltask(docs, prepared), search_tokens=search_tokens,
                                   context_size=context_size, by_attr=by_attr,
                                   match_type=match_type, ignore_case=ignore_case,
                                   glob_method=glob_method, inverse=inverse, highlight_keyword=highlight_keyword,
                                   with_window_indices=as_tables, only_token_masks=False)

    return _finalize_kwic_results(kwicres, only_non_empty=only_non_empty, glue=glue, as_tables=as_tables,
                                  matchattr=by_attr, with_attr=bool(with_attr))


def kwic_table(docs: Corpus, search_tokens: Any, context_size: Union[int, Tuple[int, int], List[int]] = 2,
               select: Optional[Union[str, Collection[str]]] = None,
               by_attr: Optional[str] = None, match_type: str = 'exact', ignore_case: bool = False,
               glob_method: str = 'match', inverse: bool = False, with_attr: Union[bool, str, Sequence[str]] = False,
               glue: str = ' ', highlight_keyword: Optional[str] = '*') -> pd.DataFrame:
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
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
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
                      attributes and custom defined attributes; if sequence, returns attributes specified in this
                      sequence
    :param glue: if not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: if not None, this must be a string which is used to indicate the start and end of the
                              matched keyword
    :return: dataframe with columns ``doc`` (document label), ``context`` (document-specific context number)
             and ``kwic`` (KWIC result)
    """

    kwicres = kwic(docs, search_tokens=search_tokens, context_size=context_size, select=select, by_attr=by_attr,
                   match_type=match_type, ignore_case=ignore_case, glob_method=glob_method, inverse=inverse,
                   with_attr=with_attr, as_tables=True, only_non_empty=True, glue=glue,
                   highlight_keyword=highlight_keyword)

    logger.debug('turning KWIC results into table')

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
            cols.extend([a for a in docs.spacy_token_attrs if a != by_attr])
        elif isinstance(with_attr, list):
            cols.extend([a for a in with_attr if a != by_attr])
        if isinstance(with_attr, str) and with_attr != by_attr:
            cols.append(with_attr)

        return pd.DataFrame(dict(zip(cols, [[] for _ in range(len(cols))])))


#%% Corpus I/O


@corpus_func_inplace_opt
def corpus_add_files(docs: Corpus, files: Union[str, Collection[str], Dict[str, str]], encoding: str = 'utf8',
                     doc_label_fmt: str = '{path}-{basename}', doc_label_path_join: str = '_',
                     read_size: int = -1, sample: Optional[int] = None, force_unix_linebreaks: bool = True,
                     inplace: bool = True) -> Optional[Corpus]:
    """
    Read text documents from files passed in `files` and add them to the corpus. If `files` is a dict, the dict keys
    represent the document labels. Otherwise, the document label for each new document is determined via format string
    `doc_label_fmt`.

    :param docs: a Corpus object
    :param files: single file path string or sequence of file paths or dict mapping document label to file path
    :param encoding: character encoding of the files
    :param doc_label_fmt: document label format string with placeholders "path", "basename", "ext"
    :param doc_label_path_join: string with which to join the components of the file paths
    :param custom_doc_labels: instead generating document labels from `doc_label_fmt`, pass a list of document labels
                              to be used directly
    :param read_size: max. number of characters to read. -1 means read full file.
    :param sample: if given, draw random sample of size `sample` from `files` (without replacement)
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if isinstance(files, str):
        filepaths = [files]
        filelabels = None
    elif isinstance(files, dict):
        filepaths = files.values()
        filelabels = dict(zip(files.values(), files.keys()))   # reverse mapping
        if len(filelabels) != len(filepaths):
            raise ValueError('file paths in `files` must be unique')
    else:  # list
        filepaths = files
        filelabels = None

    if sample is not None:
        logger.info(f'sampling {sample} file(s) out of {len(filepaths)}')
        filepaths = random.sample(filepaths, sample)

    logger.info(f'adding text from {len(filepaths)} file(s)')

    docs.update(_load_text_from_files(filepaths, filelabels, existing_docs=set(docs.keys()), encoding=encoding,
                                      doc_label_fmt=doc_label_fmt, doc_label_path_join=doc_label_path_join,
                                      read_size=read_size, force_unix_linebreaks=force_unix_linebreaks))


@corpus_func_inplace_opt
def corpus_add_folder(docs: Corpus, folder: str, valid_extensions: Collection[str] = ('txt', ),
                      encoding: str = 'utf8', strip_folderpath_from_doc_label: bool = True,
                      doc_label_fmt: str = '{path}-{basename}', doc_label_path_join: str = '_', read_size: int = -1,
                      sample: Optional[int] = None, force_unix_linebreaks: bool = True, inplace: bool = True) \
        -> Optional[Corpus]:
    """
    Read documents residing in folder `folder` and ending on file extensions specified via `valid_extensions` and
    add these to the corpus. This is done recursively, i.e. documents are also loaded from sub-folders inside `folder`.

    Note that only raw text files can be read, not PDFs, Word documents, etc. These must be converted to raw
    text files beforehand, for example with *pdttotext* (poppler-utils package) or *pandoc*.

    :param docs: a Corpus object
    :param folder: folder from where the files are read
    :param valid_extensions: collection of valid file extensions like .txt, .md, etc.
    :param encoding: character encoding of the files
    :param strip_folderpath_from_doc_label: if True, do not include the folder path in the document label
    :param doc_label_fmt: document label format string with placeholders "path", "basename", "ext"
    :param doc_label_path_join: string with which to join the components of the file paths
    :param read_size: max. number of characters to read. -1 means read full file.
    :param sample: if given, draw random sample of size `sample` from all loaded files
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if not os.path.exists(folder):
        raise IOError(f'path does not exist: "{folder}"')

    if isinstance(valid_extensions, str):
        valid_extensions = (valid_extensions, )

    new_docs = {}
    # iterate through all files in `folder` and its sub-folders
    logger.debug('reading files')
    for root, _, files in os.walk(folder):
        if not files:
            continue

        for fname in files:
            # get file path components
            basename, ext = os.path.splitext(fname)
            basename = basename.strip()
            if ext:
                ext = ext[1:]

            fpath = os.path.join(root, fname)

            if valid_extensions and (not ext or ext not in valid_extensions):  # skip files with wrong file ext.
                continue

            # load the text
            text = read_text_file(fpath, encoding=encoding, read_size=read_size,
                                  force_unix_linebreaks=force_unix_linebreaks)

            # create the document label
            if strip_folderpath_from_doc_label:
                dirs = path_split(root[len(folder)+1:])
            else:
                dirs = path_split(root)

            lbl = doc_label_fmt.format(path=doc_label_path_join.join(dirs), basename=basename, ext=ext)
            if lbl.startswith('-'):
                lbl = lbl[1:]

            # check for duplicate and add the data from the file
            if lbl in docs or lbl in new_docs:
                raise ValueError(f'duplicate document label "{lbl}" not allowed')

            new_docs[lbl] = text

    if sample is not None:
        logger.info(f'sampling {sample} documents(s) out of {len(new_docs)}')
        new_docs = sample_dict(new_docs, n=sample)

    logger.info(f'adding text from {len(new_docs)} documents(s)')
    docs.update(new_docs)


@corpus_func_inplace_opt
def corpus_add_tabular(docs: Corpus, files: Union[str, Collection[str]],
                       id_column: StrOrInt, text_column: StrOrInt,
                       prepend_columns: Optional[Sequence[str]] = None, encoding: str = 'utf8',
                       doc_label_fmt: str = '{basename}-{id}', sample: Optional[int] = None,
                       force_unix_linebreaks: bool = True, pandas_read_opts: Optional[Dict[str, Any]] = None,
                       inplace: bool = True) -> Optional[Corpus]:
    """
    Add documents from tabular (CSV or Excel) file(s) to the corpus.

    :param docs: a Corpus object
    :param files: single string or list of strings with path to file(s) to load
    :param id_column: column name or column index of document identifiers
    :param text_column: column name or column index of document texts
    :param prepend_columns: if not None, pass a list of columns whose contents should be added before the document
                            text, e.g. ``['title', 'subtitle']``
    :param encoding: character encoding of the files
    :param doc_label_fmt: document label format string with placeholders ``"basename"``, ``"id"`` (document ID), and
                          ``"row_index"`` (dataset row index)
    :param sample: if given, draw random sample of size `sample` from all text data
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks in texts
    :param pandas_read_opts: additional arguments passed to :func:`pandas.read_csv` or :func:`pandas.read_excel`
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """

    logger.debug('reading tabular file(s)')
    new_docs = _load_text_from_tabular_files(files,
                                             id_column=id_column,
                                             text_column=text_column,
                                             existing_docs=set(docs.keys()),
                                             prepend_columns=prepend_columns,
                                             encoding=encoding,
                                             doc_label_fmt=doc_label_fmt,
                                             force_unix_linebreaks=force_unix_linebreaks,
                                             pandas_read_opts=pandas_read_opts)

    if sample is not None:
        logger.info(f'sampling {sample} documents(s) out of {len(new_docs)}')
        new_docs = sample_dict(new_docs, n=sample)

    logger.info(f'adding text from {len(new_docs)} documents(s)')
    docs.update(new_docs)


@corpus_func_inplace_opt
def corpus_add_zip(docs: Corpus, zipfile: str, valid_extensions: Collection[str] = ('txt', 'csv', 'xls', 'xlsx'),
                   encoding: str = 'utf8', doc_label_fmt_txt: str ='{path}-{basename}', doc_label_path_join: str = '_',
                   doc_label_fmt_tabular: str = '{basename}-{id}',
                   sample: Optional[int] = None,
                   force_unix_linebreaks: bool = True,
                   add_files_opts: Optional[Dict[str, Any]] = None,
                   add_tabular_opts: Optional[Dict[str, Any]] = None,
                   inplace: bool = True) -> Optional[Corpus]:
    """
    Add documents from a ZIP file. The ZIP file may include documents with extensions listed in `valid_extensions`.

    For file extensions 'csv', 'xls' or 'xlsx' :func:`~tmtoolkit.corpus.corpus_add_tabular()` will be called. Make
    sure to pass at least the parameters `id_column` and `text_column` via `add_tabular_opts` if your ZIP contains
    such files.

    For all other file extensions :func:`~tmtoolkit.corpus.corpus_add_files()` will be called.

    :param docs: a Corpus object
    :param zipfile: path to ZIP file to be loaded
    :param valid_extensions: list of valid file extensions of ZIP file members; all other members will be ignored
    :param encoding: character encoding of the files
    :param doc_label_fmt_txt: document label format for non-tabular files; string with placeholders ``"path"``,
                              ``"basename"``, ``"ext"``
    :param doc_label_path_join: string with which to join the components of the file paths
    :param doc_label_fmt_tabular: document label format string for tabular files; placeholders ``"basename"``,
                                  ``"id"`` (document ID), and ``"row_index"`` (dataset row index)
    :param sample: if given, draw random sample of size `sample` from all text data
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks in texts
    :param add_files_opts: additional arguments passed to :func:`~tmtoolkit.corpus.corpus_add_files()`
    :param add_tabular_opts: additional arguments passed to :func:`~tmtoolkit.corpus.corpus_add_tabular()`
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    common_kwargs = dict(encoding=encoding, force_unix_linebreaks=force_unix_linebreaks)

    if add_files_opts is None:
        add_files_opts = {}
    else:
        add_files_opts = add_files_opts.copy()
    add_files_opts.update(common_kwargs)

    if add_tabular_opts is None:
        add_tabular_opts = {}
    else:
        add_tabular_opts = add_tabular_opts.copy()
    add_tabular_opts.update(common_kwargs)

    # sampling is handled *after* loading all data from the zip file in a separate step; this is necessary
    # because we don't know in advance how many documents we get from possible tabular files in the zip
    if 'sample' in add_files_opts:
        del add_files_opts['sample']
    if 'sample' in add_tabular_opts:
        del add_tabular_opts['sample']

    tmpdir = mkdtemp()

    # open the zip and iterate through its files
    logger.debug('extracting data from ZIP file')
    with ZipFile(zipfile) as zipobj:
        new_docs = {}
        for member in zipobj.namelist():
            # extract the components of the file path
            path_parts = path_split(member)

            if not path_parts:
                continue

            dirs, fname = path_parts[:-1], path_parts[-1]

            basename, ext = os.path.splitext(fname)
            basename = basename.strip()

            if ext:
                ext = ext[1:]

            if ext in valid_extensions:
                # extract to temp. location
                tmpfile = zipobj.extract(member, tmpdir)

                if ext in {'csv', 'xls', 'xlsx'}:   # this is a tabular file
                    new_docs.update(_load_text_from_tabular_files(tmpfile, doc_label_fmt=doc_label_fmt_tabular,
                                                                  **add_tabular_opts))
                else:  # otherwise it must be a text file
                    doclabel = doc_label_fmt_txt.format(path=doc_label_path_join.join(dirs),
                                                        basename=basename,
                                                        ext=ext)

                    if doclabel.startswith('-'):
                        doclabel = doclabel[1:]

                    new_docs.update(_load_text_from_files([tmpfile], {tmpfile: doclabel}, **add_files_opts))

        # apply sampling here after loading all data
        if sample is not None:
            logger.info(f'sampling {sample} documents(s) out of {len(new_docs)}')
            new_docs = sample_dict(new_docs, n=sample)

        logger.info(f'adding text from {len(new_docs)} documents(s)')
        docs.update(new_docs)


def save_corpus_to_picklefile(docs: Corpus, picklefile: str) -> None:
    """
    Serialize Corpus `docs` and save to Python pickle file `picklefile`.

    .. seealso:: Use :func:`load_corpus_from_picklefile` to load the Corpus object from a pickle file.

    :param docs: a Corpus object
    :param picklefile: path to pickle file
    """
    serdata = serialize_corpus(docs, deepcopy_attrs=False)
    logger.debug('storing serialized data')
    pickle_data(serdata, picklefile)


def load_corpus_from_picklefile(picklefile: str) -> Corpus:
    """
    Load and deserialize a stored Corpus object from the Python pickle file `picklefile`.

    .. seealso:: Use :func:`save_corpus_to_picklefile` to save a Corpus object to a pickle file.

    .. warning:: Python pickle files may contain malicious code. You should only load pickle files from trusted sources.

    :param picklefile: path to pickle file
    :return: a Corpus object
    """
    logger.info('loading serialized data')
    serdata = unpickle_file(picklefile)
    return deserialize_corpus(serdata)


def load_corpus_from_tokens(tokens: Dict[str, Any],
                            sentences: bool = False,
                            doc_attr: Dict[str, Any] = None,
                            token_attr: Dict[str, Any] = None,
                            **corpus_opt) -> Corpus:
    """
    Create a :class:`~tmtoolkit.corpus.Corpus` object from a dict of tokens (optionally along with document/token
    attributes) as may be returned from :func:`doc_tokens`.

    :param tokens: dict mapping document labels to tokens (optionally along with document/token attributes)
    :param sentences: if True, `tokens` are assumed to contain another level that indicates the sentences (as from
                      :func:`doc_tokens` with ``sentences=True``)
    :param doc_attr: document attributes with their respective default values
    :param token_attr: token attributes with their respective default values
    :param corpus_opt: arguments passed to :meth:`~tmtoolkit.corpus.Corpus.__init__`; shall not contain ``docs``
                       argument; at least ``language``, ``language_model`` or ``spacy_instance`` should be given
    :return: a Corpus object
    """
    if 'docs' in corpus_opt:
        raise ValueError('`docs` parameter is obsolete when initializing a Corpus with this function')

    corp = Corpus(**corpus_opt)

    logger.debug('creating new documents')
    newdocs = {}
    for lbl, tokattr in tokens.items():
        newdocs[lbl] = document_from_attrs(corp.bimaps, corp.nlp.vocab, lbl, tokattr, sentences=sentences,
                                           doc_attr_names=set(doc_attr.keys()) if doc_attr else None,
                                           token_attr_names=set(token_attr.keys()) if token_attr else None)

    if doc_attr:
        corp._doc_attrs_defaults.update(doc_attr)
    if token_attr:
        corp._token_attrs_defaults.update(token_attr)

    logger.info(f'adding {len(newdocs)} new documents')
    corp.update(newdocs)

    return corp


def load_corpus_from_tokens_table(tokens: pd.DataFrame,
                                  doc_attr: Dict[str, Any] = None,
                                  token_attr: Dict[str, Any] = None,
                                  **corpus_kwargs) -> Corpus:
    """
    Create a :class:`~tmtoolkit.corpus.Corpus` object from a dataframe as may be returned from :func:`tokens_table`.

    :param tokens: a dataframe with tokens, optionally along with document/token attributes
    :param doc_attr: optional dict mapping document attribute names to default values
    :param token_attr: optional dict mapping token attribute names to default values
    :param corpus_kwargs: arguments passed to :meth:`~tmtoolkit.corpus.Corpus.__init__`; shall not contain ``docs``
                          argument
    :return: a Corpus object
    """
    if 'docs' in corpus_kwargs:
        raise ValueError('`docs` parameter is obsolete when initializing a Corpus with this function')

    req_columns = {'doc', 'position', 'token', 'whitespace'}
    if not req_columns.issubset(set(tokens.columns)):
        raise ValueError(f'`tokens` dataframe must at least contain the following columns: {req_columns}')

    logger.debug('preparing tokens data from table')
    tokens_dict = {}
    doc_attr_w_unknown_defaults = {}
    token_attr_w_unknown_defaults = {}
    for lbl in tokens['doc'].unique():      # TODO: could make this faster
        doc_df = tokens.loc[tokens['doc'] == lbl, :]

        colnames = doc_df.columns.tolist()
        colnames.remove('doc')
        colnames.remove('position')

        doc_attr_w_unknown_defaults.update({c: None for c in colnames[:colnames.index('token')]})
        token_attr_w_unknown_defaults.update({c: None for c in colnames[colnames.index('token')+1:]})

        tokens_dict[lbl] = {col: doc_df[col].to_list() for col in colnames}

    if doc_attr:
        doc_attr_w_unknown_defaults.update(doc_attr)
    if token_attr:
        token_attr_w_unknown_defaults.update(token_attr)

    doc_attr = {k: v for k, v in doc_attr_w_unknown_defaults.items() if k != 'sent'}
    token_attr = {k: v for k, v in token_attr_w_unknown_defaults.items() if k not in TOKENMAT_ATTRS}

    return load_corpus_from_tokens(tokens_dict,
                                   sentences=False,
                                   doc_attr=doc_attr,
                                   token_attr=token_attr,
                                   **corpus_kwargs)


def serialize_corpus(docs: Corpus, deepcopy_attrs: bool = True) -> Dict[str, Any]:
    """
    Serialize a Corpus object to a dict. The inverse operation is implemented in :func:`deserialize_corpus`.

    :param docs: a Corpus object
    :param deepcopy_attrs: apply *deep* copy to all attributes
    :return: Corpus data serialized as dict
    """
    if logger.isEnabledFor(logging.INFO):
        logger.info(f'serializing Corpus with {len(docs)} documents')
    return docs._serialize(deepcopy_attrs=deepcopy_attrs, store_nlp_instance_pointer=False)


def deserialize_corpus(serialized_corpus_data: dict) -> Corpus:
    """
    Deserialize a Corpus object from a dict. The inverse operation is implemented in :func:`serialize_corpus`.

    :param serialized_corpus_data: Corpus data serialized as dict
    :return: a Corpus object
    """
    if logger.isEnabledFor(logging.INFO):
        logger.info(f'deserializing Corpus with {len(serialized_corpus_data["docs_data"])} documents')
    return Corpus._deserialize(serialized_corpus_data)


#%% Corpus functions that modify corpus data: document / token attribute handling


@corpus_func_inplace_opt
def set_document_attr(docs: Corpus, /, attrname: str, data: Dict[str, Any], default: Optional[Any] = None,
                      inplace: bool = True) \
        -> Optional[Corpus]:
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
    if attrname in docs.token_attrs or attrname in TOKENMAT_ATTRS:
        raise ValueError(f'attribute name "{attrname}" is already used as token attribute')

    logger.debug('setting document attribute')
    for lbl, d in docs.items():
        d.doc_attrs[attrname] = data.get(lbl, default)

    if attrname not in {'label', 'has_sents'}:               # set Corpus-specific default
        docs._doc_attrs_defaults[attrname] = default


@corpus_func_inplace_opt
def remove_document_attr(docs: Corpus, /, attrname: str, inplace: bool = True) -> Optional[Corpus]:
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

    logger.debug('removing document attribute')
    for d in docs.values():
        try:
            del d.doc_attrs[attrname]
        except KeyError: pass

    del docs._doc_attrs_defaults[attrname]


@corpus_func_inplace_opt
def set_token_attr(docs: Corpus, /, attrname: str, data: Dict[str, Any], default: Optional[Any] = None,
                   per_token_occurrence: bool = True, inplace: bool = True) -> Optional[Corpus]:
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
    if attrname in TOKENMAT_ATTRS:
        raise ValueError(f'cannot set attribute with protected name "{attrname}"')

    if attrname in docs.doc_attrs:
        raise ValueError(f'attribute name "{attrname}" is already used as document attribute')

    if per_token_occurrence:
        # convert data token string keys to token hashes
        data = {hash_string(k): v for k, v in data.items()}

    logger.debug('getting token hashes')
    docs_hashes = doc_tokens(docs, tokens_as_hashes=True, as_arrays=True)

    logger.debug('setting token attributes')
    for lbl, tok_hashes in docs_hashes.items():
        if per_token_occurrence:
            # match token occurrence with token's attribute value from `data`
            attrvalues = np.array([data.get(h, default) for h in tok_hashes])
        else:
            # set the token attributes for the whole document
            if lbl not in data.keys():   # if not attribute data for this document, repeat default values
                attrvalues = np.repeat(default, len(tok_hashes))
            else:
                attrvalues = data[lbl]

            # convert to array
            if isinstance(attrvalues, (list, tuple)):
                attrvalues = np.array(attrvalues)
            elif not isinstance(attrvalues, np.ndarray):
                raise ValueError(f'token attributes for document "{lbl}" are neither tuple, list nor NumPy array')
        # set the token attributes
        docs[lbl][attrname] = attrvalues

    docs._token_attrs_defaults[attrname] = default


@corpus_func_inplace_opt
def remove_token_attr(docs: Corpus, /, attrname: str, inplace: bool = True) -> Optional[Corpus]:
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

    # remove respective attribute in each document
    logger.debug('removing token attributes')
    for d in docs.values():
        try:
            del d[attrname]
        except KeyError: pass

    # remove custom token attributes entry
    del docs._token_attrs_defaults[attrname]


#%% Corpus functions that modify corpus data: token transformations


def corpus_retokenize(docs: Corpus, collapse: Optional[str] = ' ', inplace: bool = True) -> Optional[Corpus]:
    """
    Parse the corpus again using the current – possibly modified – tokens, but the same NLP pipeline as before.

    .. note:: This function is useful when you modified the corpus' tokens, e.g. by removing punctuation characters or
              transforming to lower-case characters, which has influence on token attributes like POS tags when parsing
              the corpus again.

    :param docs: a Corpus object
    :param collapse: if None, use whitespace token attribute for collapsing tokens, otherwise use custom string
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    logger.info('generating document texts')
    texts = doc_texts(docs, collapse=collapse)

    if inplace:
        docs._docs = {}
    else:
        logger.info('making bare corpus copy')
        docs = Corpus._deserialize(docs._serialize(deepcopy_attrs=False, store_nlp_instance_pointer=True,
                                                   documents=False))

    logger.info('re-parsing document texts')
    if docs.max_workers <= 1:
        logger.info('using serial processing')
    else:
        logger.info(f'using parallel processing with {docs.max_workers} workers')

    docs.bimaps = {}
    docs._init_bimaps()
    docs._init_docs(texts)
    docs._update_bimaps()
    docs._update_workers_docs()

    if inplace:
        return None
    else:
        return docs


@corpus_func_inplace_opt
def transform_tokens(docs: Corpus, /, func: Callable, select: Optional[Union[str, Collection[str]]] = None,
                     vocab: Optional[Set[Union[int]]] = None, inplace: bool = True, **kwargs) -> Optional[Corpus]:
    """
    Transform tokens in all documents by applying function `func` to each document's tokens individually.

    :param docs: a Corpus object
    :param func: a function to apply to all documents' tokens; it must accept a single token string and vice-versa
                 return single token string
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param vocab: optional vocabulary of token *hashes* (set of integers), which should be considered for
                  transformation; if this is not given, the full vocabulary of `docs` will be generated
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :param kwargs: additional arguments passed to `func`
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """

    select = _single_str_to_set(select, check_docs=docs)

    # get unique token types as hashes
    if vocab is None:
        vocab = vocabulary(docs, select=select, tokens_as_hashes=True, force_unigrams=True, convert_uint64hashes=False)
    hash2token = docs.bimaps['token']

    # apply transformations to tokens in vocabulary
    logger.debug('applying transformation function to vocabulary')
    replacements = {}   # original token hash ->  new token hash for transformed tokens
    for t_hash in vocab:    # iterate through token type hashes
        # get string representation for hash and transform it
        t_transformed = func(hash2token[t_hash], **kwargs)
        # get hash for transformed token type string
        t_hash_transformed = hash_string(t_transformed)
        # if hashes differ (i.e. transformation changed the string), record the hashes
        if t_hash != t_hash_transformed:
            hash2token.forceput(t_hash_transformed, t_transformed)
            if select is None and t_hash in hash2token:   # remove the old hash only if applying transform to all docs.
                del hash2token[t_hash]
            replacements[t_hash] = t_hash_transformed

    # replace token hashes in token matrix for each document
    logger.info(f'replacing {len(replacements)} token hashes')
    for lbl, d in docs.items():
        if select is None or lbl in select:
            d.tokenmat[:, TOKINDEX] = np.array([replacements.get(h, h)
                                                for h in d.tokenmat[:, TOKINDEX]], dtype='uint64')

    if select is not None:
        docs._update_bimaps(which_attrs='token')


def to_lowercase(docs: Corpus, /, select: Optional[Union[str, Collection[str]]] = None, inplace: bool = True) \
        -> Optional[Corpus]:
    """
    Convert all tokens to lower-case form.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return transform_tokens(docs, str.lower, select=select, inplace=inplace)


def to_uppercase(docs: Corpus, /, select: Optional[Union[str, Collection[str]]] = None, inplace: bool = True) \
        -> Optional[Corpus]:
    """
    Convert all tokens to upper-case form.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return transform_tokens(docs, str.upper, select=select, inplace=inplace)


def remove_chars(docs: Corpus, /, chars: Iterable[str], select: Optional[Union[str, Collection[str]]] = None,
                 inplace: bool = True) -> Optional[Corpus]:
    """
    Remove all characters listed in `chars` from all tokens.

    :param docs: a Corpus object
    :param chars: list of characters to remove; each element in the list should be a single character
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    del_chars = str.maketrans('', '', ''.join(chars))
    return transform_tokens(docs, lambda t: t.translate(del_chars), select=select, inplace=inplace)


def remove_punctuation(docs: Corpus, /, select: Optional[Union[str, Collection[str]]] = None,
                       inplace: bool = True) -> Optional[Corpus]:
    """
    Removes punctuation characters *in* tokens, i.e. ``['a', '.', 'f;o;o']`` becomes ``['a', '', 'foo']``.

    If you want to remove punctuation *tokens*, use :func:`~filter_clean_tokens`.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return remove_chars(docs, docs.punctuation, select=select, inplace=inplace)


def normalize_unicode(docs: Corpus, /, select: Optional[Union[str, Collection[str]]] = None,
                      form: str = 'NFC', inplace: bool = True) -> Optional[Corpus]:
    """
    Normalize unicode characters according to `form`.

    This function only *normalizes* unicode characters in the tokens of `docs` to the form
    specified by `form`. If you want to *simplify* the characters, i.e. remove diacritics,
    underlines and other marks, use :func:`~simplify_unicode` instead.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param form: normal form (see https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return transform_tokens(docs, lambda t: unicodedata.normalize(form, t), select=select, inplace=inplace)


def simplify_unicode(docs: Corpus, /, select: Optional[Union[str, Collection[str]]] = None,
                     method: str = 'icu', ascii_encoding_errors: str = 'ignore',
                     inplace: bool = True) -> Optional[Corpus]:
    """
    *Simplify* unicode characters in the tokens of `docs`, i.e. remove diacritics, underlines and
    other marks. Requires `PyICU <https://pypi.org/project/PyICU/>`_ to be installed when using
    ``method="icu"``.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param method: either ``"icu"`` which uses `PyICU <https://pypi.org/project/PyICU/>`_ for "proper"
                   simplification or ``"ascii"`` which tries to encode the characters as ASCII; the latter
                   is not recommended and will simply dismiss any characters that cannot be converted
                   to ASCII after decomposition
    :param ascii_encoding_errors: only used if `method` is ``"ascii"``; what to do when a character cannot be
                                  encoded as ASCII character; can be either ``"ignore"`` (default – replace by empty
                                  character), ``"replace"`` (replace by ``"???"``) or ``"strict"`` (raise a
                                  ``UnicodeEncodeError``)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """

    fn = partial(simplify_unicode_chars, method=method, ascii_encoding_errors=ascii_encoding_errors)
    return transform_tokens(docs, fn, select=select, inplace=inplace)


def numbers_to_magnitudes(docs: Corpus, /, select: Optional[Union[str, Collection[str]]] = None,
                          char: str = '0', firstchar: str = '1', below_one: str = '0',
                          zero: str = '0', drop_sign: bool = False,
                          decimal_sep: str = '.', thousands_sep: str = ',',
                          value_on_conversion_error: Optional[str] = None,
                          inplace: bool = True) -> Optional[Corpus]:
    """
    Convert each string token in `docs` that represents a number (e.g. "13", "1.3" or "-1313") to a string token that
    represents the magnitude of that number by repeating `char` ("00", "0", "0000" for the mentioned examples). A
    different first character can be set via `firstchar`.

    .. seealso:: :func:`~tmtoolkit.tokenseq.numbertoken_to_magnitude`

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param char: character string used to represent single orders of magnitude
    :param firstchar: special character used for first character in the output
    :param below_one: special character used for numbers with absolute value below 1 (would otherwise return `''`)
    :param zero: if `numbertoken` evaluates to zero, return this string
    :param drop_sign: if True, drop the sign in number `numbertoken`, i.e. use absolute value
    :param decimal_sep: decimal separator used in `numbertoken`; this is language-specific
    :param thousands_sep: thousands separator used in `numbertoken`; this is language-specific
    :param value_on_conversion_error: determines placeholder when the input token cannot be converted to a number; if
                                      `value_on_conversion_error` is None, use the input token unchanged, otherwise use
                                      `value_on_conversion_error`
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    # get hashes of those tokens that qualify as "number-like"
    vocab = set()
    select = _single_str_to_set(select)
    logger.debug('getting tokens')
    tokens = doc_tokens(docs, select=select, only_non_empty=True, tokens_as_hashes=True, with_attr='like_num',
                        force_unigrams=True, as_arrays=True).values()
    logger.debug('storing number-like tokens in vocab')
    for tok in tokens:
        vocab.update(set(tok['token'][tok['like_num'].astype('bool')]))

    # apply `numbertoken_to_magnitude` function to all these number-like tokens
    fn = partial(numbertoken_to_magnitude, char=char, firstchar=firstchar, below_one=below_one, zero=zero,
                 decimal_sep=decimal_sep, thousands_sep=thousands_sep, drop_sign=drop_sign,
                 value_on_conversion_error=value_on_conversion_error)
    return transform_tokens(docs, fn, select=select, vocab=vocab, inplace=inplace)


@corpus_func_inplace_opt
def lemmatize(docs: Corpus, /, select: Optional[Union[str, Collection[str]]] = None,
              inplace: bool = True) -> Optional[Corpus]:
    """
    Lemmatize tokens, i.e. set the lemmata as tokens so that all further processing will happen
    using the lemmatized tokens.

    :param docs: a Corpus object
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """

    select = _single_str_to_set(select, check_docs=docs)
    logger.debug("copying lemma hashes to token column in each document's hash matrix")
    for lbl, d in docs.items():
        if select is None or lbl in select:
            d.tokenmat[:, TOKINDEX] = d.tokenmat[:, d.tokenmat_attrs.index('lemma')]

    if select is None:
        # all docs. were selected -> copy lemma bimap to token bimap
        logger.debug("copy lemma bimap to token bimap")
        docs.bimaps['token'] = docs.bimaps['lemma'].copy()
    else:
        # only subset was selected -> use hashes from lemma also in token map
        logger.debug("update token bimap with lemma bimap entries")
        docs.bimaps['token'].update(docs.bimaps['lemma'])


@corpus_func_update_bimaps(which_attrs='token')
@corpus_func_inplace_opt
def join_collocations_by_patterns(docs: Corpus, /, patterns: Sequence[str],
                                  select: Optional[Union[str, Collection[str]]] = None, glue: str = '_',
                                  match_type: str = 'exact', ignore_case=False, glob_method: str = 'match',
                                  return_joint_tokens: bool = False, inplace: bool = True) \
        -> Optional[Union[Corpus, Tuple[Corpus, Set[str]]]]:
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
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
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

    hash2token = docs.bimaps['token']

    @parallelexec(merge_dicts)
    def _join_colloc(chunk):
        res = {}
        for lbl, tokenmat in chunk.items():
            # convert token hashes to strings
            tok_strs = [hash2token[h] for h in tokenmat[:, 1]]

            # get the subsequent matches as boolean mask arrays
            matches = token_match_subsequent(patterns, tok_strs, match_type=match_type, ignore_case=ignore_case,
                                             glob_method=glob_method)

            # join the matched subsequent tokens; `return_mask=True` makes sure that we only get the newly
            # generated joint tokens together with an array to mask all but the first token of the subsequent tokens
            new_tok, mask = token_join_subsequent(tok_strs, matches, glue=glue, return_mask=True)

            res[lbl] = _apply_collocations(tokenmat, new_tok, mask, hash2token=hash2token, tokens_as_hashes=False,
                                           glue=glue, return_joint_tokens=return_joint_tokens)

        return res

    select = _single_str_to_set(select, check_docs=docs)
    logger.debug('getting token matrices')
    doc_tokmats = {lbl: d.tokenmat for lbl, d in docs.items() if select is None or lbl in select}

    logger.debug('joining token collocations')
    res = _join_colloc(_paralleltask(docs, doc_tokmats))

    if return_joint_tokens:
        joint_tokens = set()

    logger.debug('applying new token hash matrices')
    for lbl, colloc_res in res.items():
        if return_joint_tokens:
            tokenmat, doc_hash2token_upd, doc_joint_tok = colloc_res
            joint_tokens.update(doc_joint_tok)
        else:
            tokenmat, doc_hash2token_upd = colloc_res

        docs[lbl].tokenmat = tokenmat
        hash2token.forceupdate(doc_hash2token_upd)

    if return_joint_tokens:
        return joint_tokens


@corpus_func_update_bimaps(which_attrs='token')
@corpus_func_inplace_opt
def join_collocations_by_statistic(docs: Corpus, /, threshold: float,
                                   select: Optional[Union[str, Collection[str]]] = None, glue: str = '_',
                                   min_count: int = 1, embed_tokens_min_docfreq: Optional[Union[int, float]] = None,
                                   embed_tokens_set: Optional[Set] = None,
                                   statistic: Callable = npmi, return_joint_tokens: bool = False,
                                   inplace: bool = True, **statistic_kwargs) \
        -> Optional[Union[Corpus, Tuple[Corpus, Set[str]]]]:
    """
    Join subsequent tokens by token collocation statistic as can be computed by :func:`corpus_collocations`.

    :param docs: a Corpus object
    :param threshold: minimum statistic value for a collocation to enter the results
    :param select: if not None, this can be a single string or a sequence of strings specifying a subset of `docs`
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

    # get tokens as hashes
    logger.debug('getting flattened tokens')
    tok_flat = corpus_tokens_flattened(docs, select=select, sentences=True, tokens_as_hashes=True)
    logger.debug('getting flattened tokens')
    vocab_counts = vocabulary_counts(docs, select=select, tokens_as_hashes=True)

    # generate ``embed_tokens`` set as used in :func:`~tmtookit.tokenseq.token_collocations`
    logger.debug('creating embed tokens')
    embed_tokens = _create_embed_tokens_for_collocations(docs, embed_tokens_min_docfreq, embed_tokens_set,
                                                         tokens_as_hashes=True)

    # identify collocations
    logger.debug('identifying collocations')
    colloc = token_collocations(tok_flat, threshold=threshold, min_count=min_count, embed_tokens=embed_tokens,
                                vocab_counts=vocab_counts, statistic=statistic, return_statistic=False,
                                rank=None, tokens_as_hashes=True, **statistic_kwargs)

    hash2token = docs.bimaps['token']

    @parallelexec(merge_dicts)
    def _join_colloc(chunk):
        res = {}
        for lbl, tokenmat in chunk.items():
            # get the subsequent matches of the collocation token hashes as boolean mask arrays
            matches = []
            for hashes in colloc:
                matches.extend(token_match_subsequent(hashes, tokenmat[:, 1], match_type='exact'))

            # join the matched subsequent tokens; `return_mask=True` makes sure that we only get the newly
            # generated joint tokens together with an array to mask all but the first token of the subsequent tokens
            # `glue=None` makes sure that the token hashes are not joint
            new_tok, mask = token_join_subsequent(tokenmat[:, 1], matches, glue=None,  tokens_dtype='uint64',
                                                  return_mask=True)

            res[lbl] = _apply_collocations(tokenmat, new_tok, mask, hash2token=hash2token, tokens_as_hashes=True,
                                           glue=glue, return_joint_tokens=return_joint_tokens)

        return res

    # join collocations
    logger.debug('getting token matrices')
    select = _single_str_to_set(select)
    doc_tokmats = {lbl: d.tokenmat for lbl, d in docs.items() if select is None or lbl in select}

    logger.debug('joining token collocations')
    res = _join_colloc(_paralleltask(docs, doc_tokmats))

    if return_joint_tokens:
        joint_tokens = set()

    logger.debug('applying new token hash matrices')
    for lbl, colloc_res in res.items():
        if return_joint_tokens:
            tokenmat, doc_hash2token_upd, doc_joint_tok = colloc_res
            joint_tokens.update(doc_joint_tok)
        else:
            tokenmat, doc_hash2token_upd = colloc_res

        docs[lbl].tokenmat = tokenmat
        hash2token.forceupdate(doc_hash2token_upd)

    if return_joint_tokens:
        return joint_tokens


#%% Corpus functions that modify corpus data: filtering / KWIC

@corpus_func_update_bimaps()
@corpus_func_inplace_opt
def filter_tokens_by_mask(docs: Corpus, /, mask: Dict[str, Union[List[bool], np.ndarray]], inverse: bool = False,
                          inplace: bool = True) -> Optional[Corpus]:
    """
    Filter (i.e. remove) tokens according to a boolean mask specified by `mask`.

    .. seealso:: :func:`remove_tokens_by_mask`

    :param docs: a Corpus object
    :param mask: dict mapping document label to boolean list or NumPy array where ``False`` means "remove" and
                 ``True`` means "keep" for the respective token; the length of the mask must equal the number of tokens
                 in the document
    :param inverse: inverse the truth values in the mask arrays
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """

    if logger.isEnabledFor(logging.INFO):
        n_tok_before = corpus_num_tokens(docs)
    else:
        n_tok_before = None

    logger.debug('filtering tokens by mask')
    for lbl, m in mask.items():
        if lbl not in docs.keys():
            raise ValueError(f'document "{lbl}" does not exist in Corpus object `docs` - '
                             f'cannot set token mask')

        d = docs[lbl]

        if len(m) != len(d):
            raise ValueError(f'length of provided mask for document "{lbl}" does not match length of the document')

        if not isinstance(m, np.ndarray):
            m = np.array(m, dtype=bool)
        elif not np.issubdtype(m.dtype, bool):
            m = m.astype(bool)

        if inverse:
            m = ~m

        d.tokenmat = d.tokenmat[m, :]

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'filtered tokens by mask: num. tokens was {n_tok_before} and is now {corpus_num_tokens(docs)}')


def remove_tokens_by_mask(docs: Corpus, /, mask: Dict[str, Union[List[bool], np.ndarray]], inplace: bool = True) \
        -> Optional[Corpus]:
    """
    Remove tokens according to a boolean mask specified by `mask`.

    .. seealso:: :func:`filter_tokens_by_mask`

    :param docs: a Corpus object
    :param mask: dict mapping document label to boolean list or NumPy array where ``False`` means "keep" and
                 ``True`` means "remove" for the respective token; the length of the mask must equal the number of
                 tokens in the document
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_tokens_by_mask(docs, mask=mask, inverse=True, inplace=inplace)


def filter_tokens(docs: Corpus, /, search_tokens: Any, by_attr: Optional[str] = None,
                  match_type: str = 'exact', ignore_case: bool = False,
                  glob_method: str = 'match', inverse: bool = False, inplace: bool = True) -> Optional[Corpus]:
    """
    Filter tokens according to search pattern(s) `search_tokens` and several matching options. Only those tokens
    are retained that match the search criteria unless you set ``inverse=True``, which will *remove* all tokens
    that match the search criteria (which is the same as calling :func:`remove_tokens`).

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

    by_attr = by_attr or 'token'

    logger.debug('creating tokens filter mask by pattern search')
    try:
        matchdata = _match_against(docs, by_attr, default=docs.custom_token_attrs_defaults.get(by_attr, None))
        masks = _filter_tokens(_paralleltask(docs, matchdata))
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    return filter_tokens_by_mask(docs, masks, inverse=inverse, inplace=inplace)


def remove_tokens(docs: Corpus, /, search_tokens: Any, by_attr: Optional[str] = None,
                  match_type: str = 'exact', ignore_case: bool = False,
                  glob_method: str = 'match', inplace: bool = True) -> Optional[Corpus]:
    """
    This is a shortcut for the :func:`filter_tokens` method with ``inverse=True``, i.e. *remove* all tokens that match
    the search criteria).

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
                         by_attr=by_attr, inverse=True, inplace=inplace)


def filter_for_pos(docs: Corpus, /, search_pos: Union[str, Collection[str]], simplify_pos: bool = True,
                   tagset:str = 'ud', inverse: bool = False, inplace: bool = True) -> Optional[Corpus]:
    """
    Filter tokens for a specific POS tag (if `required_pos` is a string) or several POS tags (if `required_pos`
    is a list/tuple/set of strings). The POS tag depends on the tagset used during tagging. See
    https://spacy.io/api/annotation#pos-tagging for a general overview on POS tags in SpaCy and refer to the
    documentation of your language model for specific tags.

    If `simplify_pos` is True, then the tags are matched to the following simplified forms:

    * ``'N'`` for nouns
    * ``'V'`` for verbs
    * ``'ADJ'`` for adjectives
    * ``'ADV'`` for adverbs
    * ``None`` for all other

    :param docs: a Corpus object
    :param search_pos: single string or list of strings with POS tag(s) used for filtering
    :param simplify_pos: if True, simplify POS tags in documents to forms shown above before matching
    :param tagset: tagset used for `pos`; can be ``'wn'`` (WordNet), ``'penn'`` (Penn tagset)
                   or ``'ud'`` (universal dependencies – default)
    :param inverse: inverse the matching results, i.e. *remove* tokens that match the POS tag
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    @parallelexec(collect_fn=merge_dicts)
    def _filter_pos(chunk):
        if simplify_pos:
            chunk = {lbl: list(map(lambda x: simplified_pos(x, tagset=tagset), tok_pos))
                     for lbl, tok_pos in chunk.items()}

        return _token_pattern_matches(chunk, search_pos)

    logger.debug('creating tokens filter mask by POS matching')
    matchdata = _match_against(docs, 'pos')
    masks = _filter_pos(_paralleltask(docs, matchdata))

    return filter_tokens_by_mask(docs, masks, inverse=inverse, inplace=inplace)


def filter_tokens_by_doc_frequency(docs: Corpus, /, which: str, df_threshold: Union[int, float],
                                   proportions: Proportion = Proportion.NO,
                                   return_filtered_tokens: bool = False,
                                   inverse: bool = False,
                                   inplace: bool = True) \
        -> Union[None, Corpus, Set[str], Tuple[Corpus, Set[str]]]:
    """
    Filter tokens according to their document frequency.

    :param docs: a Corpus object
    :param which: which threshold comparison to use: either ``'common'``, ``'>'``, ``'>='`` which means that tokens
                  with higher document freq. than (or equal to) `df_threshold` will be kept;
                  or ``'uncommon'``, ``'<'``, ``'<='`` which means that tokens with lower document freq. than
                  (or equal to) `df_threshold` will be kept
    :param df_threshold: document frequency threshold value
    :param proportions: controls whether document frequency threshold is given in (log) proportions rather than absolute
                        counts
    :param return_filtered_tokens: if True, additionally return set of filtered token types
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: depending on `return_filtered_tokens` and `inplace`: if both are True, returns only filtered token types;
             if `return_filtered_tokens` is True and `inplace` is False, returns tuple with modified copy of `docs` and
             filtered token types; if `return_filtered_tokens` is False returns either original Corpus object `docs` or
             a modified copy of it
    """
    comp = _comparison_operator_from_str(which, common_alias=True)

    logger.debug('creating tokens filter mask by applying document frequency threshold')
    toks = doc_tokens(docs)
    doc_freqs = doc_frequencies(docs, proportions=proportions)
    mask = {lbl: [comp(doc_freqs[t], df_threshold) for t in dtok] for lbl, dtok in toks.items()}

    if return_filtered_tokens:
        filt_tok = set(t for t, f in doc_freqs.items() if comp(f, df_threshold))
    else:
        filt_tok = set()

    res = filter_tokens_by_mask(docs, mask=mask, inverse=inverse, inplace=inplace)
    if return_filtered_tokens:
        if inplace:
            return filt_tok
        else:
            return res, filt_tok
    else:
        return res


def remove_common_tokens(docs: Corpus, /, df_threshold: Union[int, float] = 0.95,
                         proportions: Proportion = Proportion.YES,
                         inplace: bool = True) -> Optional[Corpus]:
    """
    Shortcut for :func:`filter_tokens_by_doc_frequency` for removing tokens *above* a certain  document frequency.

    :param docs: a Corpus object
    :param df_threshold: document frequency threshold value
    :param proportions: controls whether document frequency threshold is given in (log) proportions rather than absolute
                        counts
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_tokens_by_doc_frequency(docs, which='common', df_threshold=df_threshold, proportions=proportions,
                                          inverse=True, inplace=inplace)


def remove_uncommon_tokens(docs: Corpus, /, df_threshold: Union[int, float] = 0.05,
                           proportions: Proportion = Proportion.YES,
                           inplace: bool = True) -> Optional[Corpus]:
    """
    Shortcut for :func:`filter_tokens_by_doc_frequency` for removing tokens *below* a certain  document frequency.

    :param docs: a Corpus object
    :param df_threshold: document frequency threshold value
    :param proportions: controls whether document frequency threshold is given in (log) proportions rather than absolute
                        counts
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    return filter_tokens_by_doc_frequency(docs, which='uncommon', df_threshold=df_threshold, proportions=proportions,
                                          inverse=True, inplace=inplace)


@corpus_func_update_bimaps()
@corpus_func_inplace_opt
def filter_documents_by_mask(docs: Corpus, /, mask: Dict[str, bool], inverse: bool = False) -> Optional[Corpus]:
    """
    Filter documents by setting a mask.

    .. seealso:: :func:`remove_documents_by_mask`

    :param docs: a Corpus object
    :param mask: dict that maps document labels to document attribute value
    :param inverse: inverse the mask
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if logger.isEnabledFor(logging.INFO):
        n_docs_before = len(docs)
    else:
        n_docs_before = None

    logger.debug('filtering documents by mask')
    for lbl, m in mask.items():
        if inverse:
            m = not m

        if not m:
            del docs._docs[lbl]

    docs._update_workers_docs()

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'filtered documents by mask: number of documents was {n_docs_before} and is now {len(docs)}')


def remove_documents_by_mask(docs: Corpus, /, mask: Dict[str, bool], inplace: bool = True) -> Optional[Corpus]:
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


@tabular_result_option('doc', 'n_matches')
def find_documents(docs: Corpus, /, search_tokens: Any, by_attr: Optional[str] = None,
                   matches_threshold: int = 1, match_type: str = 'exact', ignore_case: bool = False,
                   glob_method: str = 'match', inverse_result: bool = False, inverse_matches: bool = False,
                   as_table: Union[bool, str] = False)\
        -> Union[Dict[str, int], pd.DataFrame]:
    """
    For each document, the number of token matches is counted and a dict or dataframe (if `as_table` is True) is
    returned with entries of document labels when the number of matches is at least `matches_threshold`.

    .. seealso:: :func:`filter_documents` which does that same but applies the matches to the corpus, creating a subset
                 of documents instead of only reporting the matches

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param by_attr: if not None, this should be an attribute name; this attribute data will then be
                    used for matching instead of the tokens in `docs`
    :param matches_threshold: number of matches required for filtering a document
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
    :param as_table: if True, return result as dataframe; if a string, sort dataframe by this column; if string prefixed
                 with "-", sort by this column in descending order
    :return: dict of number of matches per document label or dataframe if `as_table` is active
    """
    _check_filter_args(match_type=match_type, glob_method=glob_method)

    by_attr = by_attr or 'token'

    logger.debug('creating documents filter mask by pattern search')
    try:
        matchdata = _match_against(docs, by_attr, default=docs.custom_token_attrs_defaults.get(by_attr, None))
        docs_matches = _filter_documents(_paralleltask(docs, matchdata), search_tokens=search_tokens,
                                         match_type=match_type,
                                         ignore_case=ignore_case, glob_method=glob_method,
                                         inverse_matches=inverse_matches,
                                         matches_threshold=matches_threshold, inverse_result=not inverse_result,
                                         return_num_matches=True)
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    return docs_matches


def filter_documents(docs: Corpus, /, search_tokens: Any, by_attr: Optional[str] = None,
                     matches_threshold: int = 1, match_type: str = 'exact', ignore_case: bool = False,
                     glob_method: str = 'match', inverse_result: bool = False, inverse_matches: bool = False,
                     inplace: bool = True)\
        -> Optional[Corpus]:
    """
    This function is similar to :func:`filter_tokens` but applies at document level. For each document, the number of
    matches is counted. If it is at least `matches_threshold` the document is retained, otherwise it is removed.
    If `inverse_result` is True, then documents that meet the threshold are removed.

    .. seealso:: :func:`find_documents` which does that same but only reports the found documents;
                 :func:`remove_documents` which is the same as this function but with inversed result

    :param docs: a Corpus object
    :param search_tokens: single string or list of strings that specify the search pattern(s); when `match_type` is
                          ``'exact'``, `pattern` may be of any type that allows equality checking
    :param by_attr: if not None, this should be an attribute name; this attribute data will then be
                    used for matching instead of the tokens in `docs`
    :param matches_threshold: number of matches required for filtering a document
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

    by_attr = by_attr or 'token'

    logger.debug('creating documents filter mask by pattern search')
    try:
        matchdata = _match_against(docs, by_attr, default=docs.custom_token_attrs_defaults.get(by_attr, None))
        remove = _filter_documents(_paralleltask(docs, matchdata), search_tokens=search_tokens, match_type=match_type,
                                   ignore_case=ignore_case, glob_method=glob_method, inverse_matches=inverse_matches,
                                   matches_threshold=matches_threshold, inverse_result=inverse_result,
                                   return_num_matches=False)
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    return filter_documents_by_mask(docs, mask=dict(zip(remove, [False] * len(remove))), inplace=inplace)


def remove_documents(docs: Corpus, /, search_tokens: Any, by_attr: Optional[str] = None,
                     matches_threshold: int = 1, match_type: str = 'exact', ignore_case: bool = False,
                     glob_method: str = 'match', inverse_matches: bool = False, inplace: bool = True) \
        -> Optional[Corpus]:
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
                            inverse_matches=inverse_matches, inverse_result=True, inplace=inplace)


def filter_documents_by_docattr(docs: Corpus, /, search_tokens: Any, by_attr: str,
                                match_type: str = 'exact', ignore_case: bool = False, glob_method: str = 'match',
                                inverse: bool = False, inplace: bool = True) -> Optional[Corpus]:
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

    if by_attr not in docs.doc_attrs_defaults:
        raise ValueError(f'document attribute "{by_attr}" not defined in Corpus `docs`')

    logger.debug('creating documents filter mask by pattern search on document attributes')
    default = docs.doc_attrs_defaults[by_attr]
    attr_values = [d.doc_attrs.get(by_attr, default) for d in docs.values()]
    matches = token_match_multi_pattern(search_tokens, attr_values, match_type=match_type,
                                        ignore_case=ignore_case, glob_method=glob_method)
    return filter_documents_by_mask(docs, mask=dict(zip(docs.keys(), matches)), inverse=inverse, inplace=inplace)


def remove_documents_by_docattr(docs: Corpus, /, search_tokens: Any, by_attr: str,
                                match_type: str = 'exact', ignore_case: bool = False,
                                glob_method: str = 'match', inplace: bool = True) -> Optional[Corpus]:
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
                              ignore_case: bool = False, glob_method: str = 'match',
                              inverse: bool = False, inplace: bool = True) \
        -> Optional[Corpus]:
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
                              ignore_case: bool = False, glob_method: str = 'match', inplace: bool = True) \
        -> Optional[Corpus]:
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


def filter_documents_by_length(docs: Corpus, /, relation: str, threshold: int, inverse: bool = False,
                               inplace: bool = True) \
        -> Optional[Corpus]:
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

    logger.debug('creating documents filter mask by checking the document length')
    comp = _comparison_operator_from_str(relation, equal=True, whicharg='relation')
    mask = {lbl: comp(n, threshold) for lbl, n in doc_lengths(docs).items()}

    return filter_documents_by_mask(docs, mask=mask, inverse=inverse, inplace=inplace)


def remove_documents_by_length(docs: Corpus, /, relation: str, threshold: int, inplace: bool = True) -> Optional[Corpus]:
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


def filter_clean_tokens(docs: Corpus, /,
                        remove_punct: bool = True,
                        remove_stopwords: Union[bool, Iterable[str]] = True,
                        remove_empty: bool = True,
                        remove_shorter_than: Optional[int] = None,
                        remove_longer_than: Optional[int] = None,
                        remove_numbers: bool = False,
                        inplace: bool = True) -> Optional[Corpus]:
    """
    Filter tokens in `docs` to retain only a certain, configurable subset of tokens.

    :param docs: a Corpus object
    :param remove_punct: remove all tokens that are considered to be punctuation (``"."``, ``","``, ``";"`` etc.)
                         according to the ``is_punct`` attribute of the
                         `SpaCy Token <https://spacy.io/api/token#attributes>`_
    :param remove_stopwords: remove all tokens that are considered to be stopwords; if True, remove tokens according to
                             the ``is_stop`` attribute of the `SpaCy Token <https://spacy.io/api/token#attributes>`_;
                             if `remove_stopwords` is a set/tuple/list it defines the stopword list
    :param remove_empty: remove all empty (``""``) and whitespace-only string tokens
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

    # add stopwords
    if isinstance(remove_stopwords, (list, tuple, set)):
        tokens_to_remove = remove_stopwords
    else:
        tokens_to_remove = []

    # convert to hashes
    if tokens_to_remove:
        tokens_to_remove = np.unique(np.fromiter(map(hash_string, tokens_to_remove),
                                                 dtype='uint64', count=len(tokens_to_remove)))
    else:
        tokens_to_remove = None

    if remove_empty:
        vocab = vocabulary(docs)
        h_empty = [hash_string('')] + [hash_string(t) for t in vocab if PTTRN_WS.match(t)]

        if tokens_to_remove is None:
            tokens_to_remove = np.array(h_empty, dtype='uint64')
        else:
            tokens_to_remove = np.append(tokens_to_remove, h_empty)

    # function for parallel filtering: accepts a chunk of documents as dict
    # doc. label -> doc. data and returns a dict doc. label -> doc. filter mask
    @parallelexec(collect_fn=merge_dicts)
    def _filter_clean_tokens(chunk, docs_data_attrs):
        bool_cols = [i for i, a in enumerate(docs_data_attrs) if a in {'is_punct', 'like_num', 'is_stop'}]

        docs_mask = {}
        for lbl, tokenmat in chunk.items():
            mask = np.repeat(True, len(tokenmat))

            if remove_shorter_than is not None:
                mask &= (tokenmat[:, docs_data_attrs.index('token_lengths')] >= remove_shorter_than)

            if remove_longer_than is not None:
                mask &= (tokenmat[:, docs_data_attrs.index('token_lengths')] <= remove_longer_than)

            if bool_cols:
                mask &= ~np.sum(tokenmat[:, bool_cols].astype(bool), axis=1).astype(bool)

            if tokens_to_remove is not None:
                mask &= ~np.isin(tokenmat[:, docs_data_attrs.index('token')], tokens_to_remove)

            docs_mask[lbl] = mask

        return docs_mask

    # data preparation for parallel processing: create a dict `docs_data` with
    # doc. label -> doc. data that contains all necessary information for filtering
    # the document, depending on the filtering options
    docs_data_attrs = []

    if tokens_to_remove is not None and len(tokens_to_remove) > 0:
        docs_data_attrs.append('token')

    if remove_shorter_than is not None or remove_longer_than is not None:
        token_lengths = doc_token_lengths(docs)
        docs_data_attrs.append('token_lengths')
    else:
        token_lengths = None

    if remove_punct:
        docs_data_attrs.append('is_punct')
    if remove_numbers:
        docs_data_attrs.append('like_num')
    if remove_stopwords is True:
        docs_data_attrs.append('is_stop')

    if not docs_data_attrs:
        # nothing to do
        return

    logger.debug('creating tokens filter mask by applying cleaning methods')
    docs_data = {}
    for lbl, d in docs.items():
        attr_indices = [d.tokenmat_attrs.index(a) for a in docs_data_attrs if a != 'token_lengths']
        d_data = None

        if attr_indices:
            d_data = d.tokenmat[:, attr_indices]

        if 'token_lengths' in docs_data_attrs:
            d_toklen = np.array(token_lengths[lbl], dtype='uint64').reshape((len(token_lengths[lbl]), 1))

            if attr_indices:
                d_data = np.hstack((d_data, d_toklen))
            else:
                d_data = d_toklen

        docs_data[lbl] = np.array([], dtype='uint64') if d_data is None else d_data

    # run filtering in parallel
    try:
        # move 'token_lengths' to last position
        docs_data_attrs.pop(docs_data_attrs.index('token_lengths'))
        docs_data_attrs.append('token_lengths')
    except ValueError: pass

    new_masks = _filter_clean_tokens(_paralleltask(docs, docs_data), docs_data_attrs=docs_data_attrs)
    return filter_tokens_by_mask(docs, mask=new_masks, inplace=inplace)


def filter_tokens_with_kwic(docs: Corpus, /, search_tokens: Any,
                            context_size: Union[int, Tuple[int, int], List[int]] = 2,
                            by_attr: Optional[str] = None, match_type: str = 'exact', ignore_case: bool = False,
                            glob_method: str = 'match', inverse: bool = False, inplace: bool = True) \
        -> Optional[Corpus]:
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

    logger.debug('creating tokens filter mask by applying KWIC')

    by_attr = by_attr or 'token'

    try:
        matchdata = _match_against(docs, by_attr, default=docs.custom_token_attrs_defaults.get(by_attr, None))
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    matches = _build_kwic_parallel(_paralleltask(docs, matchdata), search_tokens=search_tokens,
                                   context_size=context_size, by_attr=by_attr,
                                   match_type=match_type, ignore_case=ignore_case,
                                   glob_method=glob_method, inverse=inverse, only_token_masks=True)
    return filter_tokens_by_mask(docs, matches, inplace=inplace)


#%% Corpus functions that modify corpus data: other


@corpus_func_inplace_opt
def corpus_ngramify(docs: Corpus, /, n: int, join_str: str = ' ', inplace: bool = True) -> Optional[Corpus]:
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


def corpus_sample(docs: Corpus, /, n: int, inplace: bool = True) -> Optional[Corpus]:
    """
    Generate a sample of `n` documents of corpus `docs`. Sampling occurs without replacement, hence `n` must be smaller
    or equal ``len(docs)``.

    :param docs: a Corpus object
    :param n: sample size; must be in range ``[1, len(docs)]``
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    n_docs = len(docs)
    if n_docs == 0:
        raise ValueError('cannot sample from empty corpus')

    if not 1 <= n <= n_docs:
        raise ValueError(f'`n` must be between 1 and {n_docs}')

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'sampling {n} documents out of {len(docs)}')
    sampled_doc_lbls = random.sample(docs.keys(), n)
    return filter_documents_by_label(docs, sampled_doc_lbls, inplace=inplace)


def corpus_split_by_paragraph(docs: Corpus, /, paragraph_linebreaks: int = 2, new_doc_label_fmt: str = '{doc}-{num}',
                              force_unix_linebreaks: bool = True, inplace: bool = True) -> Optional[Corpus]:
    """
    Split documents in corpus by paragraphs and set the resulting documents as new corpus. Paragraph are divided by
    a number `paragraph_linebreaks` of line breaks (``'\n'``).

    .. seealso:: See :func:`~tmtoolkit.corpus.corpus_split_by_token` which allows to split documents by any token.

    :param docs: a Corpus object
    :param paragraph_linebreaks: number of subsequent line breaks to start a new paragraph
    :param new_doc_label_fmt: document label format string with placeholders "doc" and "num" (split number)
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if not isinstance(paragraph_linebreaks, int) or paragraph_linebreaks < 1:
        raise ValueError('`paragraph_linebreaks` must be an integer greater than or equal to one')

    return corpus_split_by_token(docs, split='\n' * paragraph_linebreaks, new_doc_label_fmt=new_doc_label_fmt,
                                 force_unix_linebreaks=force_unix_linebreaks, inplace=inplace)


def corpus_split_by_token(docs: Corpus, /, split: str, new_doc_label_fmt: str = '{doc}-{num}',
                          force_unix_linebreaks: bool = True, inplace: bool = True) -> Optional[Corpus]:
    """
    Split documents in corpus by token `split` and set the resulting documents as new corpus.

    .. seealso:: See :func:`~tmtoolkit.corpus.corpus_split_by_paragraph` for a shortcut for splitting by paragraph,
                 which is a common use case.

    :param docs: a Corpus object
    :param split: string used for splitting documents
    :param new_doc_label_fmt: document label format string with placeholders "doc" and "num" (split number)
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if logger.isEnabledFor(logging.INFO):
        n_docs_before = len(docs)
    else:
        n_docs_before = None

    logger.debug('splitting documents')
    new_docs = {}
    remove_docs = []
    for lbl, d in docs.items():
        tok = d['token']
        if force_unix_linebreaks:
            tok = list(map(linebreaks_win2unix, tok))
        tok = np.array(tok)

        # find indices that split the document
        split_indices = np.flatnonzero(tok == split)

        if len(split_indices) > 0:  # there are split tokens in this document so it can be split
            # split the token matrix' rows
            split_mat = np.vsplit(d.tokenmat, split_indices+1)   # shift indices one to the right to include split token
            # split the custom token attributes arrays
            split_custom_attrs = {k: np.split(v) for k, v in d.custom_token_attrs.items()}

            # iterate through splits
            for i, mat in enumerate(split_mat):
                # generate new document using the split data
                new_lbl = new_doc_label_fmt.format(doc=lbl, num=i+1)
                if new_lbl in new_docs:
                    raise ValueError(f'generated document label "{new_lbl}" is not unique')

                new_d = Document(docs.bimaps, new_lbl, has_sents=d.has_sents,
                                 tokenmat=mat, tokenmat_attrs=d.tokenmat_attrs,
                                 custom_token_attrs={k: v[i] for k, v in split_custom_attrs.items()},
                                 doc_attrs=d.doc_attrs)
                new_docs[new_lbl] = new_d

            # record old document label for later removal
            remove_docs.append(lbl)

    if inplace:  # remove old documents in-place
        logger.debug('removing old documents')
        for lbl in remove_docs:
            del docs[lbl]
    else:  # make a copy without the old documents
        logger.debug('copying corpus without old documents')
        docs = Corpus._deserialize(docs._serialize(deepcopy_attrs=True, store_nlp_instance_pointer=True,
                                                   documents=set(doc_labels(docs)) - set(remove_docs)))

    # add split documents
    logger.debug('adding split documents')
    docs.update(new_docs)

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'corpus had {n_docs_before} documents before splitting, now has {len(docs)} documents')

    if not inplace:
        return docs


def corpus_join_documents(docs: Corpus, /, join: Dict[str, Union[str, List[str]]], glue: str = '\n\n',
                          sort_document_labels: bool = True,
                          match_type: str = 'exact', ignore_case: bool = False,
                          glob_method: str = 'match', doc_opts: Dict[str, Any] = None,
                          inplace: bool = True) -> Optional[Corpus]:
    """
    Join documents using the document labels or patterns for document labels in `join`. For each entry in `join`, the
    document labels in `docs` are matched against a provided pattern. This may be a string or a list of strings either
    for exact matching (default) or pattern matching (controlled via `match_type`). If no match is found for an entry
    in `join`, no joint document is generated.

    .. code-block::
        # example: generate joint document named "joined-tweets-foo" with all documents whose labels
        # start with "tweets-foo"
        corpus_join_documents(corp, {'joined-tweets-foo': 'tweets-foo*'}, match_type='glob')

        # alternatively specify a list of documents to match, this time using exact matching
        corpus_join_documents(corp, {'joined-tweets-foo': ['tweets-foo-1', 'tweets-foo-2', 'tweets-foo-3']})

    :param docs: a Corpus object
    :param join: dictionary that maps a name for the newly joint document to a string pattern or a list of string
                 patterns of documents to be joint
    :param glue: string used for concatenating the documents
    :param sort_document_labels: if True, sort the matched document labels before joining the documents
    :param match_type: the type of matching that is performed: ``'exact'`` does exact string matching (optionally
                       ignoring character case if ``ignore_case=True`` is set); ``'regex'`` treats ``search_tokens``
                       as regular expressions to match the tokens against; ``'glob'`` uses "glob patterns" like
                       ``"politic*"`` which matches for example "politic", "politics" or ""politician" (see
                       `globre package <https://pypi.org/project/globre/>`_)
    :param ignore_case: ignore character case (applies to all three match types)
    :param glob_method: if `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`)
    :param doc_opts: keyword arguments passed to :class:`~tmtoolkit.corpus.Document` constructor when creating a
                     joint document
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either None (if `inplace` is True) or a modified copy of the original `docs` object
    """
    if logger.isEnabledFor(logging.INFO):
        n_docs_before = len(docs)
    else:
        n_docs_before = None

    logger.debug('joining documents')

    # generate temporary "glue" document
    if glue:
        glue_tokenmat = docs._init_document(docs.nlp(glue), label='glue_doc').tokenmat
    else:
        glue_tokenmat = None

    old_docs = set()
    new_docs = {}
    doc_lbls = np.array(docs.doc_labels)

    for new_lbl, join_pattern in join.items():
        # match document label pattern
        matches = token_match_multi_pattern(join_pattern, doc_lbls,
                                            match_type=match_type, ignore_case=ignore_case, glob_method=glob_method)
        matched_lbls = doc_lbls[matches]
        if sort_document_labels:
            matched_lbls = sorted(matched_lbls)

        tokenmats = []
        token_attrs = None
        has_sents = None
        custom_token_attrs = None
        # iterate through matched documents for this pattern and build data for new joint document
        for i, lbl in enumerate(matched_lbls):
            d = docs[lbl]

            if has_sents is None:
                has_sents = d.has_sents
            elif has_sents != d.has_sents:
                raise ValueError(f'all documents for joining must have sentences recognition '
                                 f'{"enabled" if has_sents else "disabled"} when joining')

            if token_attrs is None:
                token_attrs = d.token_attrs.copy()
            elif token_attrs != d.token_attrs:
                raise ValueError('all documents must have the same token attributes when joining')

            if custom_token_attrs is None:
                custom_token_attrs = d.custom_token_attrs.copy()
            else:
                for k, v in d.custom_token_attrs.items():
                    custom_token_attrs[k] = np.concatenate((custom_token_attrs[k], v))

            if glue_tokenmat is None or i >= len(matched_lbls) - 1:
                tokenmats.append(d.tokenmat)
            else:
                tokenmats.extend((d.tokenmat, glue_tokenmat))   # additional "glue" string between joint documents

        if tokenmats:
            # concatenate the token matrices of all documents
            new_tokenmat = np.concatenate(tokenmats)
            # generate new document
            new_d = Document(docs.bimaps, new_lbl, has_sents=has_sents,
                             tokenmat=new_tokenmat,
                             tokenmat_attrs=token_attrs[:new_tokenmat.shape[1]],
                             custom_token_attrs=custom_token_attrs, **(doc_opts or {}))
            # add it to the dictionary
            new_docs[new_lbl] = new_d

        # add the matched document labels so we can later remove these documents
        old_docs.update(matched_lbls)

    if inplace:  # remove matched documents in-place
        logger.debug('removing matched documents')
        for lbl in old_docs:
            del docs[lbl]
    else:  # make a copy without the matched documents
        logger.debug('copying corpus without matched documents')
        docs = Corpus._deserialize(docs._serialize(deepcopy_attrs=True, store_nlp_instance_pointer=True,
                                                   documents=set(doc_labels(docs)) - old_docs))

    # add joint documents
    logger.debug('adding joint documents')
    docs.update(new_docs)

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'corpus had {n_docs_before} documents before joining, now has {len(docs)} documents')

    if not inplace:
        return docs




#%% other functions

def builtin_corpora_info(with_paths: bool = False) -> Union[List[str], Dict[str, str]]:
    """
    Return list/dict of available built-in corpora.

    :param with_paths: if True, return dict mapping corpus label to absolute path to dataset, else return only
                       a list of corpus labels
    :return: dict or list, depending on `with_paths`
    """

    corpora = {}

    for fpath in glob(os.path.join(DATAPATH, '**/*.zip')):
        pathcomp = path_split(fpath)
        basename, _ = os.path.splitext(pathcomp[-1])

        corpora[pathcomp[-2] + '-' + basename] = os.path.abspath(fpath)

    if with_paths:
        return corpora
    else:
        return sorted(corpora.keys())


#%% helper functions


@parallelexec(collect_fn=merge_sets)
def _filter_documents(chunk, search_tokens, match_type, ignore_case, glob_method, inverse_matches, inverse_result,
                      matches_threshold, return_num_matches):
    matches = _token_pattern_matches(chunk, search_tokens, match_type=match_type,
                                     ignore_case=ignore_case, glob_method=glob_method)
    if return_num_matches:
        docs_matches = {}
    else:
        docs_matches = set()

    for lbl, m in matches.items():
        if inverse_matches:
            m = ~m

        n = np.sum(m)
        thresh_met = n >= matches_threshold
        if inverse_result:
            thresh_met = not thresh_met
        if not thresh_met:
            if return_num_matches:
                docs_matches[lbl] = n
            else:
                docs_matches.add(lbl)

    return docs_matches


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
    logger.debug('finalizing KWIC results')

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'KWIC found {sum(len(wins) for wins in kwic_results.values())} contexts in '
                    f'{len(kwic_results)} documents')

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


def _create_embed_tokens_for_collocations(docs: Corpus, embed_tokens_min_docfreq, embed_tokens_set, tokens_as_hashes):
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
        token_df = doc_frequencies(docs, tokens_as_hashes=tokens_as_hashes, proportions=df_prop)
        embed_tokens = {t for t, df in token_df.items() if df >= embed_tokens_min_docfreq}
        if embed_tokens_set:  # additionally use fixed set of tokens
            embed_tokens.update(embed_tokens_set)
        return embed_tokens
    else:
        # solely use fixed set of tokens
        if tokens_as_hashes and embed_tokens_set:
            return {hash_string(t) for t in embed_tokens_set}
        else:
            return embed_tokens_set


def _apply_collocations(tokenmat: np.ndarray,
                        new_tok: Sequence[StrOrInt],
                        mask: np.ndarray,
                        hash2token: Optional[bidict],
                        tokens_as_hashes: bool,
                        glue: Optional[str],
                        return_joint_tokens: bool):
    """
    Helper function to apply collocations from `joint_colloc` to documents in `docs`. `joint_colloc` maps document label
    to a tuple containing new (joint) tokens and a mask as provided by :func:`~tmtookit.tokenseq.token_join_subsequent`
    with parameter ``return_mask=True``. The tokens can be given as strings or as hashes (integers).
    """
    if return_joint_tokens:
        joint_tokens = set()

    hash2token_updates = {}

    if new_tok:
        tok_hashes = tokenmat[:, 1]

        # get new tokens as strings
        if tokens_as_hashes:
            # the tokens in the collocations are hashes:
            # 1. get the token type strings for each collocation token hash `t` from the bimap
            # 2. join the token type strings with `glue`
            new_tok_strs = [glue.join(hash2token[h] for h in colloc) for colloc in new_tok]
        else:
            # the tokens in the collocations are strings: use them as-is
            new_tok_strs = new_tok

        if return_joint_tokens:
            joint_tokens.update(new_tok_strs)

        # add the strings as new token types to the bimap and save the hashes to the array
        new_tok_hashes = list(map(hash_string, new_tok_strs))
        hash2token_updates.update(zip(new_tok_hashes, new_tok_strs))
        tok_hashes[mask > 1] = np.array(new_tok_hashes, dtype='uint64')   # replace with hashes of new tokens
        tokenmat = np.delete(tokenmat, np.flatnonzero(mask == 0), axis=0)

    if return_joint_tokens:
        return tokenmat, hash2token_updates, joint_tokens
    else:
        return tokenmat, hash2token_updates


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


def _match_against(docs: Union[Corpus, Dict[str, Document]], by_attr: str = 'token',
                   select: Optional[Union[str, Collection[str]]] = None, **kwargs) \
        -> Dict[str, Any]:
    """Return the list of values to match against in filtering functions."""
    return {lbl: document_token_attr(d, attr=by_attr, **kwargs) for lbl, d in docs.items()
            if select is None or lbl in select}


def _check_filter_args(**kwargs):
    """Helper function to check common filtering arguments match_type and glob_method."""
    if 'match_type' in kwargs and kwargs['match_type'] not in {'exact', 'regex', 'glob'}:
        raise ValueError("`match_type` must be one of `'exact', 'regex', 'glob'`")

    if 'glob_method' in kwargs and kwargs['glob_method'] not in {'search', 'match'}:
        raise ValueError("`glob_method` must be one of `'search', 'match'`")


def _token_pattern_matches(tokens: Dict[str, List[Any]], search_tokens: Any,
                           match_type: str = 'exact', ignore_case=False, glob_method: str = 'match'):
    """
    Helper function to apply `token_match` with multiple patterns in `search_tokens` to `docs`.
    The matching results for each pattern in `search_tokens` are combined via logical OR.
    Returns a dict mapping keys in `tokens` to boolean arrays that signal the pattern matches for each token in each
    document.
    """

    # search tokens may be of any type (e.g. bool when matching against token attributes)
    if not isinstance(search_tokens, (list, tuple, set)):
        search_tokens = [search_tokens]
    elif isinstance(search_tokens, (list, tuple, set)) and not search_tokens:
        raise ValueError('`search_tokens` must not be empty')

    matches = [np.repeat(False, repeats=len(dtok)) for dtok in tokens.values()]

    for dtok, dmatches in zip(tokens.values(), matches):
        for pat in search_tokens:
            dmatches |= token_match(pat, dtok, match_type=match_type, ignore_case=ignore_case, glob_method=glob_method)

    return dict(zip(tokens.keys(), matches))


def _load_text_from_files(files: Collection[str],
                          filelabels: Optional[Dict[str, str]] = None,
                          existing_docs: Optional[Collection[str]] = None,
                          encoding: str = 'utf8',
                          doc_label_fmt: str = '{path}-{basename}',
                          doc_label_path_join: str = '_',
                          read_size: int = -1,
                          force_unix_linebreaks: bool = True) \
        -> Dict[str, str]:
    """
    Helper function to load text data from text files.

    :param files: collection of files to be loaded (as full file paths)
    :param filelabels: dict mapping file paths to document labels
    :param existing_docs: collection of already existing document labels to check against duplicates
    :param encoding: character encoding
    :param doc_label_fmt: document label format for non-tabular files; string with placeholders ``"path"``,
                          ``"basename"``, ``"ext"``
    :param doc_label_path_join: string with which to join the components of the file paths
    :param read_size: max. number of characters to read. -1 means read full file.
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
    :return: dict mapping document label to document text
    """
    existing_docs = existing_docs or set()
    new_docs = {}
    for fpath in files:
        text = read_text_file(fpath, encoding=encoding, read_size=read_size,
                              force_unix_linebreaks=force_unix_linebreaks)

        path_parts = path_split(os.path.normpath(fpath))
        if not path_parts:
            continue

        dirs, fname = path_parts[:-1], path_parts[-1]
        basename, ext = os.path.splitext(fname)
        basename = basename.strip()
        if ext:
            ext = ext[1:]

        if filelabels is None:   # generate a label
            lbl = doc_label_fmt.format(path=doc_label_path_join.join(dirs), basename=basename, ext=ext)

            if lbl.startswith('-'):
                lbl = lbl[1:]
        else:                   # use from dict keys
            lbl = filelabels[fpath]

        if lbl in existing_docs or lbl in new_docs:
            raise ValueError(f'duplicate document label "{lbl}" not allowed')

        new_docs[lbl] = text

    return new_docs


def _load_text_from_tabular_files(files: Union[str, Collection[str]],
                                  id_column: StrOrInt, text_column: StrOrInt,
                                  existing_docs: Optional[Collection[str]] = None,
                                  prepend_columns: Optional[Sequence[str]] = None, encoding: str = 'utf8',
                                  doc_label_fmt: str = '{basename}-{id}',
                                  force_unix_linebreaks: bool = True,
                                  pandas_read_opts: Optional[Dict[str, Any]] = None) \
        -> Dict[str, str]:
    """
    Helper function to load text data from tabular files.

    :param files: collection of files to be loaded (as full file paths)
    :param id_column: column name or column index of document identifiers
    :param text_column: column name or column index of document texts
    :param existing_docs: collection of already existing document labels to check against duplicates
    :param prepend_columns: if not None, pass a list of columns whose contents should be added before the document
                            text, e.g. ``['title', 'subtitle']``
    :param encoding: character encoding of the files
    :param doc_label_fmt: document label format string with placeholders ``"basename"``, ``"id"`` (document ID), and
                          ``"row_index"`` (dataset row index)
    :param sample: if given, draw random sample of size `sample` from all text data
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks in texts
    :param pandas_read_opts: additional arguments passed to :func:`pandas.read_csv` or :func:`pandas.read_excel`
    :return: dict mapping document label to document text
    """
    existing_docs = existing_docs or set()

    if isinstance(files, str):
        files = [files]

    read_opts = {
        'encoding': encoding,
        'usecols': [id_column, text_column]
    }

    if prepend_columns:
        read_opts['usecols'] += prepend_columns

    if all(isinstance(x, int) for x in read_opts['usecols']):
        id_column, text_column = 0, 1
        if prepend_columns:
            prepend_columns = list(range(2, len(prepend_columns) + 2))

    if pandas_read_opts:
        read_opts.update(pandas_read_opts)

    read_opts_excel = read_opts.copy()
    del read_opts_excel['encoding']

    new_docs = {}
    for fpath in files:
        if fpath.endswith('.csv'):
            data = pd.read_csv(fpath, **read_opts)
        elif fpath.endswith('.xls') or fpath.endswith('.xlsx'):
            if fpath.endswith('.xlsx') and 'engine' not in read_opts_excel:
                read_opts_excel['engine'] = 'openpyxl'
            data = pd.read_excel(fpath, **read_opts_excel)
        else:
            raise ValueError('only file extensions ".csv", ".xls" and ".xlsx" are supported')

        basename, _ = os.path.splitext(fpath)
        basename = os.path.basename(basename).strip()

        for idx, row in data.iterrows():
            lbl = doc_label_fmt.format(basename=basename, id=row[id_column], row_index=idx)

            if lbl in existing_docs or lbl in new_docs:
                raise ValueError(f'duplicate document label "{lbl}" not allowed')

            if prepend_columns:
                text = '\n\n'.join([row[col] for col in (prepend_columns + [text_column]) if pd.notna(row[col])])
            else:
                text = row[text_column] if pd.notna(row[text_column]) else ''

            if force_unix_linebreaks:
                text = linebreaks_win2unix(text)

            new_docs[lbl] = text

    return new_docs


def _spacydocs_for_vectors(docs, select, collapse):
    select = _single_str_to_set(select, check_docs=docs)

    if isinstance(docs, Corpus):
        if docs.nlp.meta.get('vectors', {}).get('width', 0) == 0:
            raise RuntimeError("Corpus object `docs` doesn't use a SpaCy language model with word vectors; you should "
                               "enable the 'vectors' feature via `load_features` parameter or specify a different language "
                               "model (i.e. an ..._md or ..._lg model) via `language_model` parameter when initializing "
                               "the Corpus object")
        return spacydocs(docs, select=select, collapse=collapse)
    elif isinstance(docs, dict):
        if select:
            return {lbl: d for lbl, d in docs.items() if lbl in select}
        else:
            return docs
    else:
        raise ValueError('`docs` must be Corpus object or dict of SpaCy Doc objects')


def _single_str_to_set(select: Optional[Union[str, Collection[str]]], check_docs: Optional[Corpus] = None) \
        -> Optional[Collection[str]]:
    if isinstance(select, str):
        select = {select}

    if select is not None and check_docs is not None and not set(select) <= set(check_docs.keys()):
        raise KeyError(f'one or more documents not found in corpus: {set(select) - set(check_docs.keys())}')

    return select
