"""
Internal module that implements functions that operate on :class:`~tmtoolkit.corpus.Corpus` objects.

The source is separated into sections using a ``#%% ...`` marker.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import operator
import os
import unicodedata
from copy import copy
from functools import partial, wraps
from inspect import signature
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Union, List, Callable, Iterable, Optional, Any, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from spacy.tokens import Doc

from ..bow.dtm import create_sparse_dtm, dtm_to_dataframe, dtm_to_datatable
from ..utils import merge_dicts, merge_counters, empty_chararray, as_chararray, \
    flatten_list, combine_sparse_matrices_columnwise, arr_replace, pickle_data, unpickle_file, merge_sets
from .._pd_dt_compat import USE_DT, FRAME_TYPE, pd_dt_frame, pd_dt_concat, pd_dt_sort, pd_dt_colnames
from ..tokenseq import token_lengths, token_ngrams, token_match_multi_pattern, index_windows_around_matches, \
    token_match_subsequent, token_join_subsequent, npmi, token_collocations

from ._common import LANGUAGE_LABELS, simplified_pos
from ._corpus import Corpus
from ._helpers import _filtered_doc_token_attr, _filtered_doc_tokens, _corpus_from_tokens, \
    _ensure_writable_array, _check_filter_args, _token_pattern_matches, _match_against, _apply_matches_array


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


def parallelexec(collect_fn: Callable):
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
                workers_data = [{lbl: docs_or_task.data[lbl] for lbl in itemlabels
                                 if lbl in docs_or_task.data.keys()}
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
        ret = fn(corp, *args[1:], **kwargs)
        if ret is None:
            return corp
        else:
            if inplace:
                return ret
            else:
                return corp, ret


    return inner_fn


def corpus_func_filters_tokens(fn):
    @wraps(fn)
    def inner_fn(*args, **kwargs):
        assert isinstance(args[0], Corpus), 'first argument must be a Corpus object'

        corp = args[0]

        # apply fn to `corp`, passing all other arguments
        fn(corp, *args[1:], **kwargs)

        corp._tokens_masked = True

        return corp

    return inner_fn


def corpus_func_processes_tokens(fn):
    @wraps(fn)
    def inner_fn(*args, **kwargs):
        assert isinstance(args[0], Corpus), 'first argument must be a Corpus object'

        corp = args[0]

        # apply fn to `corp`, passing all other arguments
        ret = fn(corp, *args[1:], **kwargs)

        corp._tokens_processed = True

        if ret is None:
            return corp
        else:
            return ret

    return inner_fn

#%% Corpus functions with readonly access to Corpus data


def doc_tokens(docs: Union[Corpus, Dict[str, Doc]],
               only_non_empty=False,
               tokens_as_hashes=False,
               with_attr: Union[bool, list, tuple] = False,
               with_mask=False,
               as_datatables=False, as_arrays=False,
               apply_document_filter=True,
               apply_token_filter=True,
               force_unigrams=False) \
        -> Dict[str, Union[List[Union[str, int]],
                           np.ndarray,
                           Dict[str, Union[list, np.ndarray]],
                           FRAME_TYPE]]:
    """
    Retrieve document tokens from a Corpus or dict of SpaCy documents. Optionally also retrieve document and token
    attributes.

    :param docs: a Corpus object or a dict mapping document labels to SpaCy `Doc` objects
    :param only_non_empty: if True, only return non-empty result documents
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
                             as from `SpaCy StringStore <https://spacy.io/api/stringstore/>`_
    :param with_attr: also return document and token attributes along with each token; if True, returns all default
                      attributes and custom defined attributes; if list or tuple, returns attributes specified in this
                      sequence
    :param with_mask: if True, also return the document and token mask attributes; this disables the document or token
                      filtering (i.e. `apply_token_filter` and `apply_document_filter` are set to False)
    :param as_datatables: return result as datatable/dataframe with tokens and document and token attributes in columns
    :param as_arrays: return result as NumPy arrays instead of lists
    :param apply_document_filter: if False, ignore document filter mask
    :param apply_token_filter: if False, ignore token filter mask
    :param force_unigrams: ignore n-grams setting if `docs` is a Corpus with ngrams and always return unigrams
    :return: dict mapping document labels to document tokens data, which can be of different form, depending on the
             arguments passed to this function: (1) list of token strings or hash integers; (2)  NumPy array of token
             strings or hash integers; (3) dict containing ``"token"`` key with values from (1) or (2) and document
             and token attributes with their values as list or NumPy array; (4) datatable/dataframe with tokens and
             document and token attributes in columns
    """
    if with_mask and not with_attr:
        with_attr = ['mask']
    mask_in_attr = isinstance(with_attr, (list, tuple)) and 'mask' in with_attr
    with_mask = with_mask or mask_in_attr

    if with_mask:  # requesting the document and token mask disables the token filtering
        apply_token_filter = False
        apply_document_filter = False

    ng = 1
    ng_join_str = None
    doc_attrs = {}
    custom_token_attrs_defaults = None   # set to None if docs is not a Corpus but a dict of SpaCy Docs

    if isinstance(docs, Corpus):
        if not force_unigrams:
            ng = docs.ngrams
            ng_join_str = docs.ngrams_join_str

        doc_attrs = docs.doc_attrs_defaults.copy()
        # rely on custom token attrib. w/ defaults as reported from Corpus
        custom_token_attrs_defaults = docs.custom_token_attrs_defaults
        docs = docs.spacydocs_ignore_filter if with_mask else docs.spacydocs

    if with_mask and 'mask' not in doc_attrs.keys():
        doc_attrs['mask'] = True

    if isinstance(with_attr, (list, tuple)):
        doc_attrs = {k: doc_attrs[k] for k in with_attr if k in doc_attrs.keys()}

    res = {}
    for lbl, d in docs.items():
        # skip this document if it is empty and `only_non_empty` is True
        # or if the document is masked and `apply_document_filter` is True
        if (only_non_empty and len(d) == 0) or (apply_document_filter and not d._.mask):
            continue

        # get the tokens of the document
        tok = _filtered_doc_tokens(d, tokens_as_hashes=tokens_as_hashes, apply_filter=apply_token_filter)

        if ng > 1:
            tok = token_ngrams(tok, n=ng, join=True, join_str=ng_join_str)

        if with_attr is not False:   # extract document and token attributes
            resdoc = {}

            # document attributes
            for k, default in doc_attrs.items():
                a = 'doc_mask' if k == 'mask' else k
                v = getattr(d._, k)
                if v is None:     # can't use default arg (third arg) in `getattr` b/c Doc extension *always* returns
                                  # a value; it will be None by Doc extension default
                    v = default
                resdoc[a] = [v] * len(tok) if as_datatables else v

            # always set tokens
            resdoc['token'] = tok

            # standard (SpaCy) token attributes
            if isinstance(with_attr, (list, tuple)):
                spacy_attrs = [k for k in with_attr if k != 'mask']
            else:
                spacy_attrs = Corpus.STD_TOKEN_ATTRS

            for k in spacy_attrs:
                v = _filtered_doc_token_attr(d, k, apply_filter=apply_token_filter)
                if k == 'whitespace':
                    v = list(map(lambda ws: ws != '', v))
                if ng > 1:
                    v = token_ngrams(list(map(str, v)), n=ng, join=True, join_str=ng_join_str)
                resdoc[k] = v

            # custom token attributes
            # if docs is not a Corpus but a dict of SpaCy Docs, use the keys in `user_data` as custom token attributes
            # -> risky since these `user_data` dict keys may differ between documents
            user_attrs = list(d.user_data.keys() if custom_token_attrs_defaults is None
                              else custom_token_attrs_defaults.keys())
            if isinstance(with_attr, (list, tuple)):
                user_attrs = [k for k in user_attrs if k in with_attr]

            if with_mask and 'mask' not in user_attrs:
                user_attrs.append('mask')

            for k in user_attrs:
                if isinstance(k, str):
                    default = None if custom_token_attrs_defaults is None else custom_token_attrs_defaults.get(k, None)
                    v = _filtered_doc_token_attr(d, k, default=default, custom=True, apply_filter=apply_token_filter)
                    if not as_datatables and not as_arrays:
                        v = list(v)
                    if ng > 1:
                        v = token_ngrams(list(map(str, v)), n=ng, join=True, join_str=ng_join_str)
                    resdoc[k] = v
            res[lbl] = resdoc
        else:
            res[lbl] = tok

    if as_datatables:
        res = dict(zip(res.keys(), map(pd_dt_frame, res.values())))
    elif as_arrays:
        if with_attr:
            res = dict(zip(res.keys(),
                           [dict(zip(d.keys(), map(np.array, d.values()))) for d in res.values()]))
        else:
            res = dict(zip(res.keys(), map(np.array, res.values())))

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

    if docs.is_processed or docs.tokens_filtered:
        res = {}
        vocab = docs.nlp.vocab
        for lbl, tok_hashes in doc_tokens(docs, tokens_as_hashes=True, force_unigrams=True).items():
            tok_vecs = [vocab.get_vector(h) for h in tok_hashes if not omit_oov or vocab.has_vector(h)]
            res[lbl] = np.vstack(tok_vecs) if tok_vecs else np.array([], dtype='float32')
        return res
    else:   # fast track
        return {dl: np.vstack([t.vector for t in d]) if len(d) > 0 else np.array([], dtype='float32')
                for dl, d in docs.spacydocs.items()}


def vocabulary(docs: Union[Corpus, Dict[str, List[str]]], tokens_as_hashes=False, force_unigrams=False, sort=False)\
        -> Union[set, list]:
    if isinstance(docs, Corpus):
        tok = doc_tokens(docs, tokens_as_hashes=tokens_as_hashes, force_unigrams=force_unigrams).values()
    else:
        tok = docs.values()

    v = set(flatten_list(tok))

    if sort:
        return sorted(v)
    else:
        return v


def vocabulary_counts(docs: Corpus, tokens_as_hashes=False) -> Counter:
    """
    Return :class:`collections.Counter` instance of vocabulary containing counts of occurrences of tokens across
    all documents.

    :param docs: list of string tokens or spaCy documents
    :param tokens_as_hashes: if True, return token type hashes (integers) instead of textual representations (strings)
                             as from `SpaCy StringStore <https://spacy.io/api/stringstore/>`_
    :return: :class:`collections.Counter` instance of vocabulary containing counts of occurrences of tokens across
             all documents
    """
    @parallelexec(collect_fn=merge_counters)
    def _vocabulary_counts(tokens):
        return Counter(flatten_list(tokens.values()))
    tok = doc_tokens(docs, tokens_as_hashes=tokens_as_hashes)
    return _vocabulary_counts(_paralleltask(docs, tok))


def vocabulary_size(docs: Corpus, force_unigrams=False) -> int:
    return len(vocabulary(docs, force_unigrams=force_unigrams))


def tokens_with_attr(docs: Corpus) -> Dict[str, FRAME_TYPE]:
    return doc_tokens(docs, with_attr=True, as_datatables=True)


def tokens_datatable(docs: Corpus, with_attr: Union[bool, list, tuple, set] = True, with_mask=False)\
        -> FRAME_TYPE:
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

    tokens = doc_tokens(docs, only_non_empty=False, with_attr=with_attr or [],
                        with_mask=with_mask, as_datatables=True)
    dfs = _tokens_datatable(_paralleltask(docs, tokens))
    res = None

    if dfs:
        res = pd_dt_concat(dfs)

    if res is None or len(res) == 0:
        res = pd_dt_frame({'doc': [], 'position': [], 'token': []})

    return pd_dt_sort(res, ['doc', 'position'])


def tokens_dataframe(docs: Corpus, with_attr: Union[bool, list, tuple, set] = True, with_mask=False)\
        -> pd.DataFrame:
    # note that generating a datatable first and converting it to pandas is faster than generating a pandas data
    # frame right away

    df = tokens_datatable(docs, with_attr=with_attr, with_mask=with_mask)

    if USE_DT:
        df = df.to_pandas()

    return df.set_index(['doc', 'position'])


def tokens_with_pos_tags(docs: Corpus) -> Dict[str, FRAME_TYPE]:
    """
    Document tokens with POS tag as dict with mapping document label to datatable. The datatables have two
    columns, ``token`` and ``pos``.
    """
    return {dl: df[:, ['token', 'pos']] if USE_DT else df.loc[:, ['token', 'pos']]
            for dl, df in doc_tokens(docs, with_attr=True, as_datatables=True).items()}


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
        return np.concatenate(list(tok.values()))
    else:
        return flatten_list(tok.values())


def corpus_num_tokens(docs: Corpus) -> int:
    return sum(doc_lengths(docs).values())


def corpus_num_chars(docs: Corpus) -> int:
    return sum(sum(n) for n in doc_token_lengths(docs).values())


def corpus_collocations(docs: Corpus, threshold: Optional[float] = None,
                        min_count: int = 1, embed_tokens_min_docfreq: Optional[Union[int, float]] = None,
                        embed_tokens_set: Optional[Union[set, tuple, list]] = None,
                        statistic: Callable = npmi, return_statistic=True, rank: Optional[str] = 'desc',
                        as_datatable=True, glue: str = ' ', **statistic_kwargs):
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
    :param as_datatable: return result as datatable / dataframe with columns "collocation" and optionally "statistic"
    :param glue: if not None, provide a string that is used to join the collocation tokens; must be set if
                 `as_datatable` is True
    :param statistic_kwargs: additional arguments passed to `statistic` function
    :return:
    """
    if as_datatable and glue is None:
        raise ValueError('`glue` cannot be None if `as_datatable` is True')

    tok = [corpus_tokens_flattened(docs)]    # TODO: use sentences
    vocab_counts = vocabulary_counts(docs)

    embed_tokens = _create_embed_tokens_for_collocations(docs, embed_tokens_min_docfreq, embed_tokens_set)

    colloc = token_collocations(tok, threshold=threshold, min_count=min_count, embed_tokens=embed_tokens,
                                vocab_counts=vocab_counts, statistic=statistic, return_statistic=return_statistic,
                                rank=rank, glue=glue, **statistic_kwargs)

    if as_datatable:
        if return_statistic:
            bg, stat = zip(*colloc)
            cols = {'collocation': bg, 'statistic': stat}
        else:
            cols = {'collocation': colloc}
        return pd_dt_frame(cols)
    else:
        return colloc


def corpus_summary(docs: Corpus, max_documents=None, max_tokens_string_length=None) -> str:
    """
    Print a summary of this object, i.e. the first tokens of each document and some summary statistics.

    :param max_documents: maximum number of documents to print; ``None`` uses default value 10; set to -1 to
                          print *all* documents
    :param max_tokens_string_length: maximum string length of concatenated tokens for each document; ``None`` uses
                                     default value 50; set to -1 to print complete documents
    """

    if max_tokens_string_length is None:
        max_tokens_string_length = docs.print_summary_default_max_tokens_string_length
    if max_documents is None:
        max_documents = docs.print_summary_default_max_documents

    summary = f'Corpus with {docs.n_docs} document' \
              f'{"s" if docs.n_docs > 1 else ""} ({docs.n_docs_masked} masked) in ' \
              f'{LANGUAGE_LABELS[docs.language].capitalize()}'

    texts = doc_texts(docs)
    dlengths = doc_lengths(docs)

    for dl, tokstr in texts.items():
        if max_tokens_string_length >= 0 and len(tokstr) > max_tokens_string_length:
            tokstr = tokstr[:max_tokens_string_length] + '...'

        summary += f'\n> {dl} ({dlengths[dl]} tokens): {tokstr}'

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
    def _ngrams(tokens):
        return {dl: token_ngrams(dt, n, join, join_str) for dl, dt in tokens.items()}

    return _ngrams(_paralleltask(docs))


def kwic(docs: Corpus, search_tokens: Union[Any, list], context_size: Union[int, tuple, list] = 2,
         by_attr: Optional[str] = None, match_type: str = 'exact', ignore_case=False, glob_method: str = 'match',
         inverse=False, with_attr: Union[bool, list, tuple] = False, as_datatables=False, only_non_empty=False,
         glue: Optional[str] = None, highlight_keyword: Optional[str] = None):
    """
    Perform *keyword-in-context (KWIC)* search for `search_tokens`. Uses similar search parameters as
    :func:`filter_tokens`. Returns results as dict with document label to KWIC results mapping. For
    tabular output, use :func:`kwic_table`.

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
    :param as_datatables: return result as datatable/dataframe with "doc" (document label) and "context" (context
                          ID per document) and optionally "position" (original token position in the document) if
                          tokens are not glued via `glue` parameter
    :param only_non_empty: if True, only return non-empty result documents
    :param glue: if not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: if not None, this must be a string which is used to indicate the start and end of the
                              matched keyword
    :return: dict with `document label -> kwic for document` mapping or a data frame, depending on `as_datatables`
    """
    if isinstance(context_size, int):
        context_size = (context_size, context_size)
    elif not isinstance(context_size, (list, tuple)):
        raise ValueError('`context_size` must be integer or list/tuple')

    if len(context_size) != 2:
        raise ValueError('`context_size` must be list/tuple of length 2')

    if highlight_keyword is not None and not isinstance(highlight_keyword, str):
        raise ValueError('if `highlight_keyword` is given, it must be of type str')

    if glue:
        if with_attr or as_datatables:
            raise ValueError('when `glue` is set to True, `with_attr` and `as_datatables` must be False')
        if not isinstance(glue, str):
            raise ValueError('if `glue` is given, it must be of type str')

    try:
        matchdata = _match_against(docs.spacydocs, by_attr, default=docs.custom_token_attrs_defaults.get(by_attr, None))
    except AttributeError:
        raise AttributeError(f'attribute name "{by_attr}" does not exist')

    if with_attr:
        docs_w_attr = doc_tokens(docs, with_attr=with_attr, as_arrays=True)
        prepared = {}
        for lbl, matchagainst in matchdata.items():
            prepared[lbl] = merge_dicts((docs_w_attr[lbl], {'_matchagainst': matchagainst}))
    else:
        prepared = {k: {'_matchagainst': v} for k, v in matchdata.items()}

    kwicres = _build_kwic_parallel(_paralleltask(docs, prepared), search_tokens=search_tokens,
                                   context_size=context_size, by_attr=by_attr,
                                   match_type=match_type, ignore_case=ignore_case,
                                   glob_method=glob_method, inverse=inverse, highlight_keyword=highlight_keyword,
                                   with_window_indices=as_datatables, only_token_masks=False)

    return _finalize_kwic_results(kwicres, only_non_empty=only_non_empty, glue=glue, as_datatables=as_datatables,
                                  matchattr=by_attr or 'token', with_attr=bool(with_attr))


def kwic_table(docs: Corpus, search_tokens: Union[Any, list], context_size: Union[int, tuple, list] = 2,
               by_attr: Optional[str] = None, match_type: str = 'exact', ignore_case=False, glob_method: str = 'match',
               inverse=False, glue: str = ' ', highlight_keyword: Optional[str] = '*'):
    """
    Perform *keyword-in-context (KWIC)* search for `search_tokens` and return result as datatable/dataframe with
    columns ``doc`` (document label), ``context`` (document-specific context number) and ``kwic`` (KWIC result).
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
    :param glue: if not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: if not None, this must be a string which is used to indicate the start and end of the
                              matched keyword
    :return: datatable/dataframe with columns ``doc`` (document label), ``context`` (document-specific context number)
             and ``kwic`` (KWIC result)
    """

    if not isinstance(glue, str):
        raise ValueError('`glue` must be of type str')

    kwicres = kwic(docs, search_tokens=search_tokens, context_size=context_size, by_attr=by_attr, match_type=match_type,
                   ignore_case=ignore_case, glob_method=glob_method, inverse=inverse, with_attr=False,
                   as_datatables=False, only_non_empty=True, glue=glue, highlight_keyword=highlight_keyword)

    return _datatable_from_kwic_results(kwicres)


#%% Corpus I/O


def save_corpus_to_picklefile(docs: Corpus, picklefile: str):
    pickle_data(serialize_corpus(docs, deepcopy_attrs=False), picklefile)


def load_corpus_from_picklefile(picklefile: str) -> Corpus:
    return deserialize_corpus(unpickle_file(picklefile))


def load_corpus_from_tokens(tokens: Dict[str, Union[list, tuple, Dict[str, List]]],
                            doc_attr_names: Optional[Sequence] = None,
                            token_attr_names: Optional[Sequence] = None,
                            **corpus_kwargs) -> Corpus:
    if 'docs' in corpus_kwargs:
        raise ValueError('`docs` parameter is obsolete when initializing a Corpus with this function')

    corp = Corpus(**corpus_kwargs)
    _corpus_from_tokens(corp, tokens, doc_attr_names=doc_attr_names, token_attr_names=token_attr_names)

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
    doc_attr_names = set()
    token_attr_names = set()
    for dl in dt.unique(tokens[:, dt.f.doc]).to_list()[0]:
        doc_df = tokens[dt.f.doc == dl, :]

        colnames = pd_dt_colnames(doc_df)
        colnames.pop(colnames.index('doc'))
        colnames.pop(colnames.index('position'))

        doc_attr_names.update(colnames[:colnames.index('token')])
        token_attr_names.update(colnames[colnames.index('token')+1:])

        tokens_dict[dl] = doc_df[:, colnames]

    return load_corpus_from_tokens(tokens_dict,
                                   doc_attr_names=list(doc_attr_names),
                                   token_attr_names=list(token_attr_names.difference(Corpus.STD_TOKEN_ATTRS)),
                                   **corpus_kwargs)


def serialize_corpus(docs: Corpus, deepcopy_attrs=True):
    return docs._create_state_object(deepcopy_attrs=deepcopy_attrs)


def deserialize_corpus(serialized_corpus_data: dict):
    return Corpus._deserialize(serialized_corpus_data)

#%% Corpus functions that modify corpus data: document / token attribute handling

@corpus_func_copiable
def set_document_attr(docs: Corpus, /, attrname: str, data: Dict[str, Any], default=None, inplace=True):
    """
    Set a document attribute named `attrname` for documents in Corpus object `docs`.

    :param docs: a Corpus object
    :param attrname: name of the document attribute
    :param data: dict that maps document labels to document attribute value
    :param default: default document attribute value
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either original Corpus object `docs` or a modified copy of it
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
def set_token_attr(docs: Corpus, /, attrname: str, data: Dict[str, Any], default=None, per_token_occurrence=True,
                   inplace=True):
    """
    Set a token attribute named `attrname` for all tokens in all documents in Corpus object `docs`.

    There are two ways of assigning token attributes which are determined by the argument `per_token_occurrence`. If
    `per_token_occurrence` is True, then `data` is a dict that maps token occurrences (or "word types") to attribute
    values, i.e. ``{'foo': True}`` will assign the attribute value ``True`` to every occurrence of the token ``"foo"``.
    If `per_token_occurrence` is False, then `data` is a dict that maps document labels to token attributes. In this
    case the token attributes must be a list, tuple or NumPy array with a length according to the number of (unmasked)
    tokens.

    :param docs: a Corpus object
    :param attrname: name of the token attribute
    :param data: depends on `per_token_occurrence` –
    :param per_token_occurrence: determines how `data` is interpreted when assigning token attributes
    :param default: default token attribute value
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either original Corpus object `docs` or a modified copy of it
    """
    if attrname in docs.STD_TOKEN_ATTRS + ['mask', 'processed']:
        raise ValueError(f'cannot set attribute with protected name "{attrname}"')

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


#%% Corpus functions that modify corpus data: token transformations


@corpus_func_copiable
@corpus_func_processes_tokens
def transform_tokens(docs: Corpus, /, func: Callable, inplace=True, **kwargs):
    vocab = vocabulary(docs, tokens_as_hashes=True, force_unigrams=True)
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
    """
    Removes punctuation characters *in* tokens, i.e. ``['a', '.', 'f;o;o']`` becomes ``['a', '', 'foo']``.

    If you want to remove punctuation tokens, use :func:`~filter_clean_tokens`

    :param docs:
    :param inplace:
    :return:
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
    :return: either original Corpus object `docs` or a modified copy of it
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
    :return: either original Corpus object `docs` or a modified copy of it
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
    for d in docs.spacydocs.values():
        d.user_data['processed'] = np.fromiter((t.lemma for t in d), dtype='uint64', count=len(d))


@corpus_func_copiable
@corpus_func_processes_tokens
def join_collocations_by_patterns(docs: Corpus, /, patterns: Union[Any, list], glue: str = '_',
                                  match_type: str = 'exact', ignore_case=False, glob_method: str = 'match',
                                  return_joint_tokens=False, inverse=False, inplace=True):
    """
    Match N *subsequent* tokens to the N patterns in `patterns` using match options like in :func:`filter_tokens`.
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
    :param inverse: inverse the match results for filtering (i.e. *remove* all tokens that match the search
                    criteria)
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either original Corpus object `docs` or a modified copy of it; if `return_joint_tokens` is True,
             return set of joint collocations instead (if `inplace` is True) or additionally in tuple
             ``(modified Corpus copy, set of joint collocations)`` (if `inplace` is False)
    """
    if not isinstance(patterns, (list, tuple)) or len(patterns) < 2:
        raise ValueError('`patterns` must be a list or tuple containing at least two elements')

    if not isinstance(glue, str):
        raise ValueError('`glue` must be a string')

    @parallelexec(merge_dicts)
    def _join_colloc(chunk: Dict[str, List[str]]):
        res = {}
        for lbl, tok in chunk.items():
            # get the subsequent matches as binary mask arrays
            matches = token_match_subsequent(patterns, tok, match_type=match_type, ignore_case=ignore_case,
                                             glob_method=glob_method)
            if inverse:
                matches = [~m for m in matches]

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
                                   embed_tokens_set: Optional[Union[set, tuple, list]] = None,
                                   statistic: Callable = npmi, return_joint_tokens=False, inverse=False, inplace=True,
                                   **statistic_kwargs):
    if not isinstance(glue, str):
        raise ValueError('`glue` must be a string')

    @parallelexec(merge_dicts)
    def _join_colloc(chunk: Dict[str, List[str]], colloc):
        res = {}
        for lbl, tok in chunk.items():
            # get the subsequent matches of the collocation token hashes as binary mask arrays
            matches = []
            for hashes in colloc:
                matches.extend(token_match_subsequent(hashes, tok, match_type='exact'))

            if inverse:
                matches = [~m for m in matches]

            # join the matched subsequent tokens; `return_mask=True` makes sure that we only get the newly
            # generated joint tokens together with an array to mask all but the first token of the subsequent tokens
            # `glue=None` makes sure that the token hashes are not joint
            res[lbl] = token_join_subsequent(tok, matches, glue=None, return_mask=True)

        return res

    tok = doc_tokens(docs, tokens_as_hashes=True)
    tok_flat = [flatten_list(tok.values())]   # TODO: use sentences
    vocab_counts = vocabulary_counts(docs, tokens_as_hashes=True)

    embed_tokens = _create_embed_tokens_for_collocations(docs, embed_tokens_min_docfreq, embed_tokens_set)

    colloc = token_collocations(tok_flat, threshold=threshold, min_count=min_count, embed_tokens=embed_tokens,
                                vocab_counts=vocab_counts, statistic=statistic, return_statistic=False,
                                rank=None, **statistic_kwargs)

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
    :return: either original Corpus object `docs` or a modified copy of it
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
    Filter tokens according to a binary mask specified by `mask`.

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
    :return: either original Corpus object `docs` or a modified copy of it
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
    Remove tokens according to a binary mask specified by `mask`.

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
    :return: either original Corpus object `docs` or a modified copy of it
    """
    return filter_tokens_by_mask(docs, mask=mask, inverse=True, replace=replace, inplace=inplace)


def filter_tokens(docs: Corpus, /, search_tokens: Union[Any, list], by_attr: Optional[str] = None,
                  match_type: str = 'exact', ignore_case=False,
                  glob_method: str ='match', inverse=False, inplace=True):
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
    :return: either original Corpus object `docs` or a modified copy of it
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


def remove_tokens(docs: Corpus, /, search_tokens: Union[Any, list], by_attr: Optional[str] = None,
                  match_type: str = 'exact', ignore_case=False,
                  glob_method: str ='match', inplace=True):
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
    :return: either original Corpus object `docs` or a modified copy of it
    """
    return filter_tokens(docs, search_tokens=search_tokens, match_type=match_type,
                         ignore_case=ignore_case, glob_method=glob_method,
                         by_attr=by_attr, inverse=True)


def filter_for_pos(docs: Corpus, /, search_tokens: Union[Any, list], simplify_pos=True, tagset:str = 'ud',
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
    which_opts = {'common', '>', '>=', 'uncommon', '<', '<='}

    if which not in which_opts:
        raise ValueError('`which` must be one of: %s' % ', '.join(which_opts))

    if proportions:
        if not 0 <= df_threshold <= 1:
            raise ValueError('`df_threshold` must be in range [0, 1]')
    else:
        n_docs = len(docs)
        if not 0 <= df_threshold <= n_docs:
            raise ValueError(f'`df_threshold` must be in range [0, {n_docs}]')

    if which in ('common', '>='):
        comp = operator.ge
    elif which == '>':
        comp = operator.gt
    elif which == '<':
        comp = operator.lt
    else:
        comp = operator.le

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
    :return: either original Corpus object `docs` or a modified copy of it
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
    :return: either original Corpus object `docs` or a modified copy of it
    """
    return filter_tokens_by_doc_frequency(docs, which='uncommon', df_threshold=df_threshold, proportions=proportions,
                                          inverse=True, inplace=inplace)


@corpus_func_copiable
def filter_documents(docs: Corpus, /, search_tokens: Union[Any, list], by_attr: Optional[str] = None,
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
    :return: either original Corpus object `docs` or a modified copy of it
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


def remove_documents(docs: Corpus, /, search_tokens: Union[Any, list], by_attr: Optional[str] = None,
                     matches_threshold: int = 1, match_type: str = 'exact', ignore_case=False,
                     glob_method: str ='match', inverse_matches=False, inplace=True):
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
    :return: either original Corpus object `docs` or a modified copy of it
    """
    return filter_documents(docs, search_tokens=search_tokens, by_attr=by_attr, matches_threshold=matches_threshold,
                            match_type=match_type, ignore_case=ignore_case, glob_method=glob_method,
                            inverse_matches=inverse_matches, inverse_result=True)


def filter_documents_by_mask(docs: Corpus, /, mask: Dict[str, List[bool]], inverse=False, inplace=True):
    """
    Filter documents by setting a mask.

    :param docs: a Corpus object
    :param mask: dict that maps document labels to document attribute value
    :param inverse: inverse the mask
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either original Corpus object `docs` or a modified copy of it
    """
    if inverse:
        mask = {lbl: list(~np.array(m)) for lbl, m in mask.items()}

    return set_document_attr(docs, 'mask', data=mask)


def remove_documents_by_mask(docs: Corpus, /, mask: Dict[str, List[bool]], inplace=True):
    return filter_documents_by_mask(docs, mask=mask, inverse=True, inplace=inplace)


def filter_documents_by_docattr(docs: Corpus, /, search_tokens: Union[Any, list], by_attr: str,
                                match_type: str = 'exact', ignore_case=False, glob_method: str ='match',
                                inverse=False, inplace=True):
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


def remove_documents_by_docattr(docs: Corpus, /, search_tokens: Union[Any, list], by_attr: str,
                                match_type: str = 'exact', ignore_case=False, glob_method: str ='match', inplace=True):
    return filter_documents_by_docattr(docs, search_tokens=search_tokens, by_attr=by_attr, match_type=match_type,
                                       ignore_case=ignore_case, glob_method=glob_method, inverse=True, inplace=inplace)


def filter_documents_by_label(docs: Corpus, /, search_tokens: Union[Any, list], match_type: str = 'exact',
                             ignore_case=False, glob_method: str ='match', inverse=False, inplace=True):
    return filter_documents_by_docattr(docs, search_tokens=search_tokens, by_attr='label', match_type=match_type,
                                       ignore_case=ignore_case, glob_method=glob_method, inverse=inverse,
                                       inplace=inplace)


def remove_documents_by_label(docs: Corpus, /, search_tokens: Union[Any, list], match_type: str = 'exact',
                             ignore_case=False, glob_method: str ='match', inplace=True):
    return filter_documents_by_label(docs, search_tokens=search_tokens, match_type=match_type,
                                     ignore_case=ignore_case, glob_method=glob_method, inverse=True,
                                     inplace=inplace)


def filter_documents_by_length(docs: Corpus, /, relation: str, threshold: int, inverse=False, inplace=True):
    """
    Filter documents in `docs` by length, i.e. number of tokens.

    :param docs: a Corpus object
    :param relation: comparison operator as string; must be one of ``'<', '<=', '==', '>=', '>'``
    :param threshold: document length threshold in number of documents
    :param inverse: inverse the mask
    :param inplace: if True, modify Corpus object in place, otherwise return a modified copy
    :return: either original Corpus object `docs` or a modified copy of it
    """
    rel_opts = {'<', '<=', '==', '>=', '>'}
    if relation not in rel_opts:
        raise ValueError(f"`relation` must be one of {', '.join(rel_opts)}")

    if threshold < 0:
        raise ValueError("`threshold` cannot be negative")

    if relation == '>=':
        comp = operator.ge
    elif relation == '>':
        comp = operator.gt
    elif relation == '==':
        comp = operator.eq
    elif relation == '<':
        comp = operator.lt
    else:
        comp = operator.le

    mask = {lbl: comp(n, threshold) for lbl, n in doc_lengths(docs).items()}

    return filter_documents_by_mask(docs, mask=mask, inverse=inverse, inplace=inplace)


def remove_documents_by_length(docs: Corpus, /, relation: str, threshold: int, inplace=True):
    return filter_documents_by_length(docs, relation=relation, threshold=threshold, inverse=True, inplace=inplace)


@corpus_func_copiable
@corpus_func_filters_tokens
def filter_clean_tokens(docs: Corpus, /,
                        remove_punct: bool = True,
                        remove_stopwords: Union[bool, list, tuple, set] = True,
                        remove_empty: bool = True,
                        remove_shorter_than: Optional[int] = None,
                        remove_longer_than: Optional[int] = None,
                        remove_numbers: bool = False,
                        inplace=True):
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
        # the "doc masks" list holds a binary array for each document where
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

        if tokens_to_remove:
            doc_masks = [mask & np.array([t not in tokens_to_remove for t in doc], dtype=bool)
                         for mask, doc in zip(doc_masks, (doc['tokens'] for doc in chunk.values()))]

        return dict(zip(chunk.keys(), doc_masks))

    # add empty string if necessary
    tokens_to_remove = [''] if remove_empty else []

    # add stopwords
    if remove_stopwords is True:
        tokens_to_remove.extend(docs.stopwords)
    elif isinstance(remove_stopwords, (list, tuple, set)):
        tokens_to_remove.extend(remove_stopwords)

    # data preparation for parallel processing: create a dict `docs_data` with
    # doc. label -> doc. data that contains all necessary information for filtering
    # the document, depending on the filtering options
    docs_data = {}
    lengths = doc_lengths(docs)

    if tokens_to_remove:
        tokens = doc_tokens(docs, force_unigrams=True)
    else:
        tokens = None

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


def filter_tokens_with_kwic(docs: Corpus, /, search_tokens: Union[Any, list], context_size: Union[int, tuple, list] = 2,
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
    :return: either original Corpus object `docs` or a modified copy of it
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
    :return: either original Corpus object `docs` or a modified copy of it
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
    if n < 1:
        raise ValueError('`n` must be greater or equal 1')

    docs._ngrams = n
    docs._ngrams_join_str = join_str


#%% KWIC helpers

@parallelexec(collect_fn=merge_dicts)
def _build_kwic_parallel(docs, search_tokens, context_size, by_attr, match_type, ignore_case, glob_method,
                         inverse=False, highlight_keyword=None, with_window_indices=None, only_token_masks=False):
    # find matches for search criteria -> list of NumPy boolean mask arrays
    if only_token_masks:
        matchagainst = docs
    else:
        matchagainst = {lbl: d['_matchagainst'] for lbl, d in docs.items()}

    matches = _token_pattern_matches(matchagainst, search_tokens,
                                     match_type=match_type, ignore_case=ignore_case, glob_method=glob_method)

    if not only_token_masks and inverse:
        matches = {lbl: ~m for lbl, m in matches.items()}

    left, right = context_size
    matchattr = by_attr or 'token'

    kwic_res = {}
    for lbl, mask in matches.items():
        ind = np.where(mask)[0]
        ind_windows = index_windows_around_matches(mask, left, right,
                                                   flatten=only_token_masks, remove_overlaps=True)

        if only_token_masks:
            assert ind_windows.ndim == 1
            assert len(ind) <= len(ind_windows)

            # from indices back to binary mask; this only works with remove_overlaps=True
            win_mask = np.repeat(False, len(mask))
            win_mask[ind_windows] = True

            if inverse:
                win_mask = ~win_mask

            kwic_res[lbl] = win_mask
        else:
            docdata = docs[lbl]
            tok_arr = docdata.pop('_matchagainst')
            if not isinstance(tok_arr, np.ndarray) or not np.issubdtype(tok_arr.dtype, str):
                assert isinstance(tok_arr, (list, tuple, np.ndarray))
                tok_arr = as_chararray(tok_arr)

            assert len(ind) == len(ind_windows)

            windows_in_doc = []
            for match_ind, win in zip(ind, ind_windows):  # win is an array of indices into dtok_arr
                tok_win = tok_arr[win].tolist()

                if highlight_keyword is not None:
                    highlight_mask = win == match_ind
                    assert np.sum(highlight_mask) == 1
                    highlight_ind = np.where(highlight_mask)[0][0]
                    tok_win[highlight_ind] = highlight_keyword + tok_win[highlight_ind] + highlight_keyword

                win_res = {matchattr: tok_win}

                if with_window_indices:
                    win_res['index'] = win

                for attr_key, attr_vals in docdata.items():
                    if attr_key != matchattr:
                        win_res[attr_key] = attr_vals[win].tolist()

                windows_in_doc.append(win_res)

            kwic_res[lbl] = windows_in_doc

    assert len(kwic_res) == len(docs)

    return kwic_res


def _finalize_kwic_results(kwic_results, only_non_empty, glue, as_datatables, matchattr, with_attr):
    """
    Helper function to finalize raw KWIC results coming from `_build_kwic_parallel()`: Filter results,
    "glue" (join) tokens, transform to datatable, return or dismiss attributes.
    """
    kwic_results_ind = None

    if only_non_empty:
        if isinstance(kwic_results, dict):
            kwic_results = {dl: windows for dl, windows in kwic_results.items() if len(windows) > 0}
        else:
            assert isinstance(kwic_results, (list, tuple))
            kwic_results_w_indices = [(i, windows) for i, windows in enumerate(kwic_results) if len(windows) > 0]
            if kwic_results_w_indices:
                kwic_results_ind, kwic_results = zip(*kwic_results_w_indices)
            else:
                kwic_results_ind = []
                kwic_results = []

    if glue is not None:
        if isinstance(kwic_results, dict):
            return {dl: [glue.join(win[matchattr]) for win in windows] for dl, windows in kwic_results.items()}
        else:
            assert isinstance(kwic_results, (list, tuple))
            return [[glue.join(win[matchattr]) for win in windows] for windows in kwic_results]
    elif as_datatables:
        dfs = []
        if not kwic_results_ind:
            kwic_results_ind = range(len(kwic_results))

        for i_doc, dl_or_win in zip(kwic_results_ind, kwic_results):
            if isinstance(kwic_results, dict):
                dl = dl_or_win
                windows = kwic_results[dl]
            else:
                dl = i_doc
                windows = dl_or_win

            for i_win, win in enumerate(windows):
                if isinstance(win, list):
                    win = {matchattr: win}

                n_tok = len(win[matchattr])
                df_windata = [np.repeat(dl, n_tok),
                              np.repeat(i_win, n_tok),
                              win['index'],
                              win[matchattr]]

                if with_attr:
                    meta_cols = [col for col in win.keys() if col not in {matchattr, 'index'}]
                    df_windata.extend([win[col] for col in meta_cols])
                else:
                    meta_cols = []

                df_cols = ['doc', 'context', 'position', matchattr] + meta_cols
                dfs.append(pd_dt_frame(dict(zip(df_cols, df_windata))))

        if dfs:
            kwic_df = pd_dt_concat(dfs)
            return pd_dt_sort(kwic_df, ('doc', 'context', 'position'))
        else:
            return pd_dt_frame(dict(zip(['doc', 'context', 'position', matchattr], [[] for _ in range(4)])))
    elif not with_attr:
        if isinstance(kwic_results, dict):
            return {dl: [win[matchattr] for win in windows]
                    for dl, windows in kwic_results.items()}
        else:
            return [[win[matchattr] for win in windows] for windows in kwic_results]
    else:
        return kwic_results


def _datatable_from_kwic_results(kwic_results):
    """
    Helper function to transform raw KWIC results coming from `_build_kwic_parallel()` to a datatable for
    `kwic_table()`.
    """
    dfs = []

    for i_doc, dl_or_win in enumerate(kwic_results):
        if isinstance(kwic_results, dict):
            dl = dl_or_win
            windows = kwic_results[dl]
        else:
            dl = i_doc
            windows = dl_or_win

        dfs.append(pd_dt_frame(dict(zip(['doc', 'context', 'kwic'],
                                        [np.repeat(dl, len(windows)), np.arange(1, len(windows)+1), windows]))))
    if dfs:
        kwic_df = pd_dt_concat(dfs)
        return pd_dt_sort(kwic_df, ('doc', 'context'))
    else:
        return pd_dt_frame(dict(zip(['doc', 'context', 'kwic'], [[] for _ in range(3)])))


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
        if df_prop and df_prop < 0.0 or df_prop > 1.0:
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
