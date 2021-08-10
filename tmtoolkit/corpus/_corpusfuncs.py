import os
import unicodedata
from copy import copy
from functools import partial, wraps
from inspect import signature
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Union, List, Callable, Iterable, Optional, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from spacy.tokens import Doc

from ..bow.dtm import create_sparse_dtm, dtm_to_dataframe, dtm_to_datatable
from ..utils import merge_dicts, merge_counters, empty_chararray, as_chararray, \
    flatten_list, combine_sparse_matrices_columnwise, arr_replace, pickle_data, unpickle_file, merge_sets
from .._pd_dt_compat import USE_DT, FRAME_TYPE, pd_dt_frame, pd_dt_concat, pd_dt_sort, pd_dt_colnames

from ._common import LANGUAGE_LABELS
from ._corpus import Corpus
from ._tokenfuncs import ngrams_from_tokenlist
from ._helpers import _filtered_doc_attr, _filtered_doc_tokens, _corpus_from_tokens_metadata, \
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
        fn(corp, *args[1:], **kwargs)
        return corp

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
        fn(corp, *args[1:], **kwargs)

        corp._tokens_processed = True

        return corp

    return inner_fn

#%% Corpus functions with readonly access to Corpus data


def doc_tokens(docs: Union[Corpus, Dict[str, Doc]],
               only_non_empty=False,
               tokens_as_hashes=False,
               with_metadata: Union[bool, list, tuple] = False,
               with_mask=False,
               as_datatables=False, as_arrays=False,
               apply_token_filter=True,
               apply_document_filter=True,
               force_unigrams=False) \
        -> Dict[str, Union[List[str], dict, FRAME_TYPE]]:
    if with_mask and not with_metadata:
        with_metadata = ['mask']
    mask_in_meta = isinstance(with_metadata, (list, tuple)) and 'mask' in with_metadata
    with_mask = with_mask or mask_in_meta

    if with_mask:  # requesting the token mask disables the token filtering
        apply_token_filter = False
        apply_document_filter = False

    ng = 1
    ng_join_str = None
    doc_attrs = {}

    if isinstance(docs, Corpus):
        if not force_unigrams:
            ng = docs.ngrams
            ng_join_str = docs.ngrams_join_str
            doc_attrs = docs.doc_attrs_defaults
        docs = docs.spacydocs_ignore_filter if with_mask else docs.spacydocs

    if with_mask and 'mask' not in doc_attrs.keys():
        doc_attrs['mask'] = True

    if isinstance(with_metadata, (list, tuple)):
        doc_attrs = {k: doc_attrs[k] for k in with_metadata}

    res = {}
    for lbl, d in docs.items():
        if (only_non_empty and len(d) == 0) or (apply_document_filter and not d._.mask):
            continue

        tok = _filtered_doc_tokens(d, tokens_as_hashes=tokens_as_hashes, apply_filter=apply_token_filter)

        if ng > 1:
            tok = ngrams_from_tokenlist(tok, n=ng, join=True, join_str=ng_join_str)

        if with_metadata is not False:
            resdoc = {}

            for k, default in doc_attrs.items():
                a = 'doc_mask' if k == 'mask' else k
                v = getattr(d._, k, default)
                resdoc[a] = [v] * len(tok) if as_datatables else v

            resdoc['token'] = tok

            if isinstance(with_metadata, (list, tuple)):
                spacy_attrs = [k for k in with_metadata if not k.startswith('meta_') and k != 'mask']
            else:
                spacy_attrs = Corpus.STD_TOKEN_ATTRS

            for k in spacy_attrs:
                v = _filtered_doc_attr(d, k, apply_filter=apply_token_filter)
                if k == 'whitespace':
                    v = list(map(lambda ws: ws != '', v))
                if ng > 1:
                    v = ngrams_from_tokenlist(list(map(str, v)), n=ng, join=True, join_str=ng_join_str)
                resdoc[k] = v

            user_attrs = d.user_data.keys()
            if isinstance(with_metadata, (list, tuple)):
                user_attrs = [k for k in user_attrs if k in with_metadata]

            if with_mask and 'mask' not in user_attrs:
                user_attrs.append('mask')

            for k in user_attrs:
                if isinstance(k, str) and (k.startswith('meta_') or (with_mask and k == 'mask')):
                    v = _filtered_doc_attr(d, k, custom=True, apply_filter=apply_token_filter)
                    if not as_datatables and not as_arrays:
                        v = list(v)
                    if ng > 1:
                        v = ngrams_from_tokenlist(list(map(str, v)), n=ng, join=True, join_str=ng_join_str)
                    resdoc[k] = v
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


def vocabulary(docs: Union[Corpus, Dict[str, List[str]]], as_hashes=False, force_unigrams=False, sort=False)\
        -> Union[set, list]:
    if isinstance(docs, Corpus):
        tok = doc_tokens(docs, tokens_as_hashes=as_hashes, force_unigrams=force_unigrams).values()
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


def vocabulary_size(docs: Corpus, force_unigrams=False) -> int:
    return len(vocabulary(docs, force_unigrams=force_unigrams))


def tokens_with_metadata(docs: Corpus) -> Dict[str, FRAME_TYPE]:
    return doc_tokens(docs, with_metadata=True, as_datatables=True)


def tokens_datatable(docs: Corpus, with_metadata: Union[bool, list, tuple, set] = True, with_mask=False)\
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

    tokens = doc_tokens(docs, only_non_empty=False, with_metadata=with_metadata or [],
                        with_mask=with_mask, as_datatables=True)
    dfs = _tokens_datatable(_paralleltask(docs, tokens))

    if dfs:
        res = pd_dt_concat(dfs)
    else:
        res = pd_dt_frame({'doc': [], 'position': [], 'token': []})

    return pd_dt_sort(res, ['doc', 'position'])


def tokens_dataframe(docs: Corpus, with_metadata: Union[bool, list, tuple, set] = True, with_mask=False)\
        -> pd.DataFrame:
    # note that generating a datatable first and converting it to pandas is faster than generating a pandas data
    # frame right away

    df = tokens_datatable(docs, with_metadata=with_metadata, with_mask=with_mask)

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

    summary += f'\ntotal number of tokens: {n_tokens(docs)} / vocabulary size: {vocabulary_size(docs)}'

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
        return {dl: ngrams_from_tokenlist(dt, n, join, join_str) for dl, dt in tokens.items()}

    return _ngrams(_paralleltask(docs))

#%% Corpus I/O


def save_corpus_to_picklefile(docs: Corpus, picklefile: str):
    pickle_data(serialize_corpus(docs, deepcopy_attrs=False), picklefile)


def load_corpus_from_picklefile(picklefile: str) -> Corpus:
    return deserialize_corpus(unpickle_file(picklefile))


def load_corpus_from_tokens(tokens: Dict[str, Union[list, tuple, Dict[str, List]]], **corpus_kwargs):
    if 'docs' in corpus_kwargs:
        raise ValueError('`docs` parameter is obsolete when initializing a Corpus with this function')

    corp = Corpus(**corpus_kwargs)
    _corpus_from_tokens_metadata(corp, tokens)   # TODO: also handle document attributes

    return corp


def load_corpus_from_tokens_datatable(tokens: FRAME_TYPE, **corpus_kwargs):  # TODO: also handle document attributes
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
def set_document_attr(docs: Corpus, /, attrname: str, data: Dict[str, Any], default=None, inplace=True):
    if attrname in docs.token_attrs:
        raise ValueError(f'attribute name "{attrname}" is already used as token attribute')

    if not Doc.has_extension(attrname):
        Doc.set_extension(attrname, default=default, force=True)

    for lbl, val in data.items():
        if lbl not in docs.spacydocs_ignore_filter.keys():
            raise ValueError(f'document "{lbl}" does not exist in Corpus object `docs`')

        setattr(docs.spacydocs_ignore_filter[lbl]._, attrname, val)

    if attrname not in {'label', 'mask'}:
        docs._doc_attrs_defaults[attrname] = default


@corpus_func_copiable
def add_metadata_per_token(docs: Corpus, /, key: str, data: dict, default=None, inplace=True):
    # convert data token string keys to token hashes
    data = {docs.nlp.vocab.strings[k]: v for k, v in data.items()}

    key = 'meta_' + key

    for d in docs.spacydocs.values():    # TODO: handle scenario where documents are filtered
        d.user_data[key] = np.array([data.get(hash, default) if mask else default
                                     for mask, hash in zip(d.user_data['mask'], d.user_data['processed'])])

    docs._token_attrs_defaults[key] = default


@corpus_func_copiable
@corpus_func_processes_tokens
def transform_tokens(docs: Corpus, /, func: Callable, inplace=True, **kwargs):
    vocab = vocabulary(docs, as_hashes=True, force_unigrams=True)
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


#%% Corpus functions that modify corpus data: filtering

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


def filter_tokens(docs: Corpus, /, search_tokens: Union[Any, List[Any]], by_meta: Optional[str] = None,
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
    :param by_meta: if not None, this should be a string of a meta data key; this meta data will then be
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
        masks = _filter_tokens(_paralleltask(docs, _match_against(docs.spacydocs, by_meta)))
    except AttributeError:
        raise AttributeError(f'token meta data key "{by_meta}" does not exist')

    return filter_tokens_by_mask(docs, masks, inverse=inverse)


def remove_tokens(docs: Corpus, /, search_tokens: Union[Any, List[Any]], by_meta: Optional[str] = None,
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
    :param by_meta: if not None, this should be a string of a meta data key; this meta data will then be
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
                         by_meta=by_meta, inverse=True)


@corpus_func_copiable
def filter_documents(docs: Corpus, /, search_tokens: Union[Any, List[Any]], by_meta: Optional[str] = None,
                     matches_threshold: int = 1, match_type: str = 'exact', ignore_case=False,
                     glob_method: str ='match', inverse_result=False, inverse_matches=False, inplace=True):
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
    :param by_meta: if not None, this should be a string of a meta data key; this meta data will then be
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
        remove = _filter_documents(_paralleltask(docs, _match_against(docs.spacydocs, by_meta)))
    except AttributeError:
        raise AttributeError(f'token meta data key "{by_meta}" does not exist')

    return filter_documents_by_mask(docs, mask=dict(zip(remove, [False] * len(remove))))


def remove_documents(docs: Corpus, /, search_tokens: Union[Any, List[Any]], by_meta: Optional[str] = None,
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
    :param by_meta: if not None, this should be a string of a meta data key; this meta data will then be
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
    return filter_documents(docs, search_tokens=search_tokens, by_meta=by_meta, matches_threshold=matches_threshold,
                            match_type=match_type, ignore_case=ignore_case, glob_method=glob_method,
                            inverse_matches=inverse_matches, inverse_result=True)


def filter_documents_by_mask(docs: Corpus, /, mask: Dict[str, List[bool]], inverse=False):
    if inverse:
        mask = {lbl: list(~np.array(m)) for lbl, m in mask.items()}

    return set_document_attr(docs, 'mask', data=mask)


def remove_documents_by_mask(docs: Corpus, /, mask: Dict[str, List[bool]]):
    return filter_documents_by_mask(docs, mask=mask, inverse=True)


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
    tok = doc_tokens(docs, with_metadata=True, force_unigrams=True,
                     apply_token_filter=which in {'all', 'tokens'},
                     apply_document_filter=which in {'all', 'documents'})
    _corpus_from_tokens_metadata(docs, tok)   # re-create spacy docs
    if which != 'documents':
        docs._tokens_masked = False
    docs._tokens_processed = False


@corpus_func_copiable
def ngramify(docs: Corpus, /, n: int, join_str=' ', inplace=True):
    if n < 1:
        raise ValueError('`n` must be greater or equal 1')

    docs._ngrams = n
    docs._ngrams_join_str = join_str

