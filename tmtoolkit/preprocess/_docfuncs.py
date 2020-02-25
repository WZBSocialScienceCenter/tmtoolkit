"""
Functions that operate on lists of spaCy documents.
"""

from collections import Counter, OrderedDict

import numpy as np
import spacy
from spacy.tokens import Doc

from ._common import DEFAULT_LANGUAGE_MODELS
from ._tokenfuncs import token_match, make_index_window_around_matches
from ..utils import empty_chararray, flatten_list, require_listlike
from ..bow.dtm import create_sparse_dtm
from .._pd_dt_compat import pd_dt_frame, pd_dt_concat, pd_dt_sort


#%% global spaCy nlp instance

#: Global spaCy nlp instance which must be initiated via :func:`tmtoolkit.preprocess.init_for_language` when using
#: the functional preprocess API
nlp = None


#%% initialization and tokenization

def init_for_language(language=None, language_model=None, **spacy_opts):
    """
    Initialize the functional API for a given language code `language` or a spaCy language model `language_model`.
    The spaCy nlp instance will be returned and will also be used by default in all subsequent preprocess API calls.

    :param language: two-letter ISO 639-1 language code (lowercase)
    :param language_model: spaCy language model `language_model`
    :param spacy_opts: additional keyword arguments passed to ``spacy.load()``
    :return: spaCy nlp instance
    """
    if language is None and language_model is None:
        raise ValueError('either `language` or `language_model` must be given')

    if language_model is None:
        if not isinstance(language, str) or len(language) != 2:
            raise ValueError('`language` must be a two-letter ISO 639-1 language code')

        if language not in DEFAULT_LANGUAGE_MODELS:
            raise ValueError('language "%s" is not supported' % language)
        language_model = DEFAULT_LANGUAGE_MODELS[language]

    spacy_kwargs = dict(disable=['parser', 'ner'])
    spacy_kwargs.update(spacy_opts)

    global nlp
    nlp = spacy.load(language_model, **spacy_kwargs)

    return nlp


def tokenize(docs, as_spacy_docs=False, doc_labels=None, doc_labels_fmt='doc-{i1}', nlp_instance=None):
    """
    Tokenize a list or dict of documents `docs`, where each element contains the raw text of the document as string.

    Requires that :func:`~tmtoolkit.preprocess.init_for_language` is called before or `nlp_instance` is passed.

    :param docs: list or dict of documents with raw text strings; if dict, use dict keys as document labels
    :param as_spacy_docs: if True, return list of spaCy ``Doc`` objects, otherwise return list of string tokens
    :param doc_labels: if not None and `docs` is a list, use strings in this list as document labels
    :param doc_labels_fmt: if `docs` is a list and `doc_labels` is None, generate document labels according to this
                           format, where ``{i0}`` or ``{i1}`` are replaced by the respective zero- or one-indexed
                           document numbers
    :param nlp_instance: spaCy nlp instance
    :return: list of string tokens (default) or list of spaCy ``Doc`` objects if `as_spacy_docs` is True
    """

    dictlike = hasattr(docs, 'keys') and hasattr(docs, 'values')

    if not isinstance(docs, (list, tuple)) and not dictlike:
        raise ValueError('`docs` must be a list, tuple or dict-like object')

    if not isinstance(doc_labels_fmt, str):
        raise ValueError('`doc_labels_fmt` must be a string')

    _nlp = _current_nlp(nlp_instance)

    if doc_labels is None:
        if dictlike:
            doc_labels = docs.keys()
            docs = docs.values()
        elif as_spacy_docs:
            doc_labels = [doc_labels_fmt.format(i0=i, i1=i+1) for i in range(len(docs))]
    elif len(doc_labels) != len(docs):
        raise ValueError('`doc_labels` must have same length as `docs`')

    tokenized_docs = [_nlp.make_doc(d) for d in docs]
    del docs

    if as_spacy_docs:
        for dl, doc in zip(doc_labels, tokenized_docs):
            doc._.label = dl
            _init_doc(doc)

        return tokenized_docs
    else:
        return [[t.text for t in doc] for doc in tokenized_docs]


#%% functions that operate on lists of string tokens *or* spacy documents

def doc_tokens(docs):
    """
    If `docs` is a list of spaCy documents, return the tokens from these documents as list of string tokens, otherwise
    return the input list as-is.

    :param docs: list of string tokens or spaCy documents
    :return: list of string tokens
    """
    require_spacydocs_or_tokens(docs)

    return list(map(_filtered_doc_tokens, docs))


def doc_lengths(docs):
    """
    Return document length (number of tokens in doc.) for each document.

    :param docs: list of string tokens or spaCy documents
    :return: list of document lengths
    """
    require_spacydocs_or_tokens(docs)

    return list(map(len, doc_tokens(docs)))


def vocabulary(docs, sort=False):
    """
    Return vocabulary, i.e. set of all tokens that occur at least once in at least one of the documents in `docs`.

    :param docs: list of string tokens or spaCy documents
    :param sort: return as sorted list
    :return: either set of token strings or sorted list if `sort` is True
    """
    require_spacydocs_or_tokens(docs)

    v = set(flatten_list(doc_tokens(docs)))

    if sort:
        return sorted(v)
    else:
        return v


def vocabulary_counts(docs):
    """
    Return :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
    all documents.

    :param docs: list of string tokens or spaCy documents
    :return: :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
             all documents
    """
    require_spacydocs_or_tokens(docs)

    return Counter(flatten_list(doc_tokens(docs)))


def doc_frequencies(docs, proportions=False):
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
        df(x) = df(w) = 1 (occur only in A)
        df(y) = 2 (occurs in B and C)
        ...

    :param docs: list of string tokens or spaCy documents
    :param proportions: if True, normalize by number of documents to obtain proportions
    :return: dict mapping token to document frequency
    """
    require_spacydocs_or_tokens(docs)

    doc_freqs = Counter()

    for dtok in docs:
        for t in set(_filtered_doc_tokens(dtok)):
            doc_freqs[t] += 1

    if proportions:
        n_docs = len(docs)
        return {w: n/n_docs for w, n in doc_freqs.items()}
    else:
        return doc_freqs


def ngrams(docs, n, join=True, join_str=' '):
    """
    Generate and return n-grams of length `n`.

    :param docs: list of string tokens or spaCy documents
    :param n: length of n-grams, must be >= 2
    :param join: if True, join generated n-grams by string `join_str`
    :param join_str: string used for joining
    :return: list of n-grams; if `join` is True, the list contains strings of joined n-grams, otherwise the list
             contains lists of size `n` in turn containing the strings that make up the n-gram
    """
    require_spacydocs_or_tokens(docs)

    return [_ngrams_from_tokens(_filtered_doc_tokens(dtok, as_list=True), n=n, join=join, join_str=join_str)
            for dtok in docs]


def sparse_dtm(docs, vocab=None):
    """
    Create a sparse document-term-matrix (DTM) from a list of tokenized documents `docs`. If `vocab` is None, determine
    the vocabulary (unique terms) from `docs`, otherwise take `vocab` which must be a *sorted* list or NumPy array.
    If `vocab` is None, the generated sorted vocabulary list is returned as second value, else only a single value is
    returned -- the DTM.

    :param docs: list of string tokens or spaCy documents
    :param vocab: optional *sorted* list / NumPy array of vocabulary (unique terms) in `docs`
    :return: either a single value (sparse document-term-matrix) or a tuple with sparse DTM and sorted vocabulary if
             none was passed
    """
    require_spacydocs_or_tokens(docs)

    if vocab is None:
        vocab = vocabulary(docs, sort=True)
        return_vocab = True
    else:
        return_vocab = False

    tokens = doc_tokens(docs)
    alloc_size = sum(len(set(dtok)) for dtok in tokens)  # sum of *unique* tokens in each document

    dtm = create_sparse_dtm(vocab, tokens, alloc_size, vocab_is_sorted=True)

    if return_vocab:
        return dtm, vocab
    else:
        return dtm


def kwic(docs, search_tokens, context_size=2, match_type='exact', ignore_case=False,
         glob_method='match', inverse=False, with_metadata=False, as_dict=False, as_datatable=False, non_empty=False,
         glue=None, highlight_keyword=None):
    """
    Perform keyword-in-context (kwic) search for search pattern(s) `search_tokens`. Returns result as list of KWIC
    windows or datatable / dataframe. If you want to filter with KWIC, use
    :func:`~tmtoolkit.preprocess.filter_tokens_with_kwic`, which returns results as list of tokenized documents (same
    structure as `docs`).

    Uses similar search parameters as :func:`~tmtoolkit.preprocess.filter_tokens`.

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
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
    :param with_metadata: also return metadata (like POS) along with each token
    :param as_dict: if True, return result as dict with document labels mapping to KWIC results
    :param as_datatable: return result as data frame with indices "doc" (document label) and "context" (context
                          ID per document) and optionally "position" (original token position in the document) if
                          tokens are not glued via `glue` parameter
    :param non_empty: if True, only return non-empty result documents
    :param glue: if not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: if not None, this must be a string which is used to indicate the start and end of the
                              matched keyword
    :return: return either as: (1) list with KWIC results per document, (2) as dict with document labels mapping to
             KWIC results when `as_dict` is True or (3) dataframe / datatable when `as_datatable` is True
    """
    require_spacydocs_or_tokens(docs)

    if isinstance(context_size, int):
        context_size = (context_size, context_size)
    else:
        require_listlike(context_size)

    if highlight_keyword is not None and not isinstance(highlight_keyword, str):
        raise ValueError('if `highlight_keyword` is given, it must be of type str')

    if glue:
        if with_metadata or as_datatable:
            raise ValueError('when `glue` is set to True, `with_metadata` and `as_datatable` must be False')
        if not isinstance(glue, str):
            raise ValueError('if `glue` is given, it must be of type str')

    kwic_raw = _build_kwic(docs, search_tokens,
                           highlight_keyword=highlight_keyword,
                           with_metadata=with_metadata,
                           with_window_indices=as_datatable,
                           context_size=context_size,
                           match_type=match_type,
                           ignore_case=ignore_case,
                           glob_method=glob_method,
                           inverse=inverse)

    if as_dict or as_datatable:
        kwic_raw = dict(zip(doc_labels(docs), kwic_raw))

    return _finalize_kwic_results(kwic_raw,
                                  non_empty=non_empty,
                                  glue=glue,
                                  as_datatable=as_datatable,
                                  with_metadata=with_metadata)


def kwic_table(docs, search_tokens, context_size=2, match_type='exact', ignore_case=False,
               glob_method='match', inverse=False, glue=' ', highlight_keyword='*'):
    """
    Shortcut for :func:`~tmtoolkit.preprocess.kwic()` to directly return a data frame table with highlighted keywords
    in context.

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: single string or list of strings that specify the search pattern(s)
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`).
    :param inverse: Invert the matching results.
    :param glue: If not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                              matched keyword.
    :return: datatable or pandas DataFrame with columns "doc" (document label), "context" (context ID per document) and
             "kwic" containing strings with highlighted keywords in context.
    """

    kwic_raw = kwic(docs, search_tokens,
                    context_size=context_size,
                    match_type=match_type,
                    ignore_case=ignore_case,
                    glob_method=glob_method,
                    inverse=inverse,
                    with_metadata=False,
                    as_dict=True,
                    as_datatable=False,
                    non_empty=True,
                    glue=glue,
                    highlight_keyword=highlight_keyword)

    return _datatable_from_kwic_results(kwic_raw)


#%% functions that operate *only* on lists of spacy documents

def doc_labels(docs):
    """
    Return list of document labels that are assigned to spaCy documents `docs`.

    :param docs: list of spaCy documents
    :return: list of document labels
    """
    require_spacydocs(docs)

    return [d._.label for d in docs]


#%% helper functions


def _current_nlp(nlp_instance):
    _nlp = nlp_instance or nlp
    if not _nlp:
        raise ValueError('neither global nlp instance is set, nor `nlp_instance` argument is given; did you call '
                         '`init_for_language()` before?')
    return _nlp


def _init_doc(doc, tokens=None, mask=None):
    assert isinstance(doc, Doc)

    if tokens is None:
        tokens = [t.text for t in doc]
    else:
        assert isinstance(tokens, (list, tuple, np.ndarray))
        assert len(doc) == len(tokens)

    if mask is not None:
        assert isinstance(mask, np.ndarray)
        assert np.issubdtype(mask.dtype, np.bool_)
        assert len(doc) == len(mask)

    if len(tokens) == 0:
        doc.user_data['tokens'] = empty_chararray()
    else:
        doc.user_data['tokens'] = np.array(tokens) if not isinstance(tokens, np.ndarray) else tokens
    doc.user_data['mask'] = mask if isinstance(mask, np.ndarray) else np.repeat(True, len(doc))


def _get_docs_tokenattrs_keys(docs, default_attrs=None):
    if default_attrs:
        attrs = default_attrs
    else:
        attrs = ['lemma_', 'whitespace_']

    if len(docs) > 0:
        for doc in docs:
            if doc.is_tagged and 'pos_' not in attrs:
                attrs.append('pos_')

            if len(doc) > 0:
                firsttok = next(iter(doc))
                attrs.extend([attr for attr in dir(firsttok._)
                             if attr not in {'has', 'get', 'set'} and not attr.startswith('_')])
                break

    return attrs


def _filtered_doc_arr(lst, doc):
    return np.array(lst)[doc.user_data['mask']]


def _filtered_doc_tokens(doc, as_list=False):
    if isinstance(doc, list):
        return doc
    else:
        res = doc.user_data['tokens'][doc.user_data['mask']]
        return res.tolist() if as_list else res


def _ngrams_from_tokens(tokens, n, join=True, join_str=' '):
    """
    Helper function to produce ngrams of length `n` from a list of string tokens `tokens`.

    :param tokens: list of string tokens
    :param n: size of ngrams
    :param join: if True, join each ngram by `join_str`, i.e. return list of ngram strings; otherwise return list of
                 ngram lists
    :param join_str: if `join` is True, use this string to join the parts of the ngrams
    :return: return list of ngram strings if `join` is True, otherwise list of ngram lists
    """
    if n < 2:
        raise ValueError('`n` must be at least 2')

    if len(tokens) == 0:
        return []

    if len(tokens) < n:
        # raise ValueError('`len(tokens)` should not be smaller than `n`')
        ngrams = [tokens]
    else:
        ngrams = [[tokens[i+j] for j in range(n)]
                  for i in range(len(tokens)-n+1)]
    if join:
        return list(map(lambda x: join_str.join(x), ngrams))
    else:
        return ngrams


def _match_against(docs, by_meta=None):
    """Return the list of values to match against in filtering functions."""
    if by_meta:
        return [_filtered_doc_arr([getattr(t._, by_meta) for t in doc], doc) for doc in docs]
    else:
        return [_filtered_doc_tokens(doc) for doc in docs]


def _token_pattern_matches(tokens, search_tokens, match_type, ignore_case, glob_method):
    """
    Helper function to apply `token_match` with multiple patterns in `search_tokens` to `docs`.
    The matching results for each pattern in `search_tokens` are combined via logical OR.
    Returns a list of length `docs` containing boolean arrays that signal the pattern matches for each token in each
    document.
    """
    if not isinstance(search_tokens, (list, tuple, set)):
        search_tokens = [search_tokens]

    matches = [np.repeat(False, repeats=len(dtok)) for dtok in tokens]

    for dtok, dmatches in zip(tokens, matches):
        for pat in search_tokens:
            pat_match = token_match(pat, dtok, match_type=match_type, ignore_case=ignore_case, glob_method=glob_method)

            dmatches |= pat_match

    return matches


def _build_kwic(docs, search_tokens, context_size, match_type, ignore_case, glob_method, inverse,
                highlight_keyword=None, with_metadata=False, with_window_indices=False,
                only_token_masks=False):
    """
    Helper function to build keywords-in-context (KWIC) results from documents `docs`.

    :param docs: list of string tokens or spaCy documents
    :param search_tokens: search pattern(s)
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's :func:`re.match` or :func:`re.search`).
    :param inverse: Invert the matching results.
    :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                              matched keyword.
    :param with_metadata: add document metadata to KWIC results
    :param with_window_indices: add window indices to KWIC results
    :param only_token_masks: return only flattened token masks for filtering
    :return: list with KWIC results per document
    """

    # find matches for search criteria -> list of NumPy boolean mask arrays
    matches = _token_pattern_matches(_match_against(docs), search_tokens, match_type=match_type,
                                     ignore_case=ignore_case, glob_method=glob_method)

    if not only_token_masks and inverse:
        matches = [~m for m in matches]

    left, right = context_size

    kwic_list = []
    for mask, doc in zip(matches, docs):
        ind = np.where(mask)[0]
        ind_windows = make_index_window_around_matches(mask, left, right,
                                                       flatten=only_token_masks, remove_overlaps=True)

        if only_token_masks:
            assert ind_windows.ndim == 1
            assert len(ind) <= len(ind_windows)

            # from indices back to binary mask; this only works with remove_overlaps=True
            win_mask = np.repeat(False, len(doc))
            win_mask[ind_windows] = True

            if inverse:
                win_mask = ~win_mask

            kwic_list.append(win_mask)
        else:
            doc_arr = _filtered_doc_tokens(doc)

            assert len(ind) == len(ind_windows)

            windows_in_doc = []
            for match_ind, win in zip(ind, ind_windows):  # win is an array of indices into dtok_arr
                tok_win = doc_arr[win].tolist()

                if highlight_keyword is not None:
                    highlight_mask = win == match_ind
                    assert np.sum(highlight_mask) == 1
                    highlight_ind = np.where(highlight_mask)[0][0]
                    tok_win[highlight_ind] = highlight_keyword + tok_win[highlight_ind] + highlight_keyword

                win_res = {'token': tok_win}

                if with_window_indices:
                    win_res['index'] = win

                if with_metadata:
                    attr_keys = _get_docs_tokenattrs_keys(docs)
                    attr_vals = {}
                    attr_keys_base = []
                    attr_keys_ext = []

                    for attr in attr_keys:
                        baseattr = attr.endswith('_')

                        if baseattr:
                            attrlabel = attr[:-1]   # remove trailing underscore
                            attr_keys_base.append(attrlabel)
                        else:
                            attrlabel = attr
                            attr_keys_ext.append(attrlabel)

                        if attr == 'whitespace_':
                            attrdata = [bool(t.whitespace_) for t in doc]
                        else:
                            attrdata = [getattr(t if baseattr else t._, attr) for t in doc]

                        attr_vals[attrlabel] = np.array(attrdata)[win].tolist()

                    for attrlabel in list(sorted(attr_keys_base)) + list(sorted(attr_keys_ext)):
                        win_res[attrlabel] = attr_vals[attrlabel]

                windows_in_doc.append(win_res)

            kwic_list.append(windows_in_doc)

    assert len(kwic_list) == len(docs)

    return kwic_list


def _finalize_kwic_results(kwic_results, non_empty, glue, as_datatable, with_metadata):
    """
    Helper function to finalize raw KWIC results coming from `_build_kwic()`: Filter results, "glue" (join) tokens,
    transform to datatable, return or dismiss metadata.
    """
    kwic_results_ind = None

    if non_empty:
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
            return {dl: [glue.join(win['token']) for win in windows] for dl, windows in kwic_results.items()}
        else:
            assert isinstance(kwic_results, (list, tuple))
            return [[glue.join(win['token']) for win in windows] for windows in kwic_results]
    elif as_datatable:
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
                    win = {'token': win}

                n_tok = len(win['token'])
                df_windata = [np.repeat(dl, n_tok),
                              np.repeat(i_win, n_tok),
                              win['index'],
                              win['token']]

                if with_metadata:
                    meta_cols = [col for col in win.keys() if col not in {'token', 'index'}]
                    df_windata.extend([win[col] for col in meta_cols])
                else:
                    meta_cols = []

                df_cols = ['doc', 'context', 'position', 'token'] + meta_cols
                dfs.append(pd_dt_frame(OrderedDict(zip(df_cols, df_windata))))

        if dfs:
            kwic_df = pd_dt_concat(dfs)
            return pd_dt_sort(kwic_df, ('doc', 'context', 'position'))
        else:
            return pd_dt_frame(OrderedDict(zip(['doc', 'context', 'position', 'token'], [[] for _ in range(4)])))
    elif not with_metadata:
        if isinstance(kwic_results, dict):
            return {dl: [win['token'] for win in windows]
                    for dl, windows in kwic_results.items()}
        else:
            return [[win['token'] for win in windows] for windows in kwic_results]
    else:
        return kwic_results


def _datatable_from_kwic_results(kwic_results):
    """
    Helper function to transform raw KWIC results coming from `_build_kwic()` to a datatable for `kwic_table()`.
    """
    dfs = []

    for i_doc, dl_or_win in enumerate(kwic_results):
        if isinstance(kwic_results, dict):
            dl = dl_or_win
            windows = kwic_results[dl]
        else:
            dl = i_doc
            windows = dl_or_win

        dfs.append(pd_dt_frame(OrderedDict(zip(['doc', 'context', 'kwic'],
                                               [np.repeat(dl, len(windows)), np.arange(len(windows)), windows]))))
    if dfs:
        kwic_df = pd_dt_concat(dfs)
        return pd_dt_sort(kwic_df, ('doc', 'context'))
    else:
        return pd_dt_frame(OrderedDict(zip(['doc', 'context', 'kwic'], [[] for _ in range(3)])))


def require_spacydocs(docs, types=(Doc, ), error_msg='the argument must be a list of spaCy documents'):
    require_listlike(docs)

    if docs:
        first_doc = next(iter(docs))
        if not isinstance(first_doc, types):
            raise ValueError(error_msg)


def require_spacydocs_or_tokens(docs):
    require_spacydocs(docs, (Doc, list, np.ndarray), error_msg='the argument must be a list of string token documents '
                                                               'or spaCy documents')
