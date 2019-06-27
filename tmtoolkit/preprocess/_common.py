"""
Common functions and constants.
"""

from collections import Counter, OrderedDict

import numpy as np
import datatable as dt
import nltk

from .. import defaults
from ..bow.dtm import create_sparse_dtm
from ..utils import flatten_list, require_listlike, token_match, make_index_window_around_matches


PATTERN_SUBMODULES = {
    'english': 'en',
    'german': 'de',
    'spanish': 'es',
    'french': 'fr',
    'italian': 'it',
    'dutch': 'nl',
}


def tokenize(docs, language=defaults.language):
    require_listlike(docs)

    return [nltk.tokenize.word_tokenize(text, language) for text in docs]


def doc_lengths(docs):
    return list(map(len, docs))


def vocabulary(docs, sort=False):
    v = set(flatten_list(docs))

    if sort:
        return sorted(v)
    else:
        return v


def vocabulary_counts(docs):
    return Counter(flatten_list(docs))


def doc_frequencies(docs, proportions=False):
    doc_freqs = Counter()

    for dtok in docs:
        for t in set(dtok):
            doc_freqs[t] += 1

    if proportions:
        n_docs = len(docs)
        return {w: n/n_docs for w, n in doc_freqs.items()}
    else:
        return doc_freqs


def ngrams(docs, n, join=True, join_str=' '):
    return [_ngrams_from_tokens(dtok, n=n, join=join, join_str=join_str) for dtok in docs]


def sparse_dtm(docs, vocab=None):
    """
    Create a sparse document-term-matrix (DTM) from a list of tokenized documents `docs`. If `vocab` is None, determine
    the vocabulary (unique terms) from `docs`, otherwise take `vocab` which must be a *sorted* list or NumPy array.
    If `vocab` is None, the generated sorted vocabulary list is returned as second value, else only a single value is
    returned -- the DTM.

    :param docs: list of tokenized documents
    :param vocab: optional *sorted* list / NumPy array of vocabulary (unique terms) in `docs`
    :return: either a single value (sparse document-term-matrix) or a tuple with sparse DTM and sorted vocabulary if
             none was passed
    """

    if vocab is None:
        vocab = vocabulary(docs, sort=True)
        return_vocab = True
    else:
        return_vocab = False

    alloc_size = sum(len(set(dtok)) for dtok in docs)  # sum of *unique* tokens in each document

    dtm = create_sparse_dtm(vocab, docs, alloc_size, vocab_is_sorted=True)

    if return_vocab:
        return dtm, vocab
    else:
        return dtm


def kwic(docs, search_token, context_size=2, match_type='exact', ignore_case=False, glob_method='match',
         inverse=False, with_metadata=False, as_data_table=False, non_empty=False, glue=None,
         highlight_keyword=None):
    """
    Perform keyword-in-context (kwic) search for `search_token`. Uses similar search parameters as
    `filter_tokens()`.

    :param docs: list of tokenized documents, optionally as 2-tuple where each element in `docs` is a tuple
                 of (tokens list, tokens metadata dict)
    :param search_token: search pattern
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's `re.match` or `re.search`).
    :param inverse: Invert the matching results.
    :param with_metadata: Also return metadata (like POS) along with each token.
    :param as_data_table: Return result as data frame with indices "doc" (document label) and "context" (context
                          ID per document) and optionally "position" (original token position in the document) if
                          tokens are not glued via `glue` parameter.
    :param non_empty: If True, only return non-empty result documents.
    :param glue: If not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                              matched keyword.
    :return: Return list with KWIC results per document or a data frame, depending
    on `as_data_table`.
    """
    if isinstance(context_size, int):
        context_size = (context_size, context_size)
    else:
        require_listlike(context_size)

    if highlight_keyword is not None and not isinstance(highlight_keyword, str):
        raise ValueError('if `highlight_keyword` is given, it must be of type str')

    if glue:
        if with_metadata or as_data_table:
            raise ValueError('when `glue` is set to True, `with_metadata` and `as_data_table` must be False')
        if not isinstance(glue, str):
            raise ValueError('if `glue` is given, it must be of type str')

    kwic_raw = _build_kwic(docs, search_token,
                           highlight_keyword=highlight_keyword,
                           with_metadata=with_metadata,
                           with_window_indices=as_data_table,
                           context_size=context_size,
                           match_type=match_type,
                           ignore_case=ignore_case,
                           glob_method=glob_method,
                           inverse=inverse)

    return _finalize_kwic_results(kwic_raw,
                                  non_empty=non_empty,
                                  glue=glue,
                                  as_data_table=as_data_table,
                                  with_metadata=with_metadata)


def kwic_table(docs, search_token, context_size=2, match_type='exact', ignore_case=False, glob_method='match',
               inverse=False, glue=' ', highlight_keyword='*'):
    """
    Shortcut for `get_kwic` to directly return a data frame table with highlighted keywords in context.

    :param docs: list of tokenized documents, optionally as 2-tuple where each element in `docs` is a tuple
                 of (tokens list, tokens metadata dict)
    :param search_token: search pattern
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's `re.match` or `re.search`).
    :param inverse: Invert the matching results.
    :param glue: If not None, this must be a string which is used to combine all tokens per match to a single string
    :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                              matched keyword.
    :return: Data frame with indices "doc" (document label) and "context" (context ID per document) and column
             "kwic" containing strings with highlighted keywords in context.
    """

    kwic_raw = kwic(docs, search_token,
                    context_size=context_size,
                    match_type=match_type,
                    ignore_case=ignore_case,
                    glob_method=glob_method,
                    inverse=inverse,
                    with_metadata=False,
                    as_data_table=False,
                    non_empty=True,
                    glue=glue,
                    highlight_keyword=highlight_keyword)

    return _datatable_from_kwic_results(kwic_raw)


def _build_kwic(docs, search_token, highlight_keyword, with_metadata, with_window_indices, context_size,
                match_type, ignore_case, glob_method, inverse):
    """
    Helper function to build keywords-in-context (KWIC) results from documents `docs`.

    :param docs: list of tokenized documents, optionally as 2-tuple where each element in `docs` is a tuple
                 of (tokens list, tokens metadata dict)
    :param search_token: search pattern
    :param highlight_keyword: If not None, this must be a string which is used to indicate the start and end of the
                              matched keyword.
    :param with_metadata: add document metadata to KWIC results
    :param with_window_indices: add window indices to KWIC results
    :param context_size: either scalar int or tuple (left, right) -- number of surrounding words in keyword context.
                         if scalar, then it is a symmetric surrounding, otherwise can be asymmetric.
    :param match_type: One of: 'exact', 'regex', 'glob'. If 'regex', `search_token` must be RE pattern. If `glob`,
                       `search_token` must be a "glob" pattern like "hello w*"
                       (see https://github.com/metagriffin/globre).
    :param ignore_case: If True, ignore case for matching.
    :param glob_method: If `match_type` is 'glob', use this glob method. Must be 'match' or 'search' (similar
                        behavior as Python's `re.match` or `re.search`).
    :param inverse: Invert the matching results.
    :return: list with KWIC results per document
    """
    # find matches for search criteria -> list of NumPy boolean mask arrays
    matches = [token_match(search_token, dtok[0] if isinstance(dtok, tuple) else dtok,
                           match_type=match_type,
                           ignore_case=ignore_case,
                           glob_method=glob_method) for dtok in docs]

    if inverse:
        matches = [~m for m in matches]

    left, right = context_size

    kwic_list = []
    for mask, dtok in zip(matches, docs):
        if isinstance(dtok, tuple):
            dtok, dmeta = dtok
        else:
            dmeta = None

        dtok_arr = np.array(dtok, dtype=str)

        ind = np.where(mask)[0]
        ind_windows = make_index_window_around_matches(mask, left, right, flatten=False)

        assert len(ind) == len(ind_windows)
        windows_in_doc = []
        for match_ind, win in zip(ind, ind_windows):  # win is an array of indices into dtok_arr
            tok_win = dtok_arr[win]

            if highlight_keyword is not None:
                highlight_mask = win == match_ind
                assert np.sum(highlight_mask) == 1
                new_tok = highlight_keyword + tok_win[highlight_mask][0] + highlight_keyword
                if len(new_tok) > np.char.str_len(tok_win[highlight_mask]).max():  # may need to create more space
                    tok_win = tok_win.astype('<U' + str(len(new_tok)))             # for this token
                tok_win[highlight_mask] = new_tok

            win_res = {'token': tok_win.tolist()}

            if with_window_indices:
                win_res['index'] = win

            if with_metadata and dmeta is not None:
                for meta_key, meta_vals in dmeta.items():
                    win_res[meta_key] = np.array(meta_vals)[win].tolist()

            windows_in_doc.append(win_res)

        kwic_list.append(windows_in_doc)

    return kwic_list


def _finalize_kwic_results(kwic_results, non_empty, glue, as_data_table, with_metadata):
    """
    Helper function to finalize raw KWIC results coming from `_build_kwic()`: Filter results, "glue" (join) tokens,
    transform to datatable, return or dismiss metadata.
    """
    if non_empty:
        if isinstance(kwic_results, dict):
            kwic_results = {dl: windows for dl, windows in kwic_results.items() if len(windows) > 0}
        else:
            assert isinstance(kwic_results, list)
            kwic_results = [windows for windows in kwic_results if len(windows) > 0]

    if glue is not None:
        if isinstance(kwic_results, dict):
            return {dl: [glue.join(win['token']) for win in windows] for dl, windows in kwic_results.items()}
        else:
            assert isinstance(kwic_results, list)
            return [[glue.join(win['token']) for win in windows] for windows in kwic_results]
    elif as_data_table:
        dfs = []
        for i_doc, dl_or_win in enumerate(kwic_results):
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
                dfs.append(dt.Frame(OrderedDict(zip(df_cols, df_windata))))

        if dfs:
            kwic_df = dt.rbind(*dfs)
            return kwic_df[:, :, dt.sort('doc', 'context', 'position')]
        else:
            return dt.Frame(OrderedDict(zip(['doc', 'context', 'position', 'token'], [[] for _ in range(4)])))
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

        dfs.append(dt.Frame(OrderedDict(zip(['doc', 'context', 'kwic'],
                                            [np.repeat(dl, len(windows)), np.arange(len(windows)), windows]))))
    if dfs:
        kwic_df = dt.rbind(*dfs)
        return kwic_df[:, :, dt.sort('doc', 'context')]
    else:
        return dt.Frame(OrderedDict(zip(['doc', 'context', 'kwic'], [[] for _ in range(3)])))


def _ngrams_from_tokens(tokens, n, join=True, join_str=' '):
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