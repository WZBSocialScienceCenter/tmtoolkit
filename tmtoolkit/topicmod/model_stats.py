"""
Statistics for topic models and BoW matrices (doc-term-matrices).

Markus Konrad <markus.konrad@wzb.eu>
"""

import numpy as np

from tmtoolkit.topicmod._common import DEFAULT_RANK_NAME_FMT
from tmtoolkit.utils import token_match
from tmtoolkit.bow.bow_stats import doc_lengths, doc_frequencies, codoc_frequencies,\
    get_term_proportions, term_frequencies


#%% Common statistics from topic-word or document-topic distribution


def get_marginal_topic_distrib(doc_topic_distrib, doc_lengths):
    """
    Return marginal topic distribution p(T) (topic proportions) given the document-topic distribution (theta)
    `doc_topic_distrib` and the document lengths `doc_lengths`. The latter can be calculated with `get_doc_lengths()`.
    """
    unnorm = (doc_topic_distrib.T * doc_lengths).sum(axis=1)
    return unnorm / unnorm.sum()


def get_marginal_word_distrib(topic_word_distrib, p_t):
    """
    Return the marginal word distribution p(w) (term proportions derived from topic model) given the
    topic-word distribution (phi) `topic_word_distrib` and the marginal topic distribution p(T) `p_t`.
    The latter can be calculated with `get_marginal_topic_distrib()`.
    """
    return (topic_word_distrib.T * p_t).sum(axis=1)


#%% General scores functions


def _words_by_score(words, score, least_to_most, n=None):
    """
    Order a vector of `words` by a `score`, either `least_to_most` or reverse. Optionally return only the top `n`
    results.
    """
    if words.shape != score.shape:
        raise ValueError('`words` and `score` must have the same shape')

    if n is not None and (n <= 0 or n > len(words)):
        raise ValueError('`n` must be in range [0, len(words)]')

    indices = np.argsort(score)
    if not least_to_most:
        indices = indices[::-1]

    ordered_words = words[indices]

    if n is not None:
        return ordered_words[:n]
    else:
        return ordered_words


#%% Saliency (Chuang et al. 2012)


def get_word_saliency(topic_word_distrib, doc_topic_distrib, doc_lengths):
    """
    Calculate word saliency according to Chuang et al. 2012.
    saliency(w) = p(w) * distinctiveness(w)

    J. Chuang, C. Manning, J. Heer 2012: "Termite: Visualization Techniques for Assessing Textual Topic Models"
    """
    p_t = get_marginal_topic_distrib(doc_topic_distrib, doc_lengths)
    p_w = get_marginal_word_distrib(topic_word_distrib, p_t)

    return p_w * get_word_distinctiveness(topic_word_distrib, p_t)


def _words_by_salience_score(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n=None, least_to_most=False):
    """Return words in `vocab` ordered by saliency score."""
    saliency = get_word_saliency(topic_word_distrib, doc_topic_distrib, doc_lengths)
    return _words_by_score(vocab, saliency, least_to_most=least_to_most, n=n)


def get_most_salient_words(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n=None):
    """
    Order the words from `vocab` by "saliency score" (Chuang et al. 2012) from most to least salient. Optionally only
    return the `n` most salient words.

    J. Chuang, C. Manning, J. Heer 2012: "Termite: Visualization Techniques for Assessing Textual Topic Models"
    """
    return _words_by_salience_score(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n)


def get_least_salient_words(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n=None):
    """
    Order the words from `vocab` by "saliency score" (Chuang et al. 2012) from least to most salient. Optionally only
    return the `n` least salient words.

    J. Chuang, C. Manning, J. Heer 2012: "Termite: Visualization Techniques for Assessing Textual Topic Models"
    """
    return _words_by_salience_score(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n, least_to_most=True)


#%% Distinctiveness (Chuang et al. 2012)


def get_word_distinctiveness(topic_word_distrib, p_t):
    """
    Calculate word distinctiveness according to Chuang et al. 2012.
    distinctiveness(w) = KL(P(T|w), P(T)) = sum_T(P(T|w) log(P(T|w)/P(T)))
    with P(T) .. marginal topic distribution
         P(T|w) .. prob. of a topic given a word

    J. Chuang, C. Manning, J. Heer 2012: "Termite: Visualization Techniques for Assessing Textual Topic Models"
    """
    topic_given_w = topic_word_distrib / topic_word_distrib.sum(axis=0)
    return (topic_given_w * np.log(topic_given_w.T / p_t).T).sum(axis=0)


def _words_by_distinctiveness_score(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n=None,
                                    least_to_most=False):
    """Return words in `vocab` ordered by distinctiveness score."""
    p_t = get_marginal_topic_distrib(doc_topic_distrib, doc_lengths)
    distinct = get_word_distinctiveness(topic_word_distrib, p_t)

    return _words_by_score(vocab, distinct, least_to_most=least_to_most, n=n)


def get_most_distinct_words(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n=None):
    """
    Order the words from `vocab` by "distinctiveness score" (Chuang et al. 2012) from most to least distinctive.
    Optionally only return the `n` most distinctive words.

    J. Chuang, C. Manning, J. Heer 2012: "Termite: Visualization Techniques for Assessing Textual Topic Models"
    """
    return _words_by_distinctiveness_score(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n)


def get_least_distinct_words(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n=None):
    """
    Order the words from `vocab` by "distinctiveness score" (Chuang et al. 2012) from least to most distinctive.
    Optionally only return the `n` least distinctive words.

    J. Chuang, C. Manning, J. Heer 2012: "Termite: Visualization Techniques for Assessing Textual Topic Models"
    """
    return _words_by_distinctiveness_score(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n,
                                           least_to_most=True)


#%% Relevance (Sievert and Shirley 2014)


def get_topic_word_relevance(topic_word_distrib, doc_topic_distrib, doc_lengths, lambda_):
    """
    Calculate the topic-word relevance score with a lambda parameter `lambda_` according to Sievert and Shirley 2014.
    relevance(w,T|lambda) = lambda * log phi_{w,t} + (1-lambda) * log (phi_{w,t} / p(w))
    with phi  .. topic-word distribution
         p(w) .. marginal word probability
    """
    p_t = get_marginal_topic_distrib(doc_topic_distrib, doc_lengths)
    p_w = get_marginal_word_distrib(topic_word_distrib, p_t)

    logtw = np.log(topic_word_distrib)
    loglift = np.log(topic_word_distrib / p_w)

    return lambda_ * logtw + (1-lambda_) * loglift


def _check_relevant_words_for_topic_args(vocab, rel_mat, topic):
    if rel_mat.ndim != 2:
        raise ValueError('`rel_mat` must be a 2D array or matrix')

    if len(vocab) != rel_mat.shape[1]:
        raise ValueError('the length of the `vocab` array must match the number of columns in `rel_mat`')

    if not 0 <= topic < rel_mat.shape[0]:
        raise ValueError('`topic` must be a topic index in range [0,%d)' % rel_mat.shape[0])


def get_most_relevant_words_for_topic(vocab, rel_mat, topic, n=None):
    """
    Get words from `vocab` for `topic` ordered by most to least relevance (Sievert and Shirley 2014) using the relevance
    matrix `rel_mat` obtained from `get_topic_word_relevance()`.
    Optionally only return the `n` most relevant words.
    """
    _check_relevant_words_for_topic_args(vocab, rel_mat, topic)
    return _words_by_score(vocab, rel_mat[topic], least_to_most=False, n=n)


def get_least_relevant_words_for_topic(vocab, rel_mat, topic, n=None):
    """
    Get words from `vocab` for `topic` ordered by least to most relevance (Sievert and Shirley 2014) using the relevance
    matrix `rel_mat` obtained from `get_topic_word_relevance()`.
    Optionally only return the `n` least relevant words.
    """
    _check_relevant_words_for_topic_args(vocab, rel_mat, topic)
    return _words_by_score(vocab, rel_mat[topic], least_to_most=True, n=n)


#%% Top words / topics


def generate_topic_labels_from_top_words(topic_word_distrib, doc_topic_distrib, doc_lengths, vocab,
                                         n_words=None, lambda_=1, labels_glue='_', labels_format='{i1}_{topwords}'):
    """
    Generate topic labels derived from the top words of each topic. The top words are determined from the
    relevance score (Sievert and Shirley 2014) depending on `lambda_`. Specify the number of top words in the label
    with `n_words`. If `n_words` is None, a minimum number of words will be used to create unique labels for each
    topic. Topic labels are formed by joining the top words with `labels_glue` and formatting them with
    `labels_format`. Placeholders in `labels_format` are `{i0}` (zero-based topic index),
    `{i1}` (one-based topic index) and `{topwords}` (top words glued with `labels_glue`).
    """
    rel_mat = get_topic_word_relevance(topic_word_distrib, doc_topic_distrib, doc_lengths, lambda_=lambda_)

    if n_words is None:
        n_words = range(1, len(vocab)+1)
    else:
        if not 1 <= n_words <= len(vocab):
            raise ValueError('`n_words` must be in range [1, %d]' % len(vocab))

        n_words = range(n_words, n_words+1)

    most_rel_words = [tuple(get_most_relevant_words_for_topic(vocab, rel_mat, t))
                      for t in range(topic_word_distrib.shape[0])]

    n_most_rel = []
    for n in n_words:
        n_most_rel = [ws[:n] for ws in most_rel_words]
        if len(n_most_rel) == len(set(n_most_rel)):   # we have a list of unique word sequences
            break

    assert n_most_rel

    topic_labels = [labels_format.format(i0=i, i1=i+1, topwords=labels_glue.join(ws))
                    for i, ws in enumerate(n_most_rel)]

    if len(topic_labels) != len(set(topic_labels)):
        raise ValueError('generated labels are not unique')

    return topic_labels


def top_n_from_distribution(distrib, top_n=10, row_labels=None, col_labels=None, val_labels=None):
    """
    Get `top_n` values from LDA model's distribution `distrib` as DataFrame. Can be used for topic-word distributions
    and document-topic distributions. Set `row_labels` to a format string or a list. Set `col_labels` to a format
    string for the column names. Set `val_labels` to return value labels instead of pure values (probabilities).
    """
    import pandas as pd

    if len(distrib) == 0:
        raise ValueError('`distrib` must contain values')

    if top_n < 1:
        raise ValueError('`top_n` must be at least 1')
    elif top_n > distrib.shape[1]:
        raise ValueError('`top_n` cannot be larger than num. of values in `distrib` rows')

    if row_labels is None:
        row_label_fixed = None
    elif isinstance(row_labels, str):
        row_label_fixed = row_labels
    else:
        row_label_fixed = None

    if val_labels is not None and type(val_labels) in (list, tuple):
        val_labels = np.array(val_labels)

    if col_labels is None:
        columns = range(top_n)
    else:
        columns = [col_labels.format(i0=i, i1=i+1) for i in range(top_n)]

    series = []

    for i, row_distrib in enumerate(distrib):
        if row_label_fixed:
            row_name = row_label_fixed.format(i0=i, i1=i+1)
        else:
            if row_labels is not None:
                row_name = row_labels[i]
            else:
                row_name = None

        # `sorter_arr` is an array of indices that would sort another array by `row_distrib` (from low to high!)
        sorter_arr = np.argsort(row_distrib)

        if val_labels is None:
            sorted_vals = row_distrib[sorter_arr][:-(top_n + 1):-1]
        else:
            if isinstance(val_labels, str):
                sorted_vals = [val_labels.format(i0=i, i1=i+1, val=row_distrib[i]) for i in sorter_arr[::-1]][:top_n]
            else:
                # first brackets: sort vocab by `sorter_arr`
                # second brackets: slice operation that reverts ordering (:-1) and then selects only `n_top` number of
                # elements
                sorted_vals = val_labels[sorter_arr][:-(top_n + 1):-1]

        series_kwargs = dict(index=columns)
        if row_name is not None:
            series_kwargs['name'] = row_name

        series.append(pd.Series(sorted_vals, **series_kwargs))

    return pd.DataFrame(series)


def top_words_for_topics(topic_word_distrib, top_n=None, vocab=None, return_prob=False):
    if not isinstance(topic_word_distrib, np.ndarray) or topic_word_distrib.ndim != 2:
        raise ValueError('`topic_word_distrib` must be a 2D NumPy array')

    if len(topic_word_distrib) == 0:
        raise ValueError('`topic_word_distrib` cannot be empty')

    if vocab is not None:
        if not isinstance(vocab, np.ndarray) or vocab.ndim != 1:
            raise ValueError('`vocab` must be a 1D NumPy array')

        if len(vocab) == 0:
            raise ValueError('`vocab` cannot be empty')

        if topic_word_distrib.shape[1] != len(vocab):
            raise ValueError('shapes of provided `topic_word_distrib` and `vocab` do not match (vocab sizes differ)')

    n_vocab = topic_word_distrib.shape[1]

    if top_n is None:
        top_n = n_vocab

    if top_n < 1:
        raise ValueError('`top_n` must be at least 1')
    elif top_n > n_vocab:
        raise ValueError('`top_n` cannot be larger than vocab size')

    topic_words = []
    topic_probs = []

    for topic in topic_word_distrib:
        sorter_arr = np.argsort(topic)
        sorter_slice = slice(None, -(top_n+1), -1) if top_n < n_vocab else slice(None)

        if vocab is None:
            topic_words.append(sorter_arr[sorter_slice])
        else:
            topic_words.append(vocab[sorter_arr][sorter_slice])

        if return_prob:
            topic_probs.append(topic[sorter_arr[sorter_slice]])

    if return_prob:
        return topic_words, topic_probs
    else:
        return topic_words


def _join_value_and_label_dfs(vals, labels, top_n, val_fmt=None, row_labels=None, col_labels=None, index_name=None):
    import pandas as pd

    val_fmt = val_fmt or '{lbl} ({val:.4})'
    col_labels = col_labels or DEFAULT_RANK_NAME_FMT
    index_name = index_name or 'document'

    if col_labels is None:
        columns = range(top_n)
    else:
        columns = [col_labels.format(i0=i, i1=i+1) for i in range(top_n)]

    df = pd.DataFrame(columns=columns)

    for i, (_, row) in enumerate(labels.iterrows()):
        joined = []
        for j, lbl in enumerate(row):
            val = vals.iloc[i, j]
            joined.append(val_fmt.format(lbl=lbl, val=val))

        if row_labels is not None:
            if isinstance(row_labels, str):
                row_name = row_labels.format(i0=i, i1=i+1)
            else:
                row_name = row_labels[i]
        else:
            row_name = None

        row_data = pd.Series(joined, name=row_name, index=columns)
        df = df.append(row_data)

    df.index.name = index_name

    return df


def filter_topics(w, vocab, topic_word_distrib, top_n=None, thresh=None, match='exact', cond='any', glob_method='match',
                  return_words_and_matches=False):
    """
    Filter topics defined as topic-word distribution `topic_word_distrib` across vocabulary `vocab` for a word (pass a
    string) or multiple words/patterns `w` (pass a list of strings). Either run pattern(s) `w` against the list of
    top words per topic (use `top_n` for number of words in top words list) or specify a minimum topic-word probability
    `thresh`, resulting in a list of words above this threshold for each topic, which will be used for pattern matching.
    You can also specify `top_n` *and* `thresh`.
    Set the `match` parameter according to the options provided by `filter_tokens.token_match()` (exact matching, RE or
    glob matching). Use `cond` to specify whether at only *one* match suffices per topic when a list of patterns `w` is
    passed (`cond='any'`) or *all* patterns must match (`cond='all'`).
    By default, this function returns a NumPy array containing the *indices* of topics that passed the filter criteria.
    If `return_words_and_matches` is True, this function additonally returns a NumPy array with the top words for each
    topic and a NumPy array with the pattern matches for each topic.
    """
    if not w:
        raise ValueError('`w` must be non empty')

    if isinstance(w, str):
        w = [w]
    elif not isinstance(w, (list, tuple, set)):
        raise ValueError('`w` must be either string or list, tuple or set')

    if top_n is None and thresh is None:
        raise ValueError('either `top_n` or `thresh` must be given')

    if cond not in {'any', 'all'}:
        raise ValueError("`cond` must be one of `'any', 'all'`")

    if thresh is None:
        top_words = top_words_for_topics(topic_word_distrib, top_n=top_n, vocab=vocab)
        top_probs = None
    else:
        top_words, top_probs = top_words_for_topics(topic_word_distrib, top_n=top_n, vocab=vocab, return_prob=True)

    found_topic_indices = []
    found_topic_words = []
    found_topic_matches = []
    cond_fn = np.any if cond == 'any' else np.all

    for t_idx, words in enumerate(top_words):
        token_matches = [token_match(x, words, match, glob_method=glob_method) for x in w]
        if top_probs:
            words_p = top_probs[t_idx]
            probs_matches = [sum(words_p[m] >= thresh) > 0 for m in token_matches]
        else:
            probs_matches = [[True]]

        token_matches_comb = np.any(token_matches, axis=1)
        assert len(token_matches_comb) == len(w)

        if cond_fn(token_matches_comb) and cond_fn(probs_matches):
            found_topic_indices.append(t_idx)
            if return_words_and_matches:
                found_topic_words.append(words)
                found_topic_matches.append(np.any(token_matches, axis=0))

    if return_words_and_matches:
        return np.array(found_topic_indices), np.array(found_topic_words), np.array(found_topic_matches)
    else:
        return np.array(found_topic_indices)


def exclude_topics(excl_topic_indices, doc_topic_distrib, topic_word_distrib=None, renormalize=True,
                   return_new_topic_mapping=False):
    """
    Exclude topics with the indices `excl_topic_indices` from the document-topic distribution `doc_topic_distrib` (i.e.
    delete the respective columns in this matrix) and optionally re-normalize the distribution so that the rows sum up
    to 1 if `renormalize` is set to `True`.
    Optionally also strip the topics from the topic-word distribution `topic_word_distrib` (i.e. remove the respective
    rows).

    If `topic_word_distrib` is given, return a tuple with the updated doc.-topic and topic-word distributions, else
    return only the updated doc.-topic distribution.

    *WARNING:* The topics to be excluded are specified by *zero-based indices*.
    """
    new_theta = np.delete(doc_topic_distrib, excl_topic_indices, axis=1)
    if renormalize:
        new_theta /= new_theta.sum(axis=1)[:, None]

    if topic_word_distrib is not None:
        new_phi = np.delete(topic_word_distrib, excl_topic_indices, axis=0)
        res_tuple = (new_theta, new_phi)
    else:
        res_tuple = (new_theta, )

    if return_new_topic_mapping:
        topic_ind = np.arange(doc_topic_distrib.shape[1])
        old_topic_ind = np.delete(topic_ind, excl_topic_indices)
        res_tuple += (dict(zip(old_topic_ind, range(len(old_topic_ind)))), )

    if len(res_tuple) == 1:
        return res_tuple[0]
    else:
        return res_tuple
