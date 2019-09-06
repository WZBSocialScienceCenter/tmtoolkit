"""
Metrics for topic model evaluation.

In order to run model evaluations in parallel use one of the modules :mod:`~tmtoolkit.topicmod.tm_gensim`,
:mod:`~tmtoolkit.topicmod.tm_lda` or :mod:`~tmtoolkit.topicmod.tm_sklearn`.
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.sparse import issparse
from scipy.special import gammaln

from ._eval_tools import FakedGensimDict
from tmtoolkit.bow.dtm import dtm_and_vocab_to_gensim_corpus_and_dict
from .model_stats import top_words_for_topics
from tmtoolkit.bow.bow_stats import doc_frequencies, codoc_frequencies
from ..utils import argsort


#%% Evaluation metrics


def metric_held_out_documents_wallach09(dtm_test, theta_test, phi_train, alpha, n_samples=10000):
    """
    Estimation of the probability of held-out documents according to [Wallach2009]_ using a
    document-topic estimation `theta_test` that was estimated via held-out documents `dtm_test` on a trained model with
    a topic-word distribution `phi_train` and a document-topic prior `alpha`. Draw `n_samples` according to `theta_test`
    for each document in `dtm_test` (memory consumption and run time can be very high for larger `n_samples` and
    a large amount of big documents in `dtm_test`).

    A document-topic estimation `theta_test` can be obtained from a trained model from the "lda" package or scikit-learn
    package with the `transform()` method.

    Adopted MATLAB code `originally from Ian Murray, 2009 <https://people.cs.umass.edu/~wallach/code/etm/>`_ and
    downloaded from `umass.edu <https://people.cs.umass.edu/~wallach/code/etm/lda_eval_matlab_code_20120930.tar.gz>`_.

    .. note:: Requires `gmpy2 <https://github.com/aleaxit/gmpy>`_ package for multiple-precision arithmetic to avoid
              numerical underflow.

    .. [Wallach2009] Wallach, H.M., Murray, I., Salakhutdinov, R. and Mimno, D., 2009. Evaluation methods for
                     topic models.

    :param dtm_test: held-out documents of shape NxM with N documents and vocabulary size M
    :param theta_test: document-topic estimation of `dtm_test`; shape NxK with K topics
    :param phi_train: topic-word distribution of a trained topic model that should be evaluated; shape KxM
    :param alpha: document-topic prior of the trained topic model that should be evaluated; either a scalar or an array
                  of length K
    :return: estimated probability of held-out documents
    """
    import gmpy2

    n_test_docs, n_vocab = dtm_test.shape

    if n_test_docs != theta_test.shape[0]:
        raise ValueError('shapes of `dtm_test` and `theta_test` do not match (unequal number of documents)')

    _, n_topics = theta_test.shape

    if n_topics != phi_train.shape[0]:
        raise ValueError('shapes of `theta_test` and `phi_train` do not match (unequal number of topics)')

    if n_vocab != phi_train.shape[1]:
        raise ValueError('shapes of `dtm_test` and `phi_train` do not match (unequal size of vocabulary)')

    if isinstance(alpha, np.ndarray):
        alpha_sum = np.sum(alpha)
    else:
        alpha_sum = alpha * n_topics
        alpha = np.repeat(alpha, n_topics)

    if alpha.shape != (n_topics, ):
        raise ValueError('`alpha` has invalid shape (should be vector of length n_topics)')

    # samples: random topic assignments for each document
    #          shape: n_test_docs x n_samples
    #          values in [0, n_topics) ~ theta_test
    samples = np.array([np.random.choice(n_topics, n_samples, p=theta_test[d, :])
                        for d in range(n_test_docs)])
    assert samples.shape == (n_test_docs, n_samples)
    assert 0 <= samples.min() < n_topics
    assert 0 <= samples.max() < n_topics

    # n_k: number of documents per topic and sample
    #      shape: n_topics x n_samples
    #      values in [0, n_test_docs]
    n_k = np.array([np.sum(samples == t, axis=0) for t in range(n_topics)])
    assert n_k.shape == (n_topics, n_samples)
    assert 0 <= n_k.min() <= n_test_docs
    assert 0 <= n_k.max() <= n_test_docs

    # calculate log p(z) for each sample
    # shape: 1 x n_samples
    log_p_z = np.sum(gammaln(n_k + alpha[:, np.newaxis]), axis=0) + gammaln(alpha_sum) \
              - np.sum(gammaln(alpha)) - gammaln(n_test_docs + alpha_sum)

    assert log_p_z.shape == (n_samples,)

    # calculate log p(w|z) for each sample
    # shape: 1 x n_samples

    log_p_w_given_z = np.zeros(n_samples)

    dtm_is_sparse = issparse(dtm_test)
    for d in range(n_test_docs):
        if dtm_is_sparse:
            word_counts_d = dtm_test[d].toarray().flatten()
        else:
            word_counts_d = dtm_test[d]
        words = np.repeat(np.arange(n_vocab), word_counts_d)
        assert words.shape == (word_counts_d.sum(),)

        phi_topics_d = phi_train[samples[d]]   # phi for topics in samples for document d
        log_p_w_given_z += np.sum(np.log(phi_topics_d[:, words]), axis=1)

    log_joint = log_p_z + log_p_w_given_z

    # calculate log theta_test
    # shape: 1 x n_samples

    log_theta_test = np.zeros(n_samples)

    for d in range(n_test_docs):
        log_theta_test += np.log(theta_test[d, samples[d]])

    # compare
    log_weights = log_joint - log_theta_test

    # calculate final log evidence
    # requires using gmpy2 to avoid numerical underflow
    exp_sum = gmpy2.mpfr(0)
    for exp in (gmpy2.exp(x) for x in log_weights):
        exp_sum += exp

    return float(gmpy2.log(exp_sum)) - np.log(n_samples)
metric_held_out_documents_wallach09.direction = 'maximize'


def metric_cao_juan_2009(topic_word_distrib):
    """
    Calculate metric as in [Cao2008]_ using topic-word distribution `topic_word_distrib`.

    .. [Cao2008] Cao Juan, Xia Tian, Li Jintao, Zhang Yongdong, and Tang Sheng. 2009. A density-based method for
                 adaptive LDA model selection. Neurocomputing — 16th European Symposium on Artificial Neural Networks
                 2008 72, 7–9: 1775–1781. <http://doi.org/10.1016/j.neucom.2008.06.011>.

    :param topic_word_distrib: topic-word distribtion; shape KxM, where K is number of topics, M is vocabulary size
    :return: calculated metric
    """
    # pdist will calculate the pair-wise cosine distance between all topics in the topic-word distribution
    # then calculate the mean of cosine similarity (1 - cosine_distance)
    cos_sim = 1 - pdist(topic_word_distrib, metric='cosine')
    return np.mean(cos_sim)
metric_cao_juan_2009.direction = 'minimize'


def metric_arun_2010(topic_word_distrib, doc_topic_distrib, doc_lengths):
    """
    Calculate metric as in [Arun2010]_ using topic-word distribution `topic_word_distrib`, document-topic
    distribution `doc_topic_distrib` and document lengths `doc_lengths`.

    .. note:: It will fail when num. of words in the vocabulary is less then the num. of topics (which is very unusual).

    .. [Arun2010] Rajkumar Arun, V. Suresh, C. E. Veni Madhavan, and M. N. Narasimha Murthy. 2010. On finding the natural
                  number of topics with latent dirichlet allocation: Some observations. In Advances in knowledge discovery and
                  data mining, Mohammed J. Zaki, Jeffrey Xu Yu, Balaraman Ravindran and Vikram Pudi (eds.). Springer Berlin
                  Heidelberg, 391–402. http://doi.org/10.1007/978-3-642-13657-3_43.

    :param topic_word_distrib: topic-word distribtion; shape KxM, where K is number of topics, M is vocabulary size
    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents
    :param doc_lengths: array of length `N` with number of tokens per document
    :return: calculated metric
    """

    # CM1 = SVD(M1)
    cm1 = np.linalg.svd(topic_word_distrib, compute_uv=False)
    #cm1 /= np.sum(cm1)  # normalize by L1 norm # the paper says nothing about normalizing so let's leave it as it is...

    # CM2 = L*M2 / norm2(L)
    if doc_lengths.shape[0] != 1:
        doc_lengths = doc_lengths.T
    cm2 = np.array(doc_lengths * np.matrix(doc_topic_distrib))[0]
    cm2 /= np.linalg.norm(doc_lengths, 2)
    # wrong:
    #cm2 /= np.linalg.norm(cm2, 2)  # normalize by L2 norm
    # also wrong:
    #cm2 /= np.sum(cm2)          # normalize by L1 norm

    # symmetric Kullback-Leibler divergence KL(cm1||cm2) + KL(cm2||cm1)
    # KL is called entropy in scipy
    # we can't use this because entropy() will normalize the vectors so that they sum up to 1 but this should not
    # be done according to the paper
    #return entropy(cm1, cm2) + entropy(cm2, cm1)

    # use it as in the paper (note: cm1 and cm2 are not prob. distributions that sum up to 1)
    return np.sum(cm1*np.log(cm1/cm2)) + np.sum(cm2*np.log(cm2/cm1))
metric_arun_2010.direction = 'minimize'


def metric_griffiths_2004(logliks):
    """
    Calculate metric as in [GriffithsSteyvers2004]_.

    Calculates the harmonic mean of the log-likelihood values `logliks`. Burn-in values
    should already be removed from `logliks`.

    .. [GriffithsSteyvers2004] Thomas L. Griffiths and Mark Steyvers. 2004. Finding scientific topics. Proceedings of
                               the National Academy of Sciences 101, suppl 1: 5228–5235.
                               http://doi.org/10.1073/pnas.0307752101

    .. note:: Requires `gmpy2 <https://github.com/aleaxit/gmpy>`_ package for multiple-precision arithmetic to avoid
              numerical underflow.

    :param logliks: array with log-likelihood values
    :return: calculated metric
    """

    import gmpy2

    # using median trick as in Martin Ponweiser's Diploma Thesis 2012, p.36
    ll_med = np.median(logliks)
    ps = [gmpy2.exp(ll_med - x) for x in logliks]
    ps_mean = gmpy2.mpfr(0)
    for p in ps:
        ps_mean += p / len(ps)
    return float(ll_med - gmpy2.log(ps_mean))   # after taking the log() we can use a Python float() again
metric_griffiths_2004.direction = 'maximize'


def metric_coherence_mimno_2011(topic_word_distrib, dtm, top_n=20, eps=1e-12, normalize=True, return_mean=False):
    """
    Calculate coherence metric according to [Mimno2011]_ (a.k.a. "U_Mass" coherence metric). There are two
    modifications to the originally suggested measure:

    - uses a different epsilon by default (set `eps=1` for original)
    - uses a normalizing constant by default (set `normalize=False` for original)

    Provide a topic word distribution as `topic_word_distrib` and a document-term-matrix `dtm` (can be sparse).
    `top_n` controls how many most probable words per topic are selected.

    By default, it will return a NumPy array of coherence values per topic (same ordering as in `topic_word_distrib`).
    Set `return_mean` to True to return the mean of all topics instead.

    .. [Mimno2011] D. Mimno, H. Wallach, E. Talley, M. Leenders, A. McCullum 2011: Optimizing semantic coherence in
                   topic models

    :param topic_word_distrib: topic-word distribtion; shape KxM, where K is number of topics, M is vocabulary size
    :param dtm: document-term matrix of shape NxM with N documents and vocabulary size M
    :param top_n: number of most probable words selected per topic
    :param eps: smoothing constant epsilon
    :param normalize: if True, normalize coherence values
    :param return_mean: if True, return mean of all coherence values, otherwise array of coherence per topic
    :return: if `return_mean` is True, mean of all coherence values, otherwise array of length K with coherence per
             topic
    """
    n_topics, n_vocab = topic_word_distrib.shape

    if n_vocab != dtm.shape[1]:
        raise ValueError('shapes of provided `topic_word_distrib` and `dtm` do not match (vocab sizes differ)')

    if top_n > n_vocab:
        raise ValueError('`top_n=%d` is larger than the vocabulary size of %d words'
                         % (top_n, topic_word_distrib.shape[1]))

    top_words = top_words_for_topics(topic_word_distrib, top_n)   # V

    if issparse(dtm) and dtm.format != 'csc':
        dtm = dtm.tocsc()

    coh = []
    for t in range(n_topics):
        c_t = 0

        v = top_words[t]
        top_dtm = dtm[:, v]
        df = doc_frequencies(top_dtm)      # D(v)
        codf = codoc_frequencies(top_dtm)  # D(v, v')

        for m in range(1, top_n):
            for l in range(m):
                c_t += np.log((codf[m, l] + eps) / df[l])

        coh.append(c_t)

    coh = np.array(coh)

    if normalize:
        coh *= 2 / (top_n * (top_n-1))

    if return_mean:
        return coh.mean()
    else:
        return coh
metric_coherence_mimno_2011.direction = 'maximize'


def metric_coherence_gensim(measure, topic_word_distrib=None, gensim_model=None, vocab=None, dtm=None,
                            gensim_corpus=None, texts=None, top_n=20,
                            return_coh_model=False, return_mean=False, **kwargs):
    """
    Calculate model coherence using Gensim's
    `CoherenceModel <https://radimrehurek.com/gensim/models/coherencemodel.html>`_. See also this `tutorial
    <https://rare-technologies.com/what-is-topic-coherence/>`_.

    Define which measure to use with parameter `measure`:

    - ``'u_mass'``
    - ``'c_v'``
    - ``'c_uci'``
    - ``'c_npmi'``

    Provide a topic word distribution `topic_word_distrib` OR a Gensim model `gensim_model`
    and the corpus' vocabulary as `vocab` OR pass a gensim corpus as `gensim_corpus`. `top_n` controls how many most
    probable words per topic are selected.

    If measure is ``'u_mass'``, a document-term-matrix `dtm` or `gensim_corpus` must be provided and `texts` can be
    None. If any other measure than ``'u_mass'`` is used, tokenized input as `texts` must be provided as 2D list::

        [['some', 'text', ...],          # doc. 1
         ['some', 'more', ...],          # doc. 2
         ['another', 'document', ...]]   # doc. 3

    If `return_coh_model` is True, the whole :class:`gensim.models.CoherenceModel` instance will be returned, otherwise:

    - if `return_mean` is True, the mean coherence value will be returned
    - if `return_mean` is False, a list of coherence values (for each topic) will be returned

    Provided `kwargs` will be passed to :class:`gensim.models.CoherenceModel` or
    :meth:`gensim.models.CoherenceModel.get_coherence_per_topic`.

    .. note:: This function also supports models from `lda` and `sklearn` (by passing `topic_word_distrib`, `dtm` and
              `vocab`)!

    :param measure: the coherence calculation type; one of the values listed above
    :param topic_word_distrib: topic-word distribtion; shape KxM, where K is number of topics, M is vocabulary size if
                               `gensim_model` is not given
    :param gensim_model: a topic model from Gensim if `topic_word_distrib` is not given
    :param vocab: vocabulary list/array if `gensim_corpus` is not given
    :param dtm: document-term matrix of shape NxM with N documents and vocabulary size M  if `gensim_corpus` is not
                given
    :param gensim_corpus: a Gensim corpus if `vocab` is not given
    :param texts: list of tokenized documents; necessary if using a `measure` other than ``'u_mass'``
    :param top_n: number of most probable words selected per topic
    :param return_coh_model: if True, return :class:`gensim.models.CoherenceModel` as result
    :param return_mean: if `return_coh_model` is False and `return_mean` is True, return mean coherence
    :param kwargs: parameters passed to :class:`gensim.models.CoherenceModel` or
                   :meth:`gensim.models.CoherenceModel.get_coherence_per_topic`
    :return: if `return_coh_model` is True, return :class:`gensim.models.CoherenceModel` as result; otherwise if
             `return_mean` is True, mean of all coherence values, otherwise array of length K with coherence per
             topic
    """
    try:
        import gensim
    except ImportError:
        raise ValueError('package `gensim` must be installed for `coherence_gensim` metric')

    if measure == 'u_mass' and dtm is None and gensim_corpus is None:
        raise ValueError('document-term-matrix `dtm` or Gensim corpus `gensim_corpus` must be provided for measure '
                         '`u_mass`')
    elif measure != 'u_mass' and texts is None:
        raise ValueError('`texts` must be provided for any other measure than `u_mass`')

    if gensim_model is None:
        if topic_word_distrib is None:
            raise ValueError('`topic_word_distrib` must be given if `gensim_model` was not passed')
        n_topics, n_vocab = topic_word_distrib.shape
    else:
        n_topics, n_vocab = None, None

    if vocab is not None:
        if len(vocab) != n_vocab:
            raise ValueError('shape of provided `topic_word_distrib` and length of `vocab` do not match '
                             '(vocab sizes differ)')
        if top_n > n_vocab:
            raise ValueError('`top_n=%d` is larger than the vocabulary size of %d words'
                             % (top_n, topic_word_distrib.shape[1]))
    elif gensim_corpus is None:
        raise ValueError('a gensim corpus `gensim_corpus` must be passed if no `vocab` is given')

    if measure == 'u_mass' and gensim_corpus is None and n_vocab != dtm.shape[1]:
        raise ValueError('shapes of provided `topic_word_distrib` and `dtm` do not match (vocab sizes differ)')

    if vocab is not None:
        top_words = top_words_for_topics(topic_word_distrib, top_n, vocab=vocab)   # V
    else:
        top_words = None

    coh_model_kwargs = {'coherence': measure}
    if measure == 'u_mass':
        if gensim_corpus is None:
            gensim_corpus, gensim_dict = dtm_and_vocab_to_gensim_corpus_and_dict(dtm, vocab)
            coh_model_kwargs.update(dict(corpus=gensim_corpus, dictionary=gensim_dict, topics=top_words))
        else:
            coh_model_kwargs.update(dict(model=gensim_model, corpus=gensim_corpus, topn=top_n))
    else:
        if gensim_corpus is None:
            coh_model_kwargs.update(dict(texts=texts, topics=top_words, dictionary=FakedGensimDict.from_vocab(vocab)))
        else:
            coh_model_kwargs.update(dict(texts=texts, model=gensim_model, corpus=gensim_corpus, topn=top_n))

    get_coh_kwargs = {}
    for opt in ('segmented_topics', 'with_std', 'with_support'):
        if opt in kwargs:
            get_coh_kwargs[opt] = kwargs.pop(opt)

    coh_model_kwargs.update(kwargs)

    coh_model = gensim.models.CoherenceModel(**coh_model_kwargs)

    if return_coh_model:
        return coh_model
    else:
        if return_mean:
            return coh_model.get_coherence()
        else:
            return coh_model.get_coherence_per_topic(**get_coh_kwargs)
metric_coherence_gensim.direction = 'maximize'


#%% Helper functions for topic model evaluation

def results_by_parameter(res, param, sort_by=None, sort_desc=False):
    """
    Takes a list of evaluation results `res` returned by a topic model evaluation function – a list in the form:

    .. code-block:: text

        [(parameter_set_1, {'<metric_name>': result_1, ...}),
         ...,
         (parameter_set_n, {'<metric_name>': result_n, ...})])

    Then returns a list with tuple pairs using only the parameter `param` from the parameter sets in the evaluation
    results such that the returned list is:

    .. code-block:: text

        [(param_1, {'<metric_name>': result_1, ...}),
         ...,
         (param_n, {'<metric_name>': result_n, ...})]

    Optionally order either by parameter value (`sort_by` is None - the default) or by result metric
    (``sort_by='<metric name>'``).

    :param res: list of evaluation results
    :param param: string of parameter name
    :param sort_by: order by parameter value if this is None, or by a certain result metric given as string
    :param sort_desc: sort in descending order
    :return: list with tuple pairs using only the parameter `param` from the parameter sets
    """
    if len(res) == 0:
        return []

    tuples = [(p[param], r) for p, r in res]

    # single validation results
    if len(tuples[0]) != 2:
        raise ValueError('invalid evaluation results passed')

    params, metric_results = list(zip(*tuples))
    if sort_by:
        sorted_ind = argsort([r[sort_by] for r in metric_results])
    else:
        sorted_ind = argsort(params)

    if sort_desc:
        sorted_ind = reversed(sorted_ind)

    measurements = tuples

    return [measurements[i] for i in sorted_ind]
