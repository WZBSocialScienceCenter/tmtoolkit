# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from scipy.spatial.distance import pdist
from scipy.sparse import issparse

from .common import top_words_for_topics, get_doc_frequencies, get_codoc_frequencies,\
    dtm_and_vocab_to_gensim_corpus_and_dict


def metric_cao_juan_2009(topic_word_distrib):
    """
    Cao Juan, Xia Tian, Li Jintao, Zhang Yongdong, and Tang Sheng. 2009. A density-based method for adaptive LDA model
    selection. Neurocomputing — 16th European Symposium on Artificial Neural Networks 2008 72, 7–9: 1775–1781.
    http://doi.org/10.1016/j.neucom.2008.06.011
    """
    # pdist will calculate the pair-wise cosine distance between all topics in the topic-word distribution
    # then calculate the mean of cosine similarity (1 - cosine_distance)
    cos_sim = 1 - pdist(topic_word_distrib, metric='cosine')
    return np.mean(cos_sim)


def metric_arun_2010(topic_word_distrib, doc_topic_distrib, doc_lengths):
    """
    Rajkumar Arun, V. Suresh, C. E. Veni Madhavan, and M. N. Narasimha Murthy. 2010. On finding the natural number of
    topics with latent dirichlet allocation: Some observations. In Advances in knowledge discovery and data mining,
    Mohammed J. Zaki, Jeffrey Xu Yu, Balaraman Ravindran and Vikram Pudi (eds.). Springer Berlin Heidelberg, 391–402.
    http://doi.org/10.1007/978-3-642-13657-3_43
    """
    # Note: It will fail when num. of words in the vocabulary is less then the num. of topics (which is very unusual).

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


def metric_griffiths_2004(logliks):
    """
    Thomas L. Griffiths and Mark Steyvers. 2004. Finding scientific topics. Proceedings of the National Academy
    of Sciences 101, suppl 1: 5228–5235. http://doi.org/10.1073/pnas.0307752101

    Calculates the harmonic mean of the loglikelihood values `logliks` as in Griffiths, Steyvers 2004. Burnin values
    should already be removed from `logliks`.

    Note: requires gmpy2 package for multiple-precision arithmetic due to very large exp() values.
          see https://github.com/aleaxit/gmpy
    """

    import gmpy2

    # using median trick as in Martin Ponweiser's Diploma Thesis 2012, p.36
    ll_med = np.median(logliks)
    ps = [gmpy2.exp(ll_med - x) for x in logliks]
    ps_mean = gmpy2.mpfr(0)
    for p in ps:
        ps_mean += p / len(ps)
    return float(ll_med - gmpy2.log(ps_mean))   # after taking the log() we can use a Python float() again


def metric_coherence_mimno_2011(topic_word_distrib, dtm, top_n=20, eps=1e-12, normalize=True, return_mean=False):
    """
    Calculate coherence metric according to Mimno et al. 2011 (a.k.a. "U_Mass" coherence metric). There are two
    modifications to the originally suggested measure:
    - uses a different epsilon by default (set `eps=1` for original)
    - uses a normalizing constant by default (set `normalize=False` for original)

    Provide a topic word distribution $\phi$ as `topic_word_distrib` and a document-term-matrix `dtm` (can be sparse).
    `top_n` controls how many most probable words per topic are selected.

    By default, it will return a NumPy array of coherence values per topic (same ordering as in `topic_word_distrib`).
    Set `return_mean` to True to return the mean of all topics instead.

    D. Mimno, H. Wallach, E. Talley, M. Leenders, A. McCullum 2011: Optimizing semantic coherence in topic models
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
        df = get_doc_frequencies(top_dtm)      # D(v)
        codf = get_codoc_frequencies(top_dtm)  # D(v, v')

        for m in range(1, top_n):
            for l in range(m):
                c_t += np.log((codf.get((m, l), codf.get((l, m))) + eps) /
                              df[l])

        coh.append(c_t)

    coh = np.array(coh)

    if normalize:
        coh *= 2 / (top_n * (top_n-1))

    if return_mean:
        return coh.mean()
    else:
        return coh


def metric_coherence_gensim(measure, topic_word_distrib=None, gensim_model=None, vocab=None, dtm=None,
                            gensim_corpus=None, texts=None, top_n=20,
                            return_coh_model=False, return_mean=False, **kwargs):
    """
    Calculate model coherence using Gensim's `CoherenceModel` [1,2]. Note: this function also supports models from
    `lda` and `sklearn` (by passing `topic_word_distrib`, `dtm` and `vocab`)!

    Define which measure to use with parameter `measure`:
    - u_mass
    - c_v
    - c_uci
    - c_npmi

    Provide a topic word distribution $\phi$ as `topic_word_distrib` OR a Gensim model `gensim_model`
    and the corpus' vocabulary as `vocab` OR pass a gensim corpus as `gensim_corpus`.
    `top_n` controls how many most probable words per topic are selected.

    If measure is `u_mass`, a document-term-matrix `dtm` or `gensim_corpus` must be provided and `texts` can be None.
    If any other measure than `u_mass` is used, tokenized input as `texts` must be provided as 2D list:
    ```
    [['some', 'text', ...],          # doc. 1
     ['some', 'more', ...],          # doc. 2
     ['another', 'document', ...]]   # doc. 3
    ```

    If `return_coh_model` is True, the whole `CoherenceModel` instance will be returned, otherwise:
    - if `return_mean` is True, the mean coherence value will be returned
    - if `return_mean` is False, a list of coherence values (for each topic) will be returned

    Provided `kwargs` will be passed to `CoherenceModel()` or `CoherenceModel.get_coherence_per_topic()`.

    [1]: https://radimrehurek.com/gensim/models/coherencemodel.html
    [2]: https://rare-technologies.com/what-is-topic-coherence/
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
            coh_model_kwargs.update(dict(texts=texts, topics=top_words, dictionary=dict(zip(range(n_vocab), vocab))))
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
