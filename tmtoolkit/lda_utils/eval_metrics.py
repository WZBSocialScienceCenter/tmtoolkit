# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import entropy


def metric_cao_juan_2009(topic_word_distrib):
    cos_dists = 1 - pdist(topic_word_distrib, metric='cosine')
    return np.mean(cos_dists)


def metric_arun_2010(topic_word_distrib, doc_topic_distrib, doc_lengths):
    # This is a somewhat sketchy metric.
    # I wouldn't use it.
    # Note: It will fail when num. of words in the vocabulary is less then the num. of topics.

    # CM1 = SVD(M1)
    cm1 = np.linalg.svd(topic_word_distrib, compute_uv=False)
    cm1 /= np.sum(cm1)  # normalize by L1 norm
    # CM2 = norm(L * M2)
    if doc_lengths.shape[0] != 1:
        doc_lengths = doc_lengths.T
    cm2 = np.array(doc_lengths * np.matrix(doc_topic_distrib))[0]
    #cm2 /= np.linalg.norm(cm2, 2)  # normalize by L2 norm
    cm2 /= np.sum(cm2)          # normalize by L1 norm

    # symmetric Kullback-Leibler divergence KL(cm1||cm2) + KL(cm2||cm1)
    # KL is called entropy in scipy
    return entropy(cm1, cm2) + entropy(cm2, cm1)
    #return np.sum(cm1*np.log(cm1/cm2)) + np.sum(cm2*np.log(cm2/cm1))


def metric_griffiths_2004(logliks):
    """
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
