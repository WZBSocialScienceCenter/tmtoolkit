import numpy as np
import pandas as pd
import pytest
from hypothesis import settings, given, strategies as st
from scipy.sparse import coo_matrix, csr_matrix, issparse

from ._testtools import strategy_dtm

from tmtoolkit import bow

try:
    import gensim
    GENSIM_INSTALLED = True
except ImportError:
    GENSIM_INSTALLED = False

pytestmark = [pytest.mark.filterwarnings("ignore:divide by zero"),   # happens due to generated data by hypothesis
              pytest.mark.filterwarnings("ignore:invalid value")]

@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_doc_lengths(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm_arr = dtm

    if dtm_arr.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.doc_lengths(dtm)
    else:
        doc_lengths = bow.bow_stats.doc_lengths(dtm)
        assert doc_lengths.ndim == 1
        assert doc_lengths.shape == (dtm_arr.shape[0],)
        assert doc_lengths.tolist() == [sum(row) for row in dtm_arr]


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_doc_frequencies(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm_arr = dtm

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.doc_frequencies(dtm)
    else:
        n_docs = dtm.shape[0]

        df_abs = bow.bow_stats.doc_frequencies(dtm)
        assert isinstance(df_abs, np.ndarray)
        assert df_abs.ndim == 1
        assert df_abs.shape == (dtm_arr.shape[1],)
        assert all([0 <= v <= n_docs for v in df_abs])

        df_rel = bow.bow_stats.doc_frequencies(dtm, proportions=1)
        assert isinstance(df_rel, np.ndarray)
        assert df_rel.ndim == 1
        assert df_rel.shape == (dtm_arr.shape[1],)
        assert all([0 <= v <= 1 for v in df_rel])

        df_log = bow.bow_stats.doc_frequencies(dtm, proportions=2)
        assert isinstance(df_log, np.ndarray)
        assert df_log.ndim == 1
        assert df_log.shape == (dtm_arr.shape[1],)
        assert np.allclose(np.exp(df_log), df_rel)


def test_doc_frequencies2():
    dtm = np.array([
        [0, 2, 3, 0, 0],
        [1, 2, 0, 5, 0],
        [0, 1, 0, 3, 1],
    ])

    df = bow.bow_stats.doc_frequencies(dtm)

    assert df.tolist() == [1, 3, 1, 2, 1]


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1),
    proportions=st.integers(min_value=0, max_value=2)
)
def test_codoc_frequencies(dtm, matrix_type, proportions):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.codoc_frequencies(dtm, proportions=proportions)
        return

    n_docs, n_vocab = dtm.shape

    if n_vocab < 2:
        with pytest.raises(ValueError):
            bow.bow_stats.codoc_frequencies(dtm, proportions=proportions)
        return

    cooc = bow.bow_stats.codoc_frequencies(dtm, proportions=proportions)

    if matrix_type == 1 and proportions != 2:
        assert issparse(cooc)
        cooc = cooc.todense()
    else:
        assert isinstance(cooc, np.ndarray)

    assert cooc.shape == (n_vocab, n_vocab)

    if proportions > 0:
        assert np.all(cooc <= 1)

        if proportions == 1:
            assert np.all(0 <= cooc)
        else:   # proportions == 2
            expected = bow.bow_stats.codoc_frequencies(dtm, proportions=1)
            if issparse(expected):
                expected = expected.todense()
            np.allclose(np.exp(cooc) - 1, expected)
    else:
        assert np.all(0 <= cooc)
        assert np.all(cooc <= n_docs)


def test_codoc_frequencies2():
    dtm = np.array([
        [0, 2, 3, 0, 0],
        [1, 2, 0, 5, 0],
        [0, 1, 0, 3, 1],
    ])

    cooc = bow.bow_stats.codoc_frequencies(dtm)

    assert cooc[0, 1] == cooc[1, 0] == 1
    assert cooc[1, 3] == cooc[3, 1] == 2
    assert cooc[0, 2] == cooc[2, 0] == 0


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_term_frequencies(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm_arr = dtm

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.term_frequencies(dtm)
    else:
        tf = bow.bow_stats.term_frequencies(dtm)
        assert tf.ndim == 1
        assert tf.shape == (dtm_arr.shape[1],)
        assert tf.tolist() == [sum(row) for row in dtm_arr.T]

        if np.sum(dtm) > 0:
            tf_prop = bow.bow_stats.term_frequencies(dtm, proportions=1)
            assert tf_prop.ndim == 1
            assert tf_prop.shape == (dtm_arr.shape[1],)
            assert np.all(tf_prop>= 0)
            assert np.all(tf_prop <= 1)
            assert np.isclose(tf_prop.sum(), 1.0)

            tf_logprop = bow.bow_stats.term_frequencies(dtm, proportions=2)
            assert tf.ndim == 1
            assert tf.shape == (dtm_arr.shape[1],)
            assert np.allclose(np.exp(tf_logprop), tf_prop)
        else:
            with pytest.raises(ValueError):
                bow.bow_stats.term_frequencies(dtm, proportions=1)
            with pytest.raises(ValueError):
                bow.bow_stats.term_frequencies(dtm, proportions=2)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_tf_binary(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm_arr = dtm

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.tf_binary(dtm)
    else:
        res = bow.bow_stats.tf_binary(dtm)
        assert res.ndim == 2
        assert res.shape == dtm.shape
        assert res.dtype.kind in {'i', 'u'}
        if matrix_type == 1:
            assert issparse(res)
            res = res.A
        else:
            assert isinstance(res, np.ndarray)

        assert set(np.unique(res)) <= {0, 1}     # subset test

        zero_ind_dtm = np.where(dtm_arr == 0)
        zero_ind_res = np.where(res == 0)
        assert len(zero_ind_dtm) == len(zero_ind_res)
        for ind_dtm, ind_res in zip(zero_ind_dtm, zero_ind_res):
            assert np.array_equal(ind_dtm, ind_res)

        notzero_ind_dtm = np.where(dtm_arr != 0)
        notzero_ind_res = np.where(res != 0)
        assert len(notzero_ind_dtm) == len(notzero_ind_res)
        for ind_dtm, ind_res in zip(notzero_ind_dtm, notzero_ind_res):
            assert np.array_equal(ind_dtm, ind_res)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_tf_proportions(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.tf_proportions(dtm)
    else:
        res = bow.bow_stats.tf_proportions(dtm)
        assert res.ndim == 2
        assert res.shape == dtm.shape
        assert res.dtype.kind == 'f'
        if matrix_type == 1:
            assert issparse(res)
            res = res.A
        else:
            assert isinstance(res, np.ndarray)

        # exclude NaNs that may be introduced when documents are of length 0
        res_flat = res.flatten()
        res_valid = res_flat[~np.isnan(res_flat)]
        assert np.all(res_valid >= -1e-10)
        assert np.all(res_valid <= 1 + 1e-10)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_tf_log(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm_arr = dtm

    if dtm.ndim != 2:
        with pytest.raises(ValueError):
            bow.bow_stats.tf_log(dtm)
    else:
        res = bow.bow_stats.tf_log(dtm)
        assert res.ndim == 2
        assert res.shape == dtm.shape
        assert res.dtype.kind == 'f'
        if matrix_type == 1:
            assert issparse(res)
            res = res.A
        else:
            assert isinstance(res, np.ndarray)

        assert np.all(res >= -1e-10)

        if 0 not in dtm.shape:
            max_res = np.log(np.max(dtm_arr) + 1)
            assert np.all(res <= max_res + 1e-10)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1),
    K=st.floats(min_value=0, max_value=1)
)
def test_tf_double_norm(dtm, matrix_type, K):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    if dtm.ndim != 2 or 0 in dtm.shape:
        with pytest.raises(ValueError):
            bow.bow_stats.tf_double_norm(dtm, K=K)
    else:
        res = bow.bow_stats.tf_double_norm(dtm, K=K)

        assert res.ndim == 2
        assert res.shape == dtm.shape
        assert res.dtype.kind == 'f'
        assert isinstance(res, np.ndarray)

        # exclude NaNs that may be introduced when documents are of length 0
        res_flat = res.flatten()
        res_valid = res_flat[~np.isnan(res_flat)]

        assert np.all(res_valid >= -1e-10)
        assert np.all(res_valid <= 1 + 1e-10)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_idf(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    if dtm.ndim != 2 or 0 in dtm.shape:
        with pytest.raises(ValueError):
            bow.bow_stats.idf(dtm)
    else:
        res = bow.bow_stats.idf(dtm)
        assert res.ndim == 1
        assert res.shape[0] == dtm.shape[1]
        assert res.dtype.kind == 'f'
        assert isinstance(res, np.ndarray)
        assert np.all(res >= -1e-10)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_idf_probabilistic(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    if dtm.ndim != 2 or 0 in dtm.shape:
        with pytest.raises(ValueError):
            bow.bow_stats.idf_probabilistic(dtm)
    else:
        res = bow.bow_stats.idf_probabilistic(dtm)
        assert res.ndim == 1
        assert res.shape[0] == dtm.shape[1]
        assert res.dtype.kind == 'f'
        assert isinstance(res, np.ndarray)
        assert np.all(res >= -1e-10)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1),
    tf_func=st.integers(min_value=0, max_value=3),
    K=st.floats(min_value=-1, max_value=1),             # negative means don't pass this parameter
    idf_func=st.integers(min_value=0, max_value=1),
    smooth=st.integers(min_value=-1, max_value=3),      # -1 means don't pass this parameter
    smooth_log=st.integers(min_value=-1, max_value=3),  # -1 means don't pass this parameter
    smooth_df=st.integers(min_value=-1, max_value=3),   # -1 means don't pass this parameter
)
def test_tfidf(dtm, matrix_type, tf_func, K, idf_func, smooth, smooth_log, smooth_df):
    tfidf_opts = {}

    tf_funcs = (
        bow.bow_stats.tf_binary,
        bow.bow_stats.tf_proportions,
        bow.bow_stats.tf_log,
        bow.bow_stats.tf_double_norm
    )
    tfidf_opts['tf_func'] = tf_funcs[tf_func]

    if tfidf_opts['tf_func'] is bow.bow_stats.tf_double_norm and K >= 0:
        tfidf_opts['K'] = K

    idf_funcs = (
        bow.bow_stats.idf,
        bow.bow_stats.idf_probabilistic,
    )
    tfidf_opts['idf_func'] = idf_funcs[idf_func]

    if tfidf_opts['idf_func'] is bow.bow_stats.idf:
        if smooth_log >= 0:
            tfidf_opts['smooth_log'] = smooth_log
        if smooth_df >= 0:
            tfidf_opts['smooth_df'] = smooth_df
    elif tfidf_opts['idf_func'] is bow.bow_stats.idf_probabilistic and smooth >= 0:
        tfidf_opts['smooth'] = smooth

    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    if dtm.ndim != 2 or 0 in dtm.shape:
        with pytest.raises(ValueError):
            bow.bow_stats.tfidf(dtm, **tfidf_opts)
    else:
        res = bow.bow_stats.tfidf(dtm, **tfidf_opts)
        assert res.ndim == 2
        assert res.shape == dtm.shape
        assert res.dtype.kind == 'f'

        # only "double norm" does not retain sparse matrices
        if matrix_type == 1 and tfidf_opts['tf_func'] is not bow.bow_stats.tf_double_norm:
            assert issparse(res)
        else:
            assert isinstance(res, np.ndarray)


def test_tfidf_example():
    dtm = np.array([
        [0, 1, 2, 3, 4, 5],
        [1, 1, 0, 2, 2, 0],
        [2, 1, 0, 1, 0, 0]
    ])

    dtm_sparse_csr = csr_matrix(dtm)
    dtm_sparse_coo = coo_matrix(dtm)

    expected = np.array([
        [0., 0.03730772, 0.1221721, 0.11192316, 0.18483925, 0.30543024],
        [0.11552453, 0.0932693, 0., 0.1865386, 0.23104906, 0.],
        [0.34657359, 0.13990395, 0., 0.13990395, 0., 0.]
    ])

    assert np.allclose(bow.bow_stats.tfidf(dtm), expected)
    assert np.allclose(bow.bow_stats.tfidf(dtm_sparse_csr).A, expected)
    assert np.allclose(bow.bow_stats.tfidf(dtm_sparse_coo).A, expected)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1),
    lo_thresh=st.integers(min_value=-1, max_value=10),
    hi_thresh=st.integers(min_value=-1, max_value=10),
    top_n=st.integers(min_value=0, max_value=10),
    ascending=st.booleans(),
)
def test_sorted_terms(dtm, matrix_type, lo_thresh, hi_thresh, top_n, ascending):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    if lo_thresh < 0:
        lo_thresh = None

    if hi_thresh < 0:
        hi_thresh = None

    if top_n < 1:
        top_n = None

    vocab = [chr(x) for x in range(65, 65 + dtm.shape[1])]

    if lo_thresh is not None and hi_thresh is not None and lo_thresh > hi_thresh:
        with pytest.raises(ValueError):
            bow.bow_stats.sorted_terms(dtm, vocab, lo_thresh, hi_thresh, top_n, ascending)
    else:
        res = bow.bow_stats.sorted_terms(dtm, vocab, lo_thresh, hi_thresh, top_n, ascending)

        assert isinstance(res, list)
        assert len(res) == dtm.shape[0]

        for doc in res:
            if not doc: continue

            terms, vals = zip(*doc)
            terms = list(terms)
            vals = list(vals)

            assert len(terms) == len(vals)
            assert all([t in vocab for t in terms])

            if lo_thresh is not None:
                assert all([v > lo_thresh for v in vals])

            if hi_thresh is not None:
                assert all([v <= hi_thresh for v in vals])

            if top_n is not None:
                assert len(terms) <= top_n

            if ascending:
                assert sorted(vals) == vals
            else:
                assert sorted(vals, reverse=True) == vals


def test_sorted_terms_example():
    dtm = np.array([
        [1, 2, 0, 3],
        [3, 0, 0, 9],
        [0, 0, 2, 1],
    ])

    vocab = list('abcd')

    expected = [
        [('d', 3), ('b', 2)],
        [('d', 9), ('a', 3)],
        [('c', 2), ('d', 1)],
    ]

    result = bow.bow_stats.sorted_terms(dtm, vocab, top_n=2)

    assert isinstance(result, list)
    assert len(result) == len(expected)

    for res_doc, exp_doc in zip(result, expected):
        assert len(res_doc) == len(exp_doc)
        for res_tuple, exp_tuple in zip(res_doc, exp_doc):
            assert res_tuple == exp_tuple


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1),
    lo_thresh=st.integers(min_value=-1, max_value=10),
    hi_thresh=st.integers(min_value=-1, max_value=10),
    top_n=st.integers(min_value=0, max_value=10),
    ascending=st.booleans(),
)
def test_sorted_terms_table(dtm, matrix_type, lo_thresh, hi_thresh, top_n, ascending):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    if lo_thresh < 0:
        lo_thresh = None

    if hi_thresh < 0:
        hi_thresh = None

    if top_n < 1:
        top_n = None

    vocab = [chr(x) for x in range(65, 65 + dtm.shape[1])]
    doc_labels = ['doc' + str(i) for i in range(dtm.shape[0])]

    if lo_thresh is not None and hi_thresh is not None and lo_thresh > hi_thresh:
        with pytest.raises(ValueError):
            bow.bow_stats.sorted_terms_table(dtm, vocab, doc_labels, lo_thresh, hi_thresh, top_n, ascending)
    else:
        res = bow.bow_stats.sorted_terms_table(dtm, vocab, doc_labels, lo_thresh, hi_thresh, top_n, ascending)

        assert isinstance(res, pd.DataFrame)
        assert res.columns.tolist() == ['token', 'value']
        assert res.index.names == ['doc', 'rank']


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_dtm_to_dataframe(dtm, matrix_type):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
        dtm_arr = dtm.A
    else:
        dtm_arr = dtm

    doc_labels = ['doc%d' % i for i in range(dtm.shape[0])]
    vocab = ['t%d' % i for i in range(dtm.shape[1])]

    # check invalid doc_labels
    if len(doc_labels) > 0:
        with pytest.raises(ValueError):
            bow.dtm.dtm_to_dataframe(dtm, doc_labels[:-1], vocab)

    # check invalid vocab
    if len(vocab) > 0:
        with pytest.raises(ValueError):
            bow.dtm.dtm_to_dataframe(dtm, doc_labels, vocab[:-1])

    # check with valid doc_labels and vocab
    df = bow.dtm.dtm_to_dataframe(dtm, doc_labels, vocab)
    assert df.shape == dtm.shape
    assert np.array_equal(df.to_numpy(), dtm_arr)
    assert np.array_equal(df.index.values, doc_labels)
    assert np.array_equal(df.columns.values, vocab)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1)
)
def test_dtm_to_gensim_corpus_and_gensim_corpus_to_dtm(dtm, matrix_type):
    if not GENSIM_INSTALLED:
        pytest.skip('gensim not installed')

    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    gensim_corpus = bow.dtm.dtm_to_gensim_corpus(dtm)
    assert isinstance(gensim_corpus, gensim.matutils.Sparse2Corpus)
    assert len(gensim_corpus) == dtm.shape[0]

    # convert back
    dtm_ = bow.dtm.gensim_corpus_to_dtm(gensim_corpus)
    assert isinstance(dtm_, coo_matrix)


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1),
    as_gensim_dictionary=st.booleans()
)
def test_dtm_and_vocab_to_gensim_corpus_and_dict(dtm, matrix_type, as_gensim_dictionary):
    if not GENSIM_INSTALLED:
        pytest.skip('gensim not installed')

    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    vocab = ['t%d' % i for i in range(dtm.shape[1])]

    gensim_corpus, id2word = bow.dtm.dtm_and_vocab_to_gensim_corpus_and_dict(dtm, vocab,
                                                                             as_gensim_dictionary=as_gensim_dictionary)
    assert isinstance(gensim_corpus, gensim.matutils.Sparse2Corpus)
    assert len(gensim_corpus) == dtm.shape[0]

    if as_gensim_dictionary:
        assert isinstance(id2word, gensim.corpora.Dictionary)
    else:
        assert isinstance(id2word, dict)
