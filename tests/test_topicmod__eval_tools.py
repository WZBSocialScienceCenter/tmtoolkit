import pytest
from scipy.sparse import coo_matrix, issparse
from hypothesis import given, strategies as st

from ._testtools import strategy_dtm

from tmtoolkit.topicmod._eval_tools import split_dtm_for_cross_validation


@given(
    dtm=strategy_dtm(),
    matrix_type=st.integers(min_value=0, max_value=1),
    n_folds=st.integers(min_value=0, max_value=20)
)
def test_split_dtm_for_cross_validation(dtm, matrix_type, n_folds):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)

    if n_folds < 2 or n_folds > dtm.shape[0]:
        with pytest.raises(ValueError):
            next(split_dtm_for_cross_validation(dtm, n_folds))
    else:
        n_docs, n_vocab = dtm.shape

        n_generated_folds = 0
        for fold, train_dtm, test_dtm in split_dtm_for_cross_validation(dtm, n_folds):
            assert 0 <= fold < n_folds

            if matrix_type == 1:
                assert issparse(train_dtm)
                assert issparse(test_dtm)

            assert train_dtm.ndim == test_dtm.ndim == 2

            assert train_dtm.shape[0] >= test_dtm.shape[0]
            assert 0 < test_dtm.shape[0] <= n_docs // n_folds
            assert train_dtm.shape[0] + test_dtm.shape[0] == n_docs
            assert train_dtm.shape[1] == test_dtm.shape[1] == n_vocab

            n_generated_folds += 1

        assert n_folds == n_generated_folds
