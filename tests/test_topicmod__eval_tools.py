import pytest
import numpy as np
from scipy.sparse import coo_matrix, issparse
from hypothesis import given, strategies as st

from tmtoolkit.topicmod._eval_tools import split_dtm_for_cross_validation


@given(dtm=st.lists(st.integers(1, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(0, 10),
                                   min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])
    ),
    matrix_type=st.integers(min_value=0, max_value=1),
    n_folds=st.integers(min_value=0, max_value=20))
def test_split_dtm_for_cross_validation(dtm, matrix_type, n_folds):
    if matrix_type == 1:
        dtm = coo_matrix(dtm)
    else:
        dtm = np.array(dtm)

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
