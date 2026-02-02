import pytest
import numpy as np

from src.lrb import (
    lrb_matmul_stats,
    lrb_3d_matmul_stats,
)
from src.suitesparse_util import (
    ground_truth,
    rand_csr,
)

@pytest.mark.parametrize("I,J,K", [(30, 40, 25), (10, 12, 8)])
@pytest.mark.parametrize("density", [0.01, 0.05])
def test_lrb_2d(I, J, K, density):
    A = rand_csr(I, J, density, seed=123)
    B = rand_csr(J, K, density, seed=456)
    true_nnz = ground_truth(A, B)

    a = np.diff(A.tocsc().indptr).astype(np.int64)
    b = np.diff(B.tocsr().indptr).astype(np.int64)
    R = np.count_nonzero((a > 0) & (b > 0))
    R = max(1, min(R, J))
    bound = lrb_matmul_stats(A, B, regions=R)

    assert true_nnz <= bound + 1e-9
    assert bound <= I * K + 1e-9


@pytest.mark.parametrize("I,J,K", [(20, 30, 15), (10, 12, 8)])
@pytest.mark.parametrize("density", [0.01, 0.05])
def test_lrb_3d(I, J, K, density):
    A = rand_csr(I, J, density, seed=123)
    B = rand_csr(J, K, density, seed=456)

    a = np.diff(A.tocsc().indptr).astype(np.int64)
    b = np.diff(B.tocsr().indptr).astype(np.int64)
    true_nnz = (a * b).sum()

    R = np.count_nonzero((a > 0) & (b > 0))
    R = max(1, min(R, J))
    bound = lrb_3d_matmul_stats(A, B, regions=R)

    assert true_nnz <= bound + 1e-9
    assert bound <= I * J * K + 1e-9
