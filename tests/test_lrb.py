import pytest
import sys
import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lrb import (
    lrb_matmul_stats,
    lrb_3d_matmul_stats,
)
from suitesparse_util import (
    structural_nnz_matmul,
    rand_csr,
    make_batches,
)

@pytest.mark.parametrize("I,J,K", [(30, 40, 25), (10, 12, 8)])
@pytest.mark.parametrize("density", [0.01, 0.05])
@pytest.mark.parametrize("regions", [1, 2, 4, 8, 16])
def test_lrb_2d(I, J, K, density, regions):
    A = rand_csr(I, J, density, seed=123)
    B = rand_csr(J, K, density, seed=456)

    true_nnz = structural_nnz_matmul(A, B)
    bound = lrb_matmul_stats(A, B, regions=min(regions, J))
    assert true_nnz <= bound + 1e-9
    assert bound <=  I * K + 1e-9

@pytest.mark.parametrize("I,J,K", [(20, 30, 15), (10, 12, 8)])
@pytest.mark.parametrize("density", [0.01, 0.05])
@pytest.mark.parametrize("regions", [1, 2, 4, 8, 16])
def test_lrb_3d(I, J, K, density, regions):
    A_batch, B_batch = make_batches(1, I, J, K, density, seed=999)
    A = A_batch[0]
    B = B_batch[0]

    a = np.diff(A.tocsc().indptr)
    b = np.diff(B.tocsr().indptr)
    true_nnz = (a * b).sum()

    bound = lrb_3d_matmul_stats(
        A,
        B,
        regions=min(regions, J),
    )

    assert true_nnz <= bound + 1e-9
    assert bound <= I * J * K + 1e-9
