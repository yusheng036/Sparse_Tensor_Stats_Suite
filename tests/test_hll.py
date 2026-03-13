import pytest
import numpy as np

from src.suitesparse_util import (
    rand_csr,
    ground_truth,
)
from src.hll import hll_cohen_estimator


@pytest.mark.parametrize("I,J,K", [(30, 40, 25), (10, 12, 8)])
@pytest.mark.parametrize("density", [0.01, 0.05])
@pytest.mark.parametrize("error", [0.1, 0.05])
def test_hll_2d_estimator(I, J, K, density, error):
    A = rand_csr(I, J, density, seed=123)
    B = rand_csr(J, K, density, seed=456)

    true_nnz = ground_truth(A, B)
    est = hll_cohen_estimator(A, B, error=error)
    max_nnz = I * K

    assert est >= 0.0
    assert est <= max_nnz * 1.5 + 1e-9

    if true_nnz == 0:
        assert est <= 1e-9
    else:
        rel_err = abs(est - true_nnz) / true_nnz
        assert rel_err <= 0.35