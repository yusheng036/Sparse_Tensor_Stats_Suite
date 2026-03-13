import pytest

from src.cohen import cohen_estimator
from src.suitesparse_util import ground_truth, rand_csr


@pytest.mark.parametrize("I,J,K", [(30, 40, 25), (10, 12, 8), (60, 80, 50)])
@pytest.mark.parametrize("density", [0.01, 0.05, 0.1])
def test_cohen_2d(I, J, K, density):
    A = rand_csr(I, J, density, seed=123)
    B = rand_csr(J, K, density, seed=456)

    true_nnz = ground_truth(A, B)
    est = cohen_estimator(A, B, seed=0, r=128)

    assert est >= 0.0
    assert est <= (I * K) + 1e-9

    if true_nnz == 0:
        assert est <= 1e-9
    else:
        rel_err = abs(est - true_nnz) / true_nnz
        assert rel_err <= 0.35