import pytest
import numpy as np

from src.mnc import MNC, mnc_stats
from src.suitesparse_util import rand_csr


@pytest.mark.parametrize("I,J,K", [(30, 40, 25), (10, 12, 8)])
@pytest.mark.parametrize("density", [0.01, 0.05, 0.1])
def test_mnc_2d(I, J, K, density):
    A = rand_csr(I, J, density, seed=123)
    B = rand_csr(J, K, density, seed=456)

    est = MNC(A, B)

    assert est >= 0.0
    assert est <= (I * K) + 1e-9


@pytest.mark.parametrize("shape,density", [
    ((20, 30), 0.01),
    ((20, 30), 0.05),
    ((10, 12), 0.1),
])
def test_mnc_stats_shapes(shape, density):
    I, J = shape
    A = rand_csr(I, J, density, seed=123)

    stats = mnc_stats(A)

    assert stats.hr.shape == (I,)
    assert stats.hc.shape == (J,)
    assert stats.her.shape == (I,)
    assert stats.hec.shape == (J,)

    assert np.all(stats.hr >= 0)
    assert np.all(stats.hc >= 0)
    assert np.all(stats.her >= 0)
    assert np.all(stats.hec >= 0)