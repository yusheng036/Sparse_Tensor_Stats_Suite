from __future__ import annotations

import pytest
import sys
import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lrb import lrb_matmul_stats, lrb_3d_matmul_stats
from suitesparse_util import (
    load_first_mtx_from_tar,
    suitesparse_tar_url,
    structural_nnz_matmul,
    regions_list,
    download_bytes,
    load_suitesparse_matrix,
)

SUITESPARSE = [
    ("HB", "arc130"),
    ("HB", "ash219"),
    ("HB", "bcspwr01"),
    ("HB", "bcsstk01"),
    ("HB", "bcsstk02"),
    ("HB", "bcsstk03"),
    ("HB", "bcsstk04"),
    ("HB", "bcsstk05"),
    ("HB", "bcsstk06"),
    ("HB", "bcsstk07"),
    ("HB", "bcsstk08"),
    ("HB", "bcsstk09"),
    ("HB", "bcsstk10"),
    ("HB", "bcsstk11"),
    ("HB", "bcsstk12"),
    ("HB", "bcsstk13"),
    ("HB", "bcsstk14"),
    ("HB", "bcsstk15"),
    ("Oberwolfach", "rail_1357"),
    ("Oberwolfach", "rail_20209"),
    ("SNAP", "ca-GrQc"),
    ("SNAP", "email-Enron"),
    ("LPnetlib", "lp_afiro"),
    ("LPnetlib", "lp_adlittle"),
    ("Schenk_AFE", "af_0_k101"),
]

@pytest.mark.parametrize("group,name", SUITESPARSE)
def test_lrb_on_suitesparse(group: str, name: str):
    url = suitesparse_tar_url(group, name)
    tar_bytes = download_bytes(url)
    M = load_first_mtx_from_tar(tar_bytes)

    M = M.tocsr()
    if M.nnz:
        M.data[:] = 1

    A = M
    B = M.transpose().tocsr()

    I, J = A.shape
    assert B.shape == (J, I)

    true_nnz = structural_nnz_matmul(A, B)

    for regions in regions_list(J):
        bound = lrb_matmul_stats(A, B, regions=regions)

        dense_cap = I * I
        assert true_nnz <= bound + 1e-9, (
            f"LRB violated for {group}/{name}: true={true_nnz} bound={bound} regions={regions}"
        )
        assert bound <= dense_cap + 1e-9, (
            f"LRB exceeded dense cap for {group}/{name}: bound={bound} cap={dense_cap} regions={regions}"
        )

@pytest.mark.parametrize("group,name", SUITESPARSE[:6])
def test_lrb_3d_suitesparse(group: str, name: str):
    M = load_suitesparse_matrix(group, name)
    A = M.tocsr()
    B = M.transpose().tocsr()

    I, J = A.shape
    K = B.shape[1]

    a = np.diff(A.tocsc().indptr)
    b = np.diff(B.tocsr().indptr)
    true_nnz = (a * b).sum()

    for R in regions_list(J):
        bound = lrb_3d_matmul_stats(A, B, regions=R)
        assert true_nnz <= bound + 1e-9, (
            f"{group}/{name} 3D violated: true={true_nnz} bound={bound} R={R}"
        )
        assert bound <= (I * J * K) + 1e-9, (
            f"{group}/{name} 3D cap: bound={bound} cap={I * J * K} R={R}"
        )
