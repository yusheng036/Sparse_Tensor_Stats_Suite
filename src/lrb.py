from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class RegionNNZ:
    """
    Per-region data of nonzero counts along one axis.
    """
    axis: str
    regions: int
    width: int
    nonempty_counts: np.ndarray
    max_region_nnz: np.ndarray
    sum_region_nnz: np.ndarray


def nnz_to_regions(axis: str, nnz: np.ndarray, regions: int) -> RegionNNZ:
    nnz = np.asarray(nnz, dtype=np.int64)
    if regions <= 0:
        raise ValueError("regions must be positive")

    width = (nnz.shape[0] + regions - 1) // regions

    nonempty = np.zeros(regions, dtype=np.int64)
    max_regions = np.zeros(regions, dtype=np.int64)
    sum_regions = np.zeros(regions, dtype=np.int64)

    for idx in range(nnz.shape[0]):
        rid = min(idx // width, regions - 1)
        sum_regions[rid] += nnz[idx]

        if nnz[idx] <= 0:
            continue

        nonempty[rid] += 1
        if nnz[idx] > max_regions[rid]:
            max_regions[rid] = nnz[idx]

    return RegionNNZ(
        axis=axis,
        regions=regions,
        width=width,
        nonempty_counts=nonempty,
        max_region_nnz=max_regions,
        sum_region_nnz=sum_regions,
    )


def lrb_matmul_stats(
    A: sparse.spmatrix, B: sparse.spmatrix, *, regions: int
) -> float:
    """
    Localized Region Bound for 2D matmul:

        C[i, k] = A[i, j] * B[j, k]

    Returns:
        Upper bound on nnz(C)
    """
    if regions <= 0:
        raise ValueError("regions must be positive")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: A{A.shape} @ B{B.shape}")

    I, J = A.shape
    _, K = B.shape

    A_j = np.diff(A.tocsc().indptr).astype(np.int64)
    B_j = np.diff(B.tocsr().indptr).astype(np.int64)

    regA = nnz_to_regions("j", A_j, regions)
    regB = nnz_to_regions("j", B_j, regions)

    total = 0.0
    for r in range(regions):
        amax = regA.max_region_nnz[r]
        bmax = regB.max_region_nnz[r]
        nnzA_r = regA.sum_region_nnz[r]
        nnzB_r = regB.sum_region_nnz[r]

        if amax == 0 or bmax == 0 or nnzA_r == 0 or nnzB_r == 0:
            continue

        total += min(nnzA_r * bmax, amax * nnzB_r)
    return min(total, I * K)


def lrb_3d_matmul_stats(
    A: sparse.spmatrix,
    B: sparse.spmatrix,
    regions: int,
) -> dict[str, float | int]:
    """
    Localized Region Bound for 3D matmul:

        C[i, j, k] = A[i, j] * B[j, k]

    Returns:
        Upper bound on nnz(C)
    """
    if regions <= 0:
        raise ValueError("regions must be positive")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: A{A.shape} @ B{B.shape}")

    I, J = A.shape
    _, K = B.shape

    A_j = np.diff(A.tocsc().indptr).astype(np.int64)
    B_j = np.diff(B.tocsr().indptr).astype(np.int64)

    regA = nnz_to_regions("j", A_j, regions)
    regB = nnz_to_regions("j", B_j, regions)

    total = 0.0
    for r in range(regions):
        amax = regA.max_region_nnz[r]
        bmax = regB.max_region_nnz[r]
        nnzA_r = regA.sum_region_nnz[r]
        nnzB_r = regB.sum_region_nnz[r]

        if amax == 0 or bmax == 0 or nnzA_r == 0 or nnzB_r == 0:
            continue

        total += min(nnzA_r * bmax, amax * nnzB_r)
    return min(total, I * J * K)


def lrb_get_stats(A: sparse.spmatrix) -> dict[str, np.ndarray | int]:
    """
    Minimal structural stats for a sparse matrix A.

    Returns:
        {
            "nnz": total nnz,
            "row_nnz": nnz per row,
            "col_nnz": nnz per column,
            "shape": (I, J)
        }
    """

    row_nnz = np.diff(A.tocsr().indptr).astype(np.int64)
    col_nnz = np.diff(A.tocsc().indptr).astype(np.int64)

    return {
        "nnz": A.nnz,
        "row_nnz": row_nnz,
        "col_nnz": col_nnz,
        "shape": A.shape,
    }