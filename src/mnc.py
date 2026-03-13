from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass
class StatsMNC:
    hr: np.ndarray
    hc: np.ndarray
    her: np.ndarray
    hec: np.ndarray


def mnc_stats(A: sparse.spmatrix) -> StatsMNC:
    A = A.tocsr()

    hr = np.asarray(A.getnnz(axis=1), dtype=np.int64)
    hc = np.asarray(A.getnnz(axis=0), dtype=np.int64)

    her = np.zeros(A.shape[0], dtype=np.int64)
    hec = np.zeros(A.shape[1], dtype=np.int64)

    for i in range(A.shape[0]):
        cols = A.indices[A.indptr[i]:A.indptr[i+1]]
        her[i]= sum(1 for j in cols if hc[j] <= 1)
        if len(cols) == 1:
            hec[cols[0]] += 1

    return StatsMNC(
        hr=hr,
        hc=hc,
        her=her,
        hec=hec
    )


def MNC(A: sparse.spmatrix, B: sparse.spmatrix) -> float:
    hA = mnc_stats(A)
    hB = mnc_stats(B)
    A_rows, A_cols = A.shape
    _ , B_cols = B.shape

    nnz_hAr = sum(1 for x in hA.hr if x != 0)
    nnz_hBc = sum(1 for x in hB.hc if x != 0)

    nnz = 0

    if max(hA.hr) <= 1 or max(hB.hc) <= 1:
        nnz = sum(a * b for a, b in zip(hA.hc, hB.hr))

    elif any(x != 0 for x in hA.hec) or any(x != 0 for x in hB.her):
        exact_nnz = sum(
            A_hec * B_hr + (A_hc - A_hec) * B_her for A_hec, A_hc, B_hr, B_her in zip(hA.hec, hA.hc, hB.hr, hB.her)
            )
        p = (nnz_hAr- sum(1 for x in hA.hr if x == 1)) * (nnz_hBc -  sum(1 for x in hB.hc if x == 1))

        remaining_hA = [A_hc - A_hec for A_hc, A_hec in zip(hA.hc, hA.hec)]
        remaining_hB = [B_hr - B_her for B_hr, B_her in zip(hB.hr, hB.her)]
        region = 0

        for A_hc, B_hr in zip(remaining_hA, remaining_hB):
            lsp = (A_hc * B_hr) / p
            region = region + lsp - region * lsp

        nnz = exact_nnz + (region * p)

    else:
        p = nnz_hAr * nnz_hBc
        region = 0

        for A_hc, B_hr in zip(hA.hc, hB.hr):
            lsp = (A_hc * B_hr) / p
            region = region + lsp - region * lsp

        nnz = region * p


    low_A = sum(1 for x in hA.hr if x > A_cols/2)
    low_B = sum(1 for x in hB.hc if x > A_cols/2)
    nnz = max(nnz, low_A * low_B)
    return nnz