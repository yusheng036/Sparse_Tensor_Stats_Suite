from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass
class StatsMNC:
    hr: list[int]
    hc: list[int]
    her: list[int]
    hec: list[int]


def density_map():
    ...

def sketch(axis: str, nnz: np.ndarray) -> StatsMNC:
    ...


def MNC(A: sparse.spmatrix, B: sparse.spmatrix) -> float:
    hA = sketch("A", A)
    hB = sketch("B", B)
    A_rows, A_cols = A.shape
    _ , B_cols = B.shape

    nnz_hAr = sum(1 for x in hA.hr if x != 0)
    nnz_hBc = sum(1 for x in hB.hc if x != 0)

    nnz = 0

    if max(hA.hr) <= 1 or max(hB.hc) <= 1:
        nnz = sum(a * b for a, b in zip(hA.hc, hB.hr))

    elif any(x != 0 for x in hA.hec) or any(x != 0 for x in hB.her):
        exact_nnz = sum(
            A_hec * B_hr + (A_hc - A_hec) * B_her for A_hec, B_hr, A_hc, B_her in zip(hA.hec, hB.hr, hA.hc, hB.her)
                    )
        p = (nnz_hAr- sum(1 for x in hA.hr if x == 1)) * (nnz_hBc -  sum(1 for x in hB.hc if x == 1))

        remaining_hA = [A_hc - A_hec for A_hc, A_hec in zip(hA.hc, hA.hec)]
        remaining_hB = [B_hr - B_her for B_hr, B_her in zip(hB.hr, hB.her)]
        nnz = exact_nnz + density_map(remaining_hA, remaining_hB, p) * p

    else:
        p = nnz_hAr * nnz_hBc
        nnz = density_map(hA.hc, hB.hr, p) * p

    low_A = sum(1 for x in hA.hr if x > A_cols/2)
    low_B = sum(1 for x in hB.hc if x > A_cols/2)
    nnz = max(nnz, low_A * low_B)
    return nnz


