import numpy as np
from scipy import sparse


def cohen_estimator(A: sparse.spmatrix, B: sparse.spmatrix, seed: int | None = 0, r: int = 64) -> float:
    m, n =  A.shape
    _, l = B.shape

    A, B = A.tocsc(), B.tocsc()

    rng = np.random.default_rng(seed)
    S_i = rng.exponential(scale=1.0, size=(m, r)).astype(np.float64, copy=False)

    S_j = np.full((n, r), np.inf, dtype=np.float64)
    for j in range(n):
        rows = A.indices[A.indptr[j]: A.indptr[j + 1]]
        if rows.size > 0:
            S_j[j, :] = S_i[rows, :].min(axis=0)

    S_k = np.full((l, r), np.inf, dtype=np.float64)
    for k in range(l):
        rows = B.indices[B.indptr[k]: B.indptr[k + 1]]
        if rows.size > 0:
            S_k[k, :] = S_j[rows, :].min(axis=0)

    sum_r = S_k.sum(axis=1)
    col = np.zeros(l, dtype=np.float64)
    mask = np.isfinite(sum_r)
    col[mask] = (r -1) / sum_r[mask]
    nnz = col.sum()

    return nnz
