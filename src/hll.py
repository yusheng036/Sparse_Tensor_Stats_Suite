import hyperloglog
from scipy import sparse

def hll_cohen_estimator(A: sparse.spmatrix, B: sparse.spmatrix, error=0.05) -> float:
    m, n =  A.shape
    _, l = B.shape

    A = A.astype(bool, copy=False).tocsc()
    B = B.astype(bool, copy=False).tocsc()

    S_j = []
    for j in range(n):
        h = hyperloglog.HyperLogLog(error)
        rows = A.indices[A.indptr[j]: A.indptr[j + 1]]
        for i in rows:
            h.add(int(i))
        S_j.append(h)

    nnz = 0.0
    for k in range(l):
        h = hyperloglog.HyperLogLog(error)
        rows = B.indices[B.indptr[k]: B.indptr[k + 1]]
        for j in rows:
            h.update(S_j[j])
        nnz += len(h)

    return nnz