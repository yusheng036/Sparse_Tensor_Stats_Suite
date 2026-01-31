from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmread


def suitesparse_tar_url(group: str, name: str) -> str:
    return f"https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"


def download_cached(url: str, cache_dir: str = ".cache/suitesparse") -> bytes:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1]
    path = cache / fname
    if not path.exists():
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def load_first_mtx_from_tar(tar_gz_bytes: bytes) -> sparse.csr_matrix:
    """
    Extract the first .mtx file inside the SuiteSparse .tar.gz and return CSR.
    """
    with tarfile.open(fileobj=io.BytesIO(tar_gz_bytes), mode="r:gz") as tf:
        mtx_members = [m for m in tf.getmembers() if m.name.endswith(".mtx")]
        if not mtx_members:
            raise RuntimeError("No .mtx file found in tarball")
        f = tf.extractfile(mtx_members[0])
        assert f is not None
        A = mmread(f)
        if not sparse.issparse(A):
            A = sparse.csr_matrix(A)
        else:
            A = A.tocsr()

    if A.nnz:
        A.data = np.ones_like(A.data, dtype=np.int8)
    return A

def load_suitesparse_matrix(group: str, name: str) -> sparse.csr_matrix:
    url = suitesparse_tar_url(group, name)
    tar_bytes = download_cached(url)
    return load_first_mtx_from_tar(tar_bytes)

def regions_list(J: int) -> list[int]:
    cand = [1, 2, 4, 8, 16, 32, 64, 128]
    return [r for r in cand if r <= J] or [1]

def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()

def structural_nnz_matmul(A: sparse.spmatrix, B: sparse.spmatrix) -> int:
    A2, B2 = A.copy(), B.copy()
    if A2.nnz:
        A2.data[:] = 1
    if B2.nnz:
        B2.data[:] = 1
    C = (A2 @ B2).tocsr()
    if C.nnz:
        C.data[:] = 1
        C.eliminate_zeros()
    return C.nnz

def rand_csr(m, n, density, seed) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    return sparse.random(
        m, n, density=density, format="csr",
        random_state=rng,
        data_rvs=lambda k: np.ones(k, dtype=np.int8),
    )

def make_batches(Bsz, I, J, K, density, seed=0):
    A_batch = [rand_csr(I, J, density, seed + 10*b) for b in range(Bsz)]
    B_batch = [rand_csr(J, K, density, seed + 10*b + 1) for b in range(Bsz)]
    return A_batch, B_batch