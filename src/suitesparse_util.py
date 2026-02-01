from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Union

import numpy as np
from scipy import sparse
from scipy.io import mmread
from urllib.parse import quote

def suitesparse_tar_url(group: str, name: str) -> str:
    return f"https://sparse.tamu.edu/MM/{quote(group, safe="")}/{quote(name, safe="")}.tar.gz"


def download_cached(url: str, cache_dir: str = ".cache/suitesparse") -> bytes:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / url.split("/")[-1]
    if not path.exists():
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    return path.read_bytes()

def load_first_mtx_from_tar(tar_gz: Union[bytes, Path, str]) -> sparse.csr_matrix:
    if isinstance(tar_gz, (Path, str)):
        tf = tarfile.open(tar_gz, mode="r:gz")
    elif isinstance(tar_gz, (bytes, bytearray, memoryview)):
        tf = tarfile.open(fileobj=io.BytesIO(tar_gz), mode="r:gz")
    else:
        raise TypeError(f"Unsupported tar_gz type: {type(tar_gz)}")

    with tf:
        mtx_members = [m for m in tf.getmembers() if m.name.endswith(".mtx")]
        if not mtx_members:
            raise RuntimeError("No .mtx file found in tarball")

        f = tf.extractfile(mtx_members[0])
        assert f is not None

        A = mmread(f)
        A = sparse.csr_matrix(A) if not sparse.issparse(A) else A.tocsr()

    if A.nnz:
        A.data = np.ones_like(A.data, dtype=np.int8)
    return A

def load_suitesparse_matrix(group: str, name: str) -> sparse.csr_matrix:
    npz_cache = Path(".cache/npz")
    npz_cache.mkdir(parents=True, exist_ok=True)
    npz_path = npz_cache / f"{group}_{name}.npz"

    if npz_path.exists():
        return sparse.load_npz(npz_path).tocsr()

    url = suitesparse_tar_url(group, name)
    tar_path = download_cached(url)
    A = load_first_mtx_from_tar(tar_path)
    sparse.save_npz(npz_path, A)
    return A

def regions_list(J: int) -> list[int]:
    cand = [1, 2, 4, 8, 16, 32, 64, 128]
    return [r for r in cand if r <= J] or [1]

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