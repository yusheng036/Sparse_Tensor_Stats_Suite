import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def mod32_matrix(n) -> sparse.csr_matrix:
    rr = np.arange(n)[np.arange(n) % 32 == 1]
    cc = np.arange(n)[np.arange(n) % 32 == 1]
    r = np.repeat(rr, len(cc))
    c = np.tile(cc, len(rr))
    data = np.ones(len(r), dtype=np.float64)
    return sparse.csr_matrix((data, (r, c)), shape=(n, n))


def banded_matrix(n, bandwidth) -> sparse.csr_matrix:
    diags = [np.ones(n)]
    offs = [0]
    for d in range(1, bandwidth):
        diags += [np.ones(n - d), np.ones(n - d)]
        offs += [d, -d]
    return sparse.diags(diags, offs, format="csr", dtype=np.float64)


def tridiagonal_matrix(n):
    diagonals = [
        -1.0 * np.ones(n - 1),
        2.0 * np.ones(n),
        -1.0 * np.ones(n - 1),
    ]
    offsets = [-1, 0, 1]

    return sparse.diags(diagonals, offsets, format="csr")


def diagonal_matrix(n):
    return sparse.diags(
        1.0 * np.ones(n),
        offsets=0,
        format="csr"
    )


def vertical_striped_matrix(n, stripe_width = 4, stripe_gap = 16) -> sparse.csr_matrix:
    rows = []
    cols = []
    data = []

    for j in range(0, n, stripe_gap):
        for w in range(stripe_width):
            col = j + w
            if col >= n:
                break
            for i in range(n):
                rows.append(i)
                cols.append(col)
                data.append(1.0)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def horizontal_striped_matrix(n, stripe_height = 4, stripe_gap = 16) -> sparse.csr_matrix:
    rows = []
    cols = []
    data = []

    for i in range(0, n, stripe_gap):
        for h in range(stripe_height):
            row = i + h
            if row >= n:
                break
            for j in range(n):
                rows.append(row)
                cols.append(j)
                data.append(1.0)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))



def upper_triangular_matrix(n) -> sparse.csr_matrix:
    rows = []
    cols = []
    data = []

    for i in range(n):
        for j in range(i, n):
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def lower_triangular_matrix(n) -> sparse.csr_matrix:
    rows = []
    cols = []
    data = []

    for i in range(n):
        for j in range(0, i + 1):
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def load_synthetic_matrix(group, name) -> sparse.csr_matrix:

    if (group, name) == ("Synthetic", "upper_triangle"):
        return upper_triangular_matrix(n=1024).tocsr()

    if (group, name) == ("Synthetic", "lower_triangle"):
        return lower_triangular_matrix(n=1024).tocsr()

    if (group, name) == ("Synthetic", "vertical_stripped"):
        return vertical_striped_matrix(n=1024, stripe_width=4, stripe_gap=16).tocsr()

    if (group, name) == ("Synthetic", "horizontal_stripped"):
        return horizontal_striped_matrix(n=1024, stripe_height=4, stripe_gap=16).tocsr()

    if (group, name) == ("Synthetic", "tridiagonal"):
        return tridiagonal_matrix(n=1024).tocsr()

    if (group, name) == ("Synthetic", "diagonal"):
        return diagonal_matrix(n=1024).tocsr()

    if (group, name) == ("Synthetic", "mod32"):
        return mod32_matrix(n=1024).tocsr()

    if (group, name) == ("Synthetic", "banded"):
        return banded_matrix(n=1024, bandwidth=8).tocsr()

    if (group, name) == ("Synthetic", "reorder_by_mod"):
        return reorder_by_mod(mod32_matrix(n=1024)).tocsr()

    if (group, name) == ("Synthetic", "reorder_vertical_stripes"):
        return reorder_vertical_stripes(vertical_striped_matrix(n=1024)).tocsr()

    if (group, name) == ("Synthetic", "reorder_horizontal_stripes"):
        return reorder_horizontal_stripes(mod32_matrix(n=1024)).tocsr()

    raise KeyError(f"Unknown synthetic matrix key: {(group, name)}")

def plot_sparsity_patterns(mats, max_n =256, markersize = 2):
    names = list(mats.keys())
    k = len(names)
    cols = 3
    rows = (k + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 4 * rows))

    for idx, name in enumerate(names, start=1):
        A = mats[name].tocsr()
        n0, n1 = A.shape
        n = min(n0, n1, max_n)
        A_view = A[:n, :n]

        plt.subplot(rows, cols, idx)
        plt.spy(A_view, markersize=markersize)
        plt.title(f"{name}\n(showing {n}×{n}, nnz={A_view.nnz})")
        plt.xlabel("col")
        plt.ylabel("row")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()

def reorder_by_mod(A):
    n = A.shape[0]
    perm = np.argsort(np.arange(n) % 32)
    return A[perm, :][:, perm]

def reorder_vertical_stripes(A):
    col_deg = np.diff(A.tocsc().indptr)
    perm = np.argsort(-col_deg)
    return A[:, perm]

def reorder_horizontal_stripes(A):
    row_deg = np.diff(A.tocsr().indptr)
    perm = np.argsort(-row_deg)
    return A[perm, :]

A = horizontal_striped_matrix(256)
A_reordered = reorder_horizontal_stripes(A).tocsr()

# plt.figure(figsize=(10,4))

# plt.subplot(1,2,1)
# plt.spy(A[:256,:256], markersize=2)
# plt.title("Original horizontal_stripe matrix")

# plt.subplot(1,2,2)
# plt.spy(A_reordered[:256,:256], markersize=2)
# plt.title("reorder_horizontal_stripe")

# plt.tight_layout()
# plt.show()

# n = 1024
# mats = {
#     "diagonal": diagonal_matrix(n),
#     "tridiagonal": tridiagonal_matrix(n),
#     "banded": banded_matrix(n, 8),
#     "mod32": mod32_matrix(n),
#     "vertical_striped": vertical_striped_matrix(n,4,16),
#     "horizontal_striped": horizontal_striped_matrix(n,4,16),
#     "upper_triangle": upper_triangular_matrix(n),
#     "lower_triangle": lower_triangular_matrix(n),
#     "reorder_by_mod": reorder_by_mod(mod32_matrix(n)),
# }

# plot_sparsity_patterns(mats, max_n=256, markersize=2)