from __future__ import annotations

import pytest
import numpy as np

from src.lrb import lrb_matmul_stats, lrb_3d_matmul_stats
from src.suitesparse_util import (
    ground_truth,
    load_suitesparse_matrix,
)

SUITESPARSE = [
    ("HB", "arc130"), # Materials Problem
    ("Hohn", "sinc12"), # Materials Problem
    ("Precima", "analytics"), # Data Analytics Problem
    ("HB", "ash219"), # Least Squares Problem
    ("HB", "ash958"), # Least Squares Problem
    ("HB", "bcspwr01"), # Power Network Problem
    ("HB", "bcspwr06"), # Power Network Problem
    ("HB", "bcsstk01"), # Structural Problem
    ("HB", "bcsstk02"), # Structural Problem
    ("HB", "bcsstk07"), # Duplicate Structural Problem
    ("HB", "bcsstk11"), # Duplicate Structural Problem
    ("HB", "bcsstk12"), # Computational Fluid Dynamics Problem
    ("HB", "sherman4"), # Computational Fluid Dynamics Problem
    ("Oberwolfach", "rail_1357"), # Model Reduction Problem
    ("Oberwolfach", "rail_20209"), # Model Reduction Problem
    ("SNAP", "ca-GrQc"), # Undirected Graph
    ("DIMACS10", "uk"), # Undirected Graph
    ("Arenas", "celegans_metabolic"), # Undirected Multigraph
    ("DIMACS10", "kron_g500-logn16"), # Undirected Multigraph
    ("Gset", "G48"), # Undirected Random Graph
    ("DIMACS10", "rgg_n_2_16_s0"), # Undirected Random Graph
    ("Gaertner", "nopoly"), # Undirected Weight Graph
    ("Pajek", "GD97_b"), # Undirected Weight Graph
    ("SNAP", "email-Enron"), # Directed Graph
    ("MathWorks", "Harvard500"), # Directed Graph
    ("Newman", "polblogs"), # Directed Multigraph
    ("Pajek", "SmaGri"), # Directed Multigraph
    ("Pajek", "GD01_a"), # Directed Weighted Graph
    ("HB", "gre_216b"), # Directed Weighted Graph
    ("LPnetlib", "lp_adlittle"), # Linear Programming Problem
    ("Meszaros", "deter3"), # Linear Programming Problem
    ("AG-Monien", "netz4504"), # 2D/3D Problem
    ("AG-Monien", "shock-9"), # 2D/3D Problem
    ("HB", "young3c"), # Acoustics Problem
    ("Cote", "mplate"), # Acoustics Problem
    ("Pajek", "Sandi_sandi"), # Bipartite Graph
    ("Pajek", "divorce"), # Bipartite Graph
    ("JGD_Homology", "ch7-9-b2"), # Combinatorial Problem
    ("JGD_Homology", "ch7-8-b3"), # Combinatorial Problem
    ("Luong", "photogrammetry2"), # Computer Graphics/Vision Problem
    ("MathWorks", "tomography"), # Computer Graphics/Vision Problem
    ("Rommes", "zeros_nopss_13k"), # Eigenvalue/Model Reduction Problem
    ("Rommes", "ww_36_pmec_36"), # Eigenvalue/Model Reduction Problem
    ("HB", "abb313"), # Least Squares Problem
    ("HB", "ash219"), # Least Squares Problem
    ("HB", "fs_541_3"), # Subsequent 2D/3D Problem
    ("HB", "fs_541_4"), # Subsequent 2D/3D Problem
    ("HB", "shl_200"), # Subsequent Optimization Problem
    ("HB", "shl_200"), # Subsequent Optimization Problem
    ("Pajek", "Cities"), # Weighted Bipartite Graph
    ("Pajek", "WorldCities"), # Weighted Bipartite Graph
]

@pytest.mark.parametrize("group,name", SUITESPARSE)
def test_lrb_on_suitesparse(group, name):
    A = load_suitesparse_matrix(group, name).tocsr()
    B = load_suitesparse_matrix(group, name).transpose().tocsr()

    I, J = A.shape
    assert B.shape == (J, I)

    true_nnz = ground_truth(A, B)

    a = np.diff(A.tocsc().indptr)
    b = np.diff(B.tocsr().indptr)
    R = np.count_nonzero((a > 0) & (b > 0))
    R = max(1, min(R, J))

    bound = lrb_matmul_stats(A, B, regions=R)

    assert true_nnz <= bound + 1e-9, (
        f"LRB violated for {group}/{name}: true={true_nnz} bound={bound} R={R}"
    )
    assert bound <= (I * I) + 1e-9, (
        f"LRB exceeded dense cap for {group}/{name}: bound={bound} cap={I * I} R={R}"
    )

@pytest.mark.parametrize("group,name", SUITESPARSE)
def test_lrb_3d_suitesparse(group, name):
    A = load_suitesparse_matrix(group, name).tocsr()
    B = load_suitesparse_matrix(group, name).transpose().tocsr()

    I, J = A.shape
    K = B.shape[1]

    a = np.diff(A.tocsc().indptr)
    b = np.diff(B.tocsr().indptr)
    true_nnz = (a * b).sum()

    R = np.count_nonzero((a > 0) & (b > 0))
    R = max(1, min(R, J))

    bound = lrb_3d_matmul_stats(A, B, regions=R)

    assert true_nnz <= bound + 1e-9, (
        f"{group}/{name} 3D violated: true={true_nnz} bound={bound} R={R}"
    )
    assert bound <= (I * J * K) + 1e-9, (
        f"{group}/{name} 3D cap: bound={bound} cap={I * J * K} R={R}"
    )