from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from suitesparse_util import load_suitesparse_matrix, ground_truth
from lrb import lrb_matmul_stats, lrb_3d_matmul_stats


def eval_one_matrix(group: str, name: str) -> dict:
    M = load_suitesparse_matrix(group, name).tocsr()
    A = M
    B = M.transpose().tocsr()

    I, J = A.shape
    K = B.shape[1]

    a = np.diff(A.tocsc().indptr).astype(np.int64)
    b = np.diff(B.tocsr().indptr).astype(np.int64)
    true3d = (a * b).sum()
    true2d = ground_truth(A, B)

    R = max(1, np.count_nonzero((a > 0) & (b > 0)))
    bound2d = lrb_matmul_stats(A, B, regions=R)
    bound3d = lrb_3d_matmul_stats(A, B, regions=R)

    return {
        "group": group,
        "name": name,
        "I": I,
        "J": J,
        "K": K,
        "true2d": true2d,
        "true3d": true3d,
        "bound2d": bound2d,
        "bound3d": bound3d,
    }


def run_suite(suitesparse_list):
    rows = []
    for group, name in suitesparse_list:
        try:
            row = eval_one_matrix(group, name)
            row["error"] = None
            rows.append(row)
        except Exception as e:
            rows.append({"group": group, "name": name, "error": str(e)})
    return pd.DataFrame(rows)


def plot_2d_tightness(df: pd.DataFrame):

    if "error" in df.columns:
        ok = df[df["error"].isna()].copy()
    else:
        ok = df.copy()

    ok["tight2d"] = ok["bound2d"] / ok["true2d"]
    ok = ok.sort_values("tight2d", ascending=False)
    labels = [f"{g}/{n}" for g, n in zip(ok["group"], ok["name"])]
    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x, ok["tight2d"].to_numpy())
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.yscale("log")
    plt.title(f"Localized Region Bound (LRB) Tightness for 2D Output Tensor C_ik")
    plt.xlabel("Matrix (sorted by decreasing tightness)")
    plt.ylabel("Tightness (LRB/ true nnz(C_ik))")
    plt.tight_layout()
    plt.show()


def plot_3d_tightness(df: pd.DataFrame):

    if "error" in df.columns:
        ok = df[df["error"].isna()].copy()
    else:
        ok = df.copy()

    ok["tight3d"] = ok["bound3d"] / ok["true3d"]
    ok = ok.sort_values("tight3d", ascending=False)
    labels = [f"{g}/{n}" for g, n in zip(ok["group"], ok["name"])]
    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x, ok["tight3d"].to_numpy())
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.yscale("log")
    plt.title(f"Localized Region Bound (LRB) Tightness for 3D Output Tensor C_ijk")
    plt.xlabel("Matrix (sorted by decreasing tightness)")
    plt.ylabel("Tightness (LRB / true nnz(C_ijk))")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
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

    df = run_suite(SUITESPARSE)
    df = df.drop(columns=["error"], errors="ignore")
    df.to_csv("lrb_tightness_results.csv", index=False)
    print(df)

    plot_2d_tightness(df)
    plot_3d_tightness(df)