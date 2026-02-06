from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from suitesparse_util import load_suitesparse_matrix, ground_truth
from lrb import lrb_matmul_stats, lrb_3d_matmul_stats


def eval_one_matrix(group: str, name: str, regions: int) -> dict:
    M = load_suitesparse_matrix(group, name).tocsr()
    A = M
    B = M.transpose().tocsr()

    I, J = A.shape
    K = B.shape[1]

    a = np.diff(A.tocsc().indptr).astype(np.int64)
    b = np.diff(B.tocsr().indptr).astype(np.int64)
    true2d = ground_truth(A, B)
    true3d = (a * b).sum()
    active_j = np.count_nonzero((a > 0) & (b > 0))
    regions_req = regions
    regions_used = max(1, min(regions_req, active_j))

    R = max(1, min(regions_req, active_j))
    bound2d = lrb_matmul_stats(A, B, regions=R)
    bound3d = lrb_3d_matmul_stats(A, B, regions=R)

    return {
        "group": group,
        "name": name,
        "regions_req": regions_req,
        "regions_used": regions_used,
        "active_j": active_j,
        "I": I,
        "J": J,
        "K": K,
        "true2d": true2d,
        "true3d": true3d,
        "bound2d": bound2d,
        "bound3d": bound3d,
    }

def run_suite(suitesparse_list, region_list):
    rows = []
    for R in region_list:
        for group, name in suitesparse_list:
            rows.append(eval_one_matrix(group, name, regions=R))
    return pd.DataFrame(rows)

def plot_tightness_2d(df: pd.DataFrame, region_list, use_regions_col="regions_req"):
    df = df.copy()
    df["matrix"] = df["group"].astype(str) + "/" + df["name"].astype(str)
    df["tight2d"] = df["bound2d"] / df["true2d"]

    matrices = df["matrix"].unique().tolist()
    matrices.sort()

    offsets = (np.arange(len(region_list)) - (len(region_list) - 1) / 2.0) * 0.85 / len(region_list)
    plt.figure(figsize=(max(12, 0.45 * len(matrices)), 5))

    for i, R in enumerate(region_list):
        sub = df[df[use_regions_col] == R].set_index("matrix")
        y = np.array([sub.loc[m, "tight2d"] if m in sub.index else np.nan for m in matrices], dtype=float)
        plt.bar(np.arange(len(matrices)) + offsets[i], y, width=0.85 / len(region_list) * 0.95, label=f"{R} regions")

    plt.yscale("log")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(matrices)), matrices, rotation=45, ha="right")
    plt.xlabel("Matrix")
    plt.ylabel("Tightness (LRB / true nnz)")
    plt.title("LRB Tightness per Matrix (2D output $C_{ik}$)")
    plt.legend(ncol=min(4, len(region_list)), frameon=True)
    plt.tight_layout()
    plt.show()


def plot_tightness_3d(df: pd.DataFrame, region_list, use_regions_col="regions_req"):
    df = df.copy()
    df["matrix"] = df["group"].astype(str) + "/" + df["name"].astype(str)
    df["tight3d"] = df["bound3d"] / df["true3d"]

    matrices = df["matrix"].unique().tolist()
    matrices.sort()

    offsets = (np.arange(len(region_list)) - (len(region_list) - 1) / 2.0) * 0.85 / len(region_list)
    plt.figure(figsize=(max(12, 0.45 * len(matrices)), 5))

    for i, R in enumerate(region_list):
        sub = df[df[use_regions_col] == R].set_index("matrix")
        y = np.array([sub.loc[m, "tight3d"] if m in sub.index else np.nan for m in matrices], dtype=float)
        plt.bar(
            np.arange(len(matrices)) + offsets[i],
            y,
            width=0.85 / len(region_list) * 0.95,
            label=f"{R} regions",
        )

    plt.yscale("log")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(matrices)), matrices, rotation=45, ha="right")
    plt.xlabel("Matrix")
    plt.ylabel("Tightness (LRB / true nnz)")
    plt.title("LRB Tightness per Matrix (3D output $C_{ijk}$)")
    plt.legend(ncol=min(4, len(region_list)), frameon=True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    SUITESPARSE = [
        ("HB", "ash219"), #438
        ("HB", "ash958"), #1916
        ("HB", "bcspwr01"), #131
        ("HB", "bcspwr06"), #5300
        ("Cote", "mplate"), #100k
        ("Hohn", "sinc12"), #200k
        ("Rothberg", "cfd1"), #1mil
        ("Precima", "analytics"), #2mil
        ("Rothberg", "cfd2"), #3mil
        ("Meszaros", "deter3")

        # -------------------------------------------------------------------
        # ("Gaertner", "nopoly"), # Undirected Weight Graph #70k
        # ("DIMACS10", "rgg_n_2_16_s0"), # Undirected Random Graph #600k
        # ("Meszaros", "deter3"), # Linear Programming Problem #44k
        # ("AG-Monien", "shock-9"), # 2D/3D Problem #100k
        # ("SNAP", "email-Enron"), # Directed Graph #300k
        # ("JGD_Homology", "ch7-8-b3"), # Combinatorial Problem #200k
        # ("Rommes", "zeros_nopss_13k"), # Eigenvalue/Model Reduction Problem #48k
    ]
    REGIONS = [1, 2, 8, 32]
    df = run_suite(SUITESPARSE, REGIONS)
    df.to_csv("lrb_tightness_results.csv", index=False)
    print(df)

    plot_tightness_2d(df, REGIONS, use_regions_col="regions_req")
    plot_tightness_3d(df, REGIONS, use_regions_col="regions_req")
