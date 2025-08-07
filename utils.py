# utils.py
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import os
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

import pdb

EPS = 1e-8


def plot_pca(orig: np.ndarray, gen: np.ndarray, method_name: str, dataset_name: str, num_samples: int = 1500):
    n_orig, n_gen = orig.shape[0], gen.shape[0]
    if n_orig + n_gen > 2 * num_samples:
        idx_orig = np.random.choice(n_orig, num_samples, replace=False)
        idx_gen = np.random.choice(n_gen, num_samples, replace=False)
        orig = orig[idx_orig]
        gen = gen[idx_gen]
    combined = np.vstack([orig, gen])
    pca = PCA(n_components=2)
    comp2d = pca.fit_transform(combined)
    orig2d = comp2d[: orig.shape[0]]
    gen2d = comp2d[orig.shape[0] :]
    plt.figure(figsize=(8, 6))
    plt.scatter(orig2d[:, 0], orig2d[:, 1], s=10, alpha=0.5, c="#1f77b4")
    plt.scatter(gen2d[:, 0], gen2d[:, 1], s=10, alpha=0.5, c="#ff7f0e")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    out_path = f"./output/{dataset_name}_{method_name}_pca.eps"
    plt.savefig(out_path, format="eps", dpi=1200)
    plt.close()
    print(f"Saved PCA plot to {out_path}")


def plot_tsne(orig: np.ndarray, gen: np.ndarray, method_name: str, dataset_name: str, num_samples: int = 1500):
    """
    Generate and save a t-SNE scatter plot comparing original and generated samples.
    Same signature as plot_comparison.
    """
    n_orig, n_gen = orig.shape[0], gen.shape[0]
    if n_orig + n_gen > 2 * num_samples:
        idx_orig = np.random.choice(n_orig, num_samples, replace=False)
        idx_gen = np.random.choice(n_gen, num_samples, replace=False)
        orig = orig[idx_orig]
        gen = gen[idx_gen]
    combined = np.vstack([orig, gen])
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    emb = tsne.fit_transform(combined)
    orig_emb = emb[: orig.shape[0]]
    gen_emb = emb[orig.shape[0] :]
    plt.figure(figsize=(8, 6))
    plt.scatter(orig_emb[:, 0], orig_emb[:, 1], s=10, alpha=0.5, c="#1f77b4")
    plt.scatter(gen_emb[:, 0], gen_emb[:, 1], s=10, alpha=0.5, c="#ff7f0e")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    # plt.legend()
    plt.tight_layout()
    out_path = f"./output/{dataset_name}_{method_name}_tsne.eps"
    plt.savefig(out_path, format="eps", dpi=1200)
    plt.close()
    print(f"Saved t-SNE plot to {out_path}")


def plot_umap(orig: np.ndarray, gen: np.ndarray, method_name: str, dataset_name: str, num_samples: int = 1500):
    """
    Generate and save a UMAP scatter plot comparing original and generated samples.
    Same signature as plot_pca and plot_tsne.
    """
    n_orig, n_gen = orig.shape[0], gen.shape[0]
    if n_orig + n_gen > 2 * num_samples:
        idx_orig = np.random.choice(n_orig, num_samples, replace=False)
        idx_gen = np.random.choice(n_gen, num_samples, replace=False)
        orig = orig[idx_orig]
        gen = gen[idx_gen]
    combined = np.vstack([orig, gen])
    reducer = UMAP(n_components=2)
    emb = reducer.fit_transform(combined)
    orig_emb = emb[: orig.shape[0]]
    gen_emb = emb[orig.shape[0] :]
    plt.figure(figsize=(8, 6))
    plt.scatter(orig_emb[:, 0], orig_emb[:, 1], s=10, alpha=0.5, c="#1f77b4")
    plt.scatter(gen_emb[:, 0], gen_emb[:, 1], s=10, alpha=0.5, c="#ff7f0e")
    # plt.title(f"{dataset_name}: Original vs {method_name} (UMAP)")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    # plt.legend()
    plt.tight_layout()
    out_path = f"./output/{dataset_name}_{method_name}_umap.eps"
    plt.savefig(out_path, format="eps", dpi=1200)
    plt.close()
    print(f"Saved UMAP plot to {out_path}")


def plot_nmds(
    orig: np.ndarray,
    gen: np.ndarray,
    method_name: str,
    dataset_name: str,
    metric: str = "braycurtis",
    random_state: int = 42,
    out_dir: str = "./output",
):
    """
    Generate and save a non-metric MDS scatter plot comparing original
    and generated samples.

    Args:
        orig:       (n_samples_orig × n_features) original data
        gen:        (n_samples_gen  × n_features) generated data
        method_name:label for the generated set (e.g. "IWAE")
        dataset_name:identifier for output filename (e.g. "ibd")
        metric:     distance metric passed to pdist (default="braycurtis")
        random_state: seed for reproducibility
        out_dir:    directory to save the .eps figure
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) stack and compute pairwise distances
    combined = np.vstack([orig, gen])
    dist_condensed = pdist(combined, metric=metric)
    dist_matrix = squareform(dist_condensed)

    # 2) run non-metric MDS
    mds = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        max_iter=3000,
    )
    emb = mds.fit_transform(dist_matrix)
    orig_emb = emb[: orig.shape[0]]
    gen_emb = emb[orig.shape[0] :]

    # 3) plot
    plt.figure(figsize=(8, 6))
    plt.scatter(orig_emb[:, 0], orig_emb[:, 1], s=10, alpha=0.5, label="Original")
    plt.scatter(gen_emb[:, 0], gen_emb[:, 1], s=10, alpha=0.5, label=method_name)
    plt.xlabel("NMDS 1")
    plt.ylabel("NMDS 2")
    # plt.legend()
    plt.tight_layout()

    # 4) save
    out_path = os.path.join(out_dir, f"{dataset_name}_{method_name}_nmds.eps")
    plt.savefig(out_path, format="eps", dpi=1200)
    plt.close()
    print(f"Saved NMDS plot to {out_path}")


def plot_violin(
    H_list: list[np.ndarray],
    labels: list[str],
    metric_name: str,
    dataset_name: str,
    out_dir: str = "./output",
):
    """
    Plot and save a violin plot of one or more 1D vectors side by side,
    with no baseline and y-limits set by the global min/max across all columns.
    """
    # pdb.set_trace()
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(1, len(H_list) + 1)

    # violin plot
    ax.violinplot(
        H_list,
        positions=positions,
        showmeans=True,
        showextrema=True,
        widths=0.7,
    )

    # scatter of individual points, jittered
    for i, H in enumerate(H_list, start=1):
        x = np.random.normal(i, 0.04, size=len(H))
        ax.scatter(x, H, s=8, color="k", alpha=0.1)

    # styling
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_name)

    # ——— NEW: global min/max across all H’s ———
    all_mins = [h.min() for h in H_list]
    all_maxs = [h.max() for h in H_list]
    ymin, ymax = min(all_mins), max(all_maxs)

    # add 10% padding
    pad = 0.1 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    plt.tight_layout()
    out_path = os.path.join(
        out_dir, f"{dataset_name}_{metric_name.replace(' ', '_')}_violin.png"
    )
    plt.savefig(out_path, format="png", dpi=1200)
    plt.close(fig)
    print(f"Saved violin plot to {out_path}")


def shannon(mat):
    # input n*p
    mat = mat.T
    proportions = mat / (mat.sum(axis=1, keepdims=True) + EPS)
    # pdb.set_trace()
    # replace 0s and nans with 1
    proportions = np.where(proportions > 0, proportions, 1)  # avoid log(0)
    return -np.sum(proportions * np.log(proportions), axis=1)


def richness(mat: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    # mat > threshold gives a boolean array; sum over features axis; input n*p
    mat= mat.T  
    return np.sum(mat > threshold, axis=1)


def bc_matrix(mat: np.ndarray) -> np.ndarray:
    mat += EPS  # input n*p
    # pdb.set_trace()
    X_rel = mat / mat.sum(axis=1, keepdims=True)

    # Compute pairwise Bray–Curtis dissimilarities
    # pdist returns a condensed distance vector
    bc_condensed = pdist(X_rel, metric="braycurtis")

    # Convert to a square form (n×n) matrix
    bc_matrix = squareform(bc_condensed)
    return bc_matrix


def jaccard_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Compute the sample-by-sample Jaccard distance matrix.

    Args:
        mat: (n_samples, n_taxa) array of non-negative counts (or abundances)

    Returns:
        (n_samples, n_samples) array of Jaccard distances between samples
    """
    # 1) Convert to boolean presence/absence
    # input n*p
    presence = mat > 0.5

    # pdb.set_trace()

    # 2) Compute condensed Jaccard distances
    #    SciPy treats boolean arrays natively for 'jaccard'
    jc_condensed = pdist(presence, metric="jaccard")

    # 3) Convert to a full square form
    return squareform(jc_condensed)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # center
    x_cent = x - np.mean(x)
    y_cent = y - np.mean(y)

    # covariance and norms
    cov = np.sum(x_cent * y_cent)
    norm = np.sqrt(np.sum(x_cent**2) * np.sum(y_cent**2))

    return cov / (norm + EPS)


def ci95(vals):
    a = np.array(vals)
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)


def match_columns(data: np.ndarray, gen: np.ndarray, mode: str = "mae") -> np.ndarray:
    """
    Permute the columns of `gen` so they best match `data`, either by
    minimizing column-wise MAE or maximizing Pearson correlation.

    Args:
        data: (n_samples, p) real-data matrix
        gen:  (n_samples, p) generated-data matrix
        mode: "mae"  → minimize mean absolute error per column
              "corr" → maximize Pearson r per column

    Returns:
        gen_reordered: (n_samples, p) same as gen but with columns permuted
    """
    n, p = data.shape
    if gen.shape != (n, p):
        raise ValueError("data and gen must have the same shape")

    if mode == "rank":
        # 1) compute squared-sum scores for original and generated features
        orig_scores = np.sum(data**2, axis=0)  # length p
        gen_scores = np.sum(gen**2, axis=0)  # length p
        # 2) sort indices descending by score
        orig_order = np.argsort(orig_scores)[::-1]
        gen_order = np.argsort(gen_scores)[::-1]
        # 3) map each original-feature position to the corresponding generated index
        col_idx = np.empty(p, dtype=int)
        col_idx[orig_order] = gen_order
        # 4) permute generated columns
        return gen[:, col_idx]

    C = np.zeros((p, p), dtype=float)

    # Precompute which columns are constant
    data_const = np.all(data == data[:1, :], axis=0)
    gen_const = np.all(gen == gen[:1, :], axis=0)

    for i in range(p):
        x = data[:, i]
        for j in range(p):
            y = gen[:, j]

            if mode == "mae":
                C[i, j] = np.mean(np.abs(x - y))

            elif mode == "corr":
                # if either column is constant, r is undefined → give max cost
                if data_const[i] or gen_const[j]:
                    C[i, j] = 1.0  # correlation r in [-1,1], cost = -r so worst is +1
                else:
                    r, _ = pearsonr(x, y)
                    # if pearsonr somehow still yields nan, treat as zero correlation
                    if np.isnan(r):
                        r = 0.0
                    C[i, j] = -r

            else:
                raise ValueError("mode must be 'mae' or 'corr'")

    # catch any remaining nan or inf
    max_cost = np.nanmax(C[np.isfinite(C)])
    C = np.nan_to_num(
        C, nan=(max_cost + 1), posinf=(max_cost + 1), neginf=(max_cost + 1)
    )

    # solve optimal assignment
    row_idx, col_idx = linear_sum_assignment(C)

    # permute gen's columns
    gen_reordered = gen[:, col_idx]
    return gen_reordered


def save_generated_samples(
    gen_samples: np.ndarray,
    method_name: str,
    dataset_name: str,
):
    """
    Save a generated‐samples matrix to a .npy file.

    Args:
        gen_samples: array of shape (n_samples, n_features)
        method_name: label for the method (e.g. "VAE", "IWAE", "MCMC")
        dataset_name: dataset identifier (e.g. "ibd")
        out_dir: where to write the file
    """
    filename = f"{dataset_name}_{method_name}_samples.npy"
    path = f"./output/{filename}"
    np.save(path, gen_samples)
    print(f"Saved generated samples to {path}")


def svd_reduce(X: np.ndarray, max_fraction: float = 0.8):
    """
    Un-centered SVD reduction of X (n_samples × n_features).
    Keeps at most floor(n_samples*max_fraction) components.
    Returns:
      Z         (n_samples × r)   — the reduced coords
      V_reduced (n_features × r)  — the projection matrix
    """
    n, d = X.shape
    # choose r = min(d, floor(n * max_fraction)), but at least 1
    r = max(1, min(d, n * max_fraction // 1))
    # full_matrices=False so U is n×min(n,d), Vt is min(n,d)×d
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # take the top-r rows of Vt → shape (r, d), then transpose to (d, r)
    V_reduced = Vt[: int(r), :].T
    # project
    Z = X @ V_reduced  # n×r
    return Z, V_reduced


def svd_reconstruct(Z: np.ndarray, V_reduced: np.ndarray):
    """
    Reconstructs an approximation of the original X from Z and V_reduced:
      X_hat = Z @ V_reduced.T
    """
    return Z @ V_reduced.T
