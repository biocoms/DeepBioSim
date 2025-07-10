# utils.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import os
from scipy.spatial.distance import pdist, squareform

EPS = 1e-8


def plot_pca(orig: np.ndarray, gen: np.ndarray, method_name: str, dataset_name: str):
    combined = np.vstack([orig, gen])
    pca = PCA(n_components=2)
    comp2d = pca.fit_transform(combined)
    orig2d = comp2d[: orig.shape[0]]
    gen2d = comp2d[orig.shape[0] :]
    plt.figure(figsize=(8, 6))
    plt.scatter(orig2d[:, 0], orig2d[:, 1], s=10, alpha=0.5, label="Original")
    plt.scatter(gen2d[:, 0], gen2d[:, 1], s=10, alpha=0.5, label=method_name)
    # plt.title(f"{dataset_name}: Original vs {method_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # plt.legend()
    plt.tight_layout()
    out_path = f"./output/{dataset_name}_{method_name}_pca.eps"
    plt.savefig(out_path, format="eps", dpi=1200)
    plt.close()
    print(f"Saved PCA plot to {out_path}")


def plot_tsne(orig: np.ndarray, gen: np.ndarray, method_name: str, dataset_name: str):
    """
    Generate and save a t-SNE scatter plot comparing original and generated samples.
    Same signature as plot_comparison.
    """

    combined = np.vstack([orig, gen])
    tsne = TSNE(n_components=2, random_state=42)
    emb = tsne.fit_transform(combined)
    orig_emb = emb[: orig.shape[0]]
    gen_emb = emb[orig.shape[0] :]
    plt.figure(figsize=(8, 6))
    plt.scatter(orig_emb[:, 0], orig_emb[:, 1], s=10, alpha=0.5, label="Original")
    plt.scatter(gen_emb[:, 0], gen_emb[:, 1], s=10, alpha=0.5, label=method_name)
    # plt.title(f"{dataset_name}: Original vs {method_name} (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # plt.legend()
    plt.tight_layout()
    out_path = f"./output/{dataset_name}_{method_name}_tsne.eps"
    plt.savefig(out_path, format="eps", dpi=1200)
    plt.close()
    print(f"Saved t-SNE plot to {out_path}")


def plot_umap(orig: np.ndarray, gen: np.ndarray, method_name: str, dataset_name: str):
    """
    Generate and save a UMAP scatter plot comparing original and generated samples.
    Same signature as plot_pca and plot_tsne.
    """

    combined = np.vstack([orig, gen])
    reducer = UMAP(n_components=2)
    emb = reducer.fit_transform(combined)
    orig_emb = emb[: orig.shape[0]]
    gen_emb = emb[orig.shape[0] :]
    plt.figure(figsize=(8, 6))
    plt.scatter(orig_emb[:, 0], orig_emb[:, 1], s=10, alpha=0.5, label="Original")
    plt.scatter(gen_emb[:, 0], gen_emb[:, 1], s=10, alpha=0.5, label=method_name)
    # plt.title(f"{dataset_name}: Original vs {method_name} (UMAP)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
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
        out_dir, f"{dataset_name}_{metric_name.replace(' ', '_')}_violin.eps"
    )
    plt.savefig(out_path, format="eps", dpi=1200)
    plt.close(fig)
    print(f"Saved violin plot to {out_path}")


def shannon(mat):
    proportions = mat / mat.sum(axis=0, keepdims=True)
    # replace 0s and nans with 1
    proportions = np.where(proportions > 0, proportions, 1)  # avoid log(0)
    return -np.sum(proportions * np.log(proportions), axis=0)


def richness(mat: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    # mat > threshold gives a boolean array; sum over features axis
    return np.sum(mat > threshold, axis=0)


def bc_matrix(mat: np.ndarray) -> np.ndarray:
    # X: (n_samples, n_taxa) array of counts or relative abundances
    # If counts, you may want to convert to relative abundances per sample:
    mat += EPS
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
    presence = mat > 0.5
    # import pdb

    # pdb.set_trace()

    # 2) Compute condensed Jaccard distances
    #    SciPy treats boolean arrays natively for 'jaccard'
    jc_condensed = pdist(presence, metric="jaccard")

    # 3) Convert to a full square form
    return squareform(jc_condensed)


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
