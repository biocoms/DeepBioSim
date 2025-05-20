# utils.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_pca(orig: np.ndarray, gen: np.ndarray, method_name: str, dataset_name: str):
    combined = np.vstack([orig, gen])
    pca = PCA(n_components=2)
    comp2d = pca.fit_transform(combined)
    orig2d = comp2d[: orig.shape[0]]
    gen2d = comp2d[orig.shape[0] :]
    plt.figure(figsize=(8, 6))
    plt.scatter(orig2d[:, 0], orig2d[:, 1], s=10, alpha=0.5, label="Original")
    plt.scatter(gen2d[:, 0], gen2d[:, 1], s=10, alpha=0.5, label=method_name)
    plt.title(f"{dataset_name}: Original vs {method_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    out_path = f"./output/{dataset_name}_{method_name}_pca.png"
    plt.savefig(out_path)
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
    plt.title(f"{dataset_name}: Original vs {method_name} (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    out_path = f"./output/{dataset_name}_{method_name}_tsne.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved t-SNE plot to {out_path}")


def shannon(mat, pseudo=1e-6):
    mat_clipped = np.clip(mat, 0, None) + pseudo
    probs = mat_clipped / mat_clipped.sum(axis=1, keepdims=True)
    return -(probs * np.log(probs)).sum(axis=1)


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
