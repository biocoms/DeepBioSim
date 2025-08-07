#!/usr/bin/env python3
"""
pred_sim.py

Load simulated GSE165512 samples from .npy (shape p×n so we transpose),
run RBF‐kPCA → 30 dims + MLP (100‐unit hidden, relu, adam, max_iter=2000, batch_size=16)
for 50 repeats (test_size=0.1), and report mean ± 95% CI of Accuracy, AUROC, Log‐Loss
for each generative method: VAE, IWAE, diffusion.

Usage:
python pred_sim.py --input_dir ./output --repeats 100 --test_size 0.2
"""
import argparse
import os

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from utils import ci95

METHODS = ["VAE", "IWAE", "diffusion"]
CLASSES = ["CD", "Control", "UC"]


def load_simulated(input_dir, method):
    """
    Expect files:
      GSE165512_CD_<method>_samples.npy
      GSE165512_Control_<method>_samples.npy
      GSE165512_UC_<method>_samples.npy
    Each is p×n, so we transpose to n×p.
    Returns X (n×p), y (n,)
    """
    arrays = []
    labels = []
    for cls in CLASSES:
        fn = f"GSE165512_{cls}_{method}_samples.npy"
        path = os.path.join(input_dir, fn)
        arr = np.load(path)  # shape p×n
        arr = arr.T  # now n×p
        arr[arr < 0] = 0  # Ensure no negative counts
        arr = np.expm1(arr)  # undo log1p
        arrays.append(arr)
        labels += [cls] * arr.shape[0]
    X = np.vstack(arrays)
    y = np.array(labels)
    return X, y


# helper to compute entropy; default base 2 so result is in bits
def entropy(probs, base=2):
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    h = -np.sum(probs * np.log(probs), axis=-1)  # natural log
    if base != np.e:
        h = h / np.log(base)
    return h


def run_pipeline(X, y, test_size, repeats):
    # We'll evaluate two classifiers: nonlinear MLP and linear logistic regression
    classifier_constructors = {
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            max_iter=2000,
            batch_size=16,
        ),
        "Linear": lambda: LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000),
    }

    inter_entropies = {name: [] for name in classifier_constructors}
    intra_entropies = {name: [] for name in classifier_constructors}

    for _ in range(repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        kpca = KernelPCA(n_components=40, kernel="rbf", fit_inverse_transform=False)
        X_tr_k = kpca.fit_transform(X_tr_s)
        X_te_k = kpca.transform(X_te_s)

        for name, ctor in classifier_constructors.items():
            clf = ctor()
            # fit and get probabilities
            clf.fit(X_tr_k, y_tr)
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_te_k)
            else:
                # fallback: use decision function and softmax manually
                logits = clf.decision_function(X_te_k)
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                y_proba = exp / np.sum(exp, axis=1, keepdims=True)

            # Intra-entropy: average entropy of p(y|x)
            ent_per_sample = entropy(y_proba, base=2)
            intra = np.mean(ent_per_sample)

            # Inter-entropy: entropy of marginal p(y)
            p_y = np.mean(y_proba, axis=0)
            inter = entropy(p_y, base=2)

            intra_entropies[name].append(intra)
            inter_entropies[name].append(inter)

    # Build result dict: {classifier: ((inter_m, lo, hi), (intra_m, lo, hi))}
    results = {}
    for name in classifier_constructors:
        results[name] = (ci95(inter_entropies[name]), ci95(intra_entropies[name]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing the 9 .npy files"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Proportion held out for testing"
    )
    parser.add_argument(
        "--repeats", type=int, default=50, help="Number of independent runs per method"
    )
    args = parser.parse_args()

    for method in METHODS:
        X, y = load_simulated(args.input_dir, method)
        results = run_pipeline(X, y, args.test_size, args.repeats)

        print(f"\n=== Method: {method} ===")
        for clf_name, ((inter_m, inter_lo, inter_hi), (intra_m, intra_lo, intra_hi)) in results.items():
            print(f"  -- Classifier: {clf_name} --")
            print(f"  Inter-Entropy (diversity) : {inter_m:.4f} (95% CI: {inter_lo:.4f}–{inter_hi:.4f})")
            print(f"  Intra-Entropy (confidence): {intra_m:.4f} (95% CI: {intra_lo:.4f}–{intra_hi:.4f})")


if __name__ == "__main__":
    main()
