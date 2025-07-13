#!/usr/bin/env python3
"""
pred_sim.py

Load simulated GSE165512 samples from .npy (shape p×n so we transpose),
run RBF‐kPCA → 30 dims + MLP (100‐unit hidden, relu, adam, max_iter=2000, batch_size=16)
for 50 repeats (test_size=0.1), and report mean ± 95% CI of Accuracy, AUROC, Log‐Loss
for each generative method: VAE, IWAE, diffusion.

Usage:
python pred_sim.py --input_dir ./output --repeats 100 --test_size 0.1
"""
import argparse
import os

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
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


def run_pipeline(X, y, test_size, repeats):
    accs, aurocs, ces = [], [], []
    for _ in range(repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        kpca = KernelPCA(n_components=40, kernel="rbf", fit_inverse_transform=False)
        X_tr_k = kpca.fit_transform(X_tr_s)
        X_te_k = kpca.transform(X_te_s)

        mlp = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            max_iter=2000,
            batch_size=16,
        )
        mlp.fit(X_tr_k, y_tr)

        y_pred = mlp.predict(X_te_k)
        y_proba = mlp.predict_proba(X_te_k)

        acc = accuracy_score(y_te, y_pred)
        ce = log_loss(y_te, y_proba)
        y_te_bin = label_binarize(y_te, classes=mlp.classes_)
        auroc = roc_auc_score(y_te_bin, y_proba, average="macro", multi_class="ovr")

        accs.append(acc)
        ces.append(ce)
        aurocs.append(auroc)

    return ci95(accs), ci95(aurocs), ci95(ces)


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
        (acc_m, acc_lo, acc_hi), (auc_m, auc_lo, auc_hi), (ce_m, ce_lo, ce_hi) = (
            run_pipeline(X, y, args.test_size, args.repeats)
        )

        print(f"\n=== Method: {method} ===")
        print(f"Accuracy     : {acc_m:.4f} (95% CI: {acc_lo:.4f}–{acc_hi:.4f})")
        print(f"Log‐Loss     : {ce_m:.4f} (95% CI: {ce_lo:.4f}–{ce_hi:.4f})")
        print(f"AUROC (macro): {auc_m:.4f} (95% CI: {auc_lo:.4f}–{auc_hi:.4f})")


if __name__ == "__main__":
    main()
