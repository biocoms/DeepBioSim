#!/usr/bin/env python3
"""
pred_orig.py

Load GSE165512 data, perform RBF-kernel PCA to 30 components,
train an MLP for 3-class classification, and repeat this process
50 times to compute mean ± 95% CI for Accuracy, AUROC, and Log-Loss.

Example:
python pred_orig.py \
  --cd ./input/GSE165512_CD.csv \
  --control ./input/GSE165512_Control.csv \
  --uc ./input/GSE165512_UC.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from utils import ci95


def load_and_label(cd_path, control_path, uc_path):
    df_cd = pd.read_csv(cd_path, index_col=0)
    df_control = pd.read_csv(control_path, index_col=0)
    df_uc = pd.read_csv(uc_path, index_col=0)

    y_cd = np.array(["CD"] * len(df_cd))
    y_control = np.array(["Control"] * len(df_control))
    y_uc = np.array(["UC"] * len(df_uc))

    X = pd.concat([df_cd, df_control, df_uc], axis=0)
    y = np.concatenate([y_cd, y_control, y_uc])

    return X.values, y


def main():
    parser = argparse.ArgumentParser(
        description="Repeat MLP+kPCA pipeline 20× and report mean ± 95% CI."
    )
    parser.add_argument("--cd", required=True, help="GSE165512_CD.csv")
    parser.add_argument("--control", required=True, help="GSE165512_Control.csv")
    parser.add_argument("--uc", required=True, help="GSE165512_UC.csv")
    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Proportion held out for testing"
    )
    parser.add_argument(
        "--repeats", type=int, default=100, help="Number of independent runs"
    )
    args = parser.parse_args()

    # Load once
    X, y = load_and_label(args.cd, args.control, args.uc)

    # storage
    accs, aurocs, ces = [], [], []

    for run in range(args.repeats):
        # split (random each time)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, stratify=y
        )

        # standardize
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # kPCA → 40 dims
        kpca = KernelPCA(n_components=40, kernel="rbf", fit_inverse_transform=False)
        X_tr_k = kpca.fit_transform(X_tr_s)
        X_te_k = kpca.transform(X_te_s)

        # train MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            max_iter=2000,
            batch_size=16,
        )
        mlp.fit(X_tr_k, y_tr)

        # predict & score
        y_pred = mlp.predict(X_te_k)
        y_proba = mlp.predict_proba(X_te_k)

        acc = accuracy_score(y_te, y_pred)
        ce = log_loss(y_te, y_proba)
        # binarize true labels for AUROC
        classes = mlp.classes_
        y_te_bin = label_binarize(y_te, classes=classes)
        auroc = roc_auc_score(y_te_bin, y_proba, average="macro", multi_class="ovr")

        accs.append(acc)
        ces.append(ce)
        aurocs.append(auroc)

    # compute stats
    acc_m, acc_lo, acc_hi = ci95(accs)
    ce_m, ce_lo, ce_hi = ci95(ces)
    auc_m, auc_lo, auc_hi = ci95(aurocs)

    # report
    print(f"Over {args.repeats} runs:")
    print(f"  ▶ Accuracy     : {acc_m:.4f}  (95% CI: {acc_lo:.4f}–{acc_hi:.4f})")
    print(f"  ▶ Log-Loss     : {ce_m:.4f}  (95% CI: {ce_lo:.4f}–{ce_hi:.4f})")
    print(f"  ▶ AUROC (macro): {auc_m:.4f}  (95% CI: {auc_lo:.4f}–{auc_hi:.4f})")


if __name__ == "__main__":
    main()
