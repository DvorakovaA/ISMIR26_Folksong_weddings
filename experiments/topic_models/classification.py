#!/usr/bin/env python
"""
Classification Pipeline - Stratified K-Fold RBF & Linear SVM

Reads the parquet from BerTopic pipeline,
runs stratified 5-fold cross-validation with held-out test set (20 %),
evaluates with F-score, and produces confusion matrices + per-label
difficulty analysis.

Expected input parquet columns:
    id, assigned_topic, topic_label, label, topic_0 ... topic_K


Dependencies:
    pip install pandas numpy scikit-learn pyarrow matplotlib seaborn tqdm

Expected call:
    python classification.py bertopic_output/probabilities.parquet --output-dir output/classification  --save-models
"""

import os
import argparse
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


#~ Load input data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_data(features_path: str, id_col: str = "id", label_col: str = "label"):
    """
    Load features and join labels by ID.

    Returns:
    X: (N, K) feature matrix
    y: (N,) label array
    ids: (N,) document IDs
    class_names: list of original label strings (index = encoded int)
    """
    log.info(f"Loading features : {features_path}")
    df = pd.read_parquet(features_path)

    topic_cols = sorted([c for c in df.columns if c.startswith("topic_") and c[6:].isdigit()],
                     key=lambda c: int(c[6:]))
    log.info(f"Features: {len(topic_cols)} PCs")
    log.info(f"Documents: {len(df):,}")

    X = df[topic_cols].to_numpy(dtype=np.float32)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str)).astype(np.int32)
    class_names = list(le.classes_)

    log.info(f"Classes ({len(class_names)}): {class_names}")
    counts = pd.Series(y).value_counts().sort_index()
    for i, cnt in counts.items():
        log.info(f"  [{i:3d}] {class_names[i]:<30s} n={cnt}")

    return X, y, df[id_col].values.astype(str), class_names


#~ Prepare pipelines ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_rbf_pipeline(C: float = 1.0, gamma: str = "scale"):
    """
    StandardScaler : RBF SVC.
    Scaling is critical for SVM - PCA output is not unit-variance.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight="balanced",   # handles class imbalance
            decision_function_shape="ovr",
            random_state=42,
        )),
    ])


def make_linear_pipeline(C: float = 0.1, max_iter: int = 2500):
    """
    StandardScaler - LinearSVC (wrapped in CalibratedClassifierCV for
    probability estimates, consistent with RBF interface).
    """
    base = LinearSVC(
        C=C,
        class_weight="balanced", # handles class imbalance
        max_iter=max_iter,
        random_state=42,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(base, cv=3, method="sigmoid")),
    ])


#~ Cross-validation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def subsample_class(X: np.ndarray, y: np.ndarray, target_class: int,
                    max_samples: int, random_state: int = 42):
    """
    Randomly subsample one class to at most max_samples within a fold's
    training split. All other classes are left untouched.
    """
    rng = np.random.default_rng(random_state)
    target_idx = np.where(y == target_class)[0]

    if len(target_idx) <= max_samples:
        return X, y   # already within limit, nothing to do

    keep = rng.choice(target_idx, size=max_samples, replace=False)
    other_idx = np.where(y != target_class)[0]
    selected = np.sort(np.concatenate([other_idx, keep]))

    return X[selected], y[selected]


def run_cross_valid(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, 
                   class_names : list[str], n_splits: int = 5, random_seed: int = 42,
                   subsample : bool = False, pca_variance : float = 0.95):
    """
    Stratified k-fold cross validation.
    Returns per-fold results.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_results = []

    all_y_true = []
    all_y_pred = []

    if subsample:
        # Determine which classes to subsample based on overall class distribution.
        # Sort classes by frequency and look for the largest gap to find a cutoff.
        counts = pd.Series(y).value_counts().sort_values(ascending=False)
        diffs = counts.values[:-1] - counts.values[1:]
        gap_idx = diffs.argmax()
        subsample_class_ids = [i for i in counts.index[:gap_idx + 1]]
        subsample_max = counts.iloc[gap_idx + 1]
        log.info(f"Subsampling enabled. Will subsample classes {', '.join(class_names[i] for i in subsample_class_ids)} to max {subsample_max} samples in each fold's training split.")

    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        log.info(f"Fold {fold}/{n_splits} ...")
        X_tr, y_tr = X[tr], y[tr]

        if subsample:
            for subsample_class_id in subsample_class_ids:
                log.info(f"  Subsampling class {class_names[subsample_class_id]} to max {subsample_max} samples in training split")
                X_tr, y_tr = subsample_class(X_tr, y_tr, subsample_class_id, subsample_max, random_state=random_seed + fold)

        # PCA: fit on training split only to avoid data leakage, then transform both train and val
        fold_pca = PCA(n_components=pca_variance, svd_solver="full", whiten=True, random_state=random_seed)
        X_tr  = fold_pca.fit_transform(X_tr)
        X_val = fold_pca.transform(X[val])
        log.info(f"  Fold {fold} PCA: {X_tr.shape[1]} -> {fold_pca.n_components_} components ")

        pipeline.fit(X_tr, y_tr)

        log.info(f"  Evaluating fold {fold} ...")
        y_pred = pipeline.predict(X_val)

        f1_macro = f1_score(y[val], y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y[val], y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y[val], y_pred)
        report = classification_report(
            y[val], y_pred,
            target_names=class_names,
            labels=range(len(class_names)),
            output_dict=True,
            zero_division=0
        )
        report_str = classification_report(
            y[val], y_pred,
            target_names=class_names,
            zero_division=0,
            labels=range(len(class_names))
        )

        fold_results.append({"fold": fold, "f1_macro": f1_macro, "f1_weighted": f1_weighted, 
                             "confusion_matrix": cm, "report_dict": report, "report_str": report_str,
                             "pca_n_components": fold_pca.n_components_, "y_val": y[val], "y_pred": y_pred})

        all_y_true.append(y[val])
        all_y_pred.append(y_pred)

        log.info(f"  Fold {fold}/{n_splits}  macro-F1={f1_macro:.4f}  weighted-F1={f1_weighted:.4f}")
    
    agg_y_true = np.concatenate(all_y_true)
    agg_y_pred = np.concatenate(all_y_pred)
    
    df_cv = pd.DataFrame(fold_results)
    return {
        "fold_results": df_cv,
        "mean_f1_macro": df_cv["f1_macro"].mean(),
        "std_f1_macro": df_cv["f1_macro"].std(),
        "mean_f1_weighted": df_cv["f1_weighted"].mean(),
        "std_f1_weighted": df_cv["f1_weighted"].std(),
        "confusion_matrices": df_cv["confusion_matrix"].tolist(), 
        "report_dicts": df_cv["report_dict"].tolist(),
        "report_strs": df_cv["report_str"].tolist(),
        "pca_n_components": df_cv["pca_n_components"].tolist(),
        "agg_y_true": agg_y_true,
        "agg_y_pred": agg_y_pred,
    }



def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str,
                          output_path: str, normalise: bool = True, figsize_per_class: float = 0.55):
    """
    Plot a (optionally row-normalised) confusion matrix heatmap.
    Scales figure size automatically with number of classes.
    """
    n = len(class_names)
    figsize = max(8, n * figsize_per_class)

    if normalise:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)
        fmt, vmax = ".2f", 1.0
        cbar_label = "Recall (row-normalised)"
    else:
        cm_plot, fmt, vmax = cm, "d", None
        cbar_label = "Count"

    # Keep ticks readable when class count is large.
    max_tick_labels = 40
    tick_step = max(1, int(np.ceil(n / max_tick_labels)))
    tick_idx = np.arange(0, n, tick_step)
    x_rotation = 90 if n > 35 else 45
    tick_fontsize = max(6, 10 - n // 12)

    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
    sns.heatmap(
        cm_plot,
        annot=n <= 30,          # skip per-cell numbers if too many classes
        fmt=fmt,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        vmin=0, vmax=vmax,
        linewidths=0.3 if n <= 30 else 0,
        ax=ax,
        cbar_kws={"label": cbar_label}
    )
    # Heatmap cell centers are at i + 0.5, so set ticks explicitly there.
    ax.set_xticks(tick_idx + 0.5)
    ax.set_yticks(tick_idx + 0.5)
    ax.set_xticklabels(
        [class_names[i] for i in tick_idx],
        rotation=x_rotation,
        ha="center" if x_rotation == 90 else "right",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(
        [class_names[i] for i in tick_idx],
        rotation=0,
        fontsize=tick_fontsize,
    )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.tick_params(axis="x", pad=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved confusion matrix : {output_path}")


def analyse_label_difficulty(cm: np.ndarray, class_names: list[str], report_dict: dict, model_name: str):
    """
    Derive per-label diagnostics that answer:
        "Which labels are easy / hard / confused?"
    """
    rows = []
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    for i, name in enumerate(class_names):
        r = report_dict.get(name, {})
        support = int(r.get("support", 0))
        precision = r.get("precision", 0.0)
        recall = r.get("recall", 0.0)
        f1 = r.get("f1-score", 0.0)
        confusion_rate = 1.0 - recall

        # Most common confusion (off-diagonal column with highest value)
        row_norm = cm_norm[i].copy()
        row_norm[i] = 0 # mask diagonal
        confused_idx = int(row_norm.argmax())
        top_confused_pct = float(row_norm[confused_idx])
        top_confused_with = class_names[confused_idx] if top_confused_pct > 0 else "—"

        rows.append({
            "model": model_name,
            "label": name,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion_rate": round(confusion_rate, 4),
            "top_confused_with": top_confused_with,
            "top_confused_pct": round(top_confused_pct, 4),
        })

    df = pd.DataFrame(rows).sort_values("f1", ascending=True)
    return df


def plot_label_difficulty(difficulty_df: pd.DataFrame, title: str, output_path: str):
    """
    Horizontal bar chart: F1 per label, coloured by confusion_rate.
    """
    df = difficulty_df.sort_values("f1", ascending=True).copy()
    n = len(df)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.35)))

    cmap = plt.get_cmap("RdYlGn")
    colors = [cmap(f) for f in df["f1"]]

    bars = ax.barh(df["label"], df["f1"], color=colors, edgecolor="white", height=0.7)

    # Annotate bars with F1 value and top confusion
    for bar, (_, row) in zip(bars, df.iterrows()):
        w = bar.get_width()
        label = f"{w:.2f}"
        if row["top_confused_with"] != "—":
            label += f"  ← confused with {row['top_confused_with']} ({row['top_confused_pct']:.0%})"
        ax.text(
            min(w + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=7.5,
        )

    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("F1 score")
    ax.set_title(title, fontsize=12, pad=10)
    ax.axvline(df["f1"].mean(), color="blue", linestyle="--", linewidth=1,
               label=f"mean F1 = {df['f1'].mean():.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved difficulty chart : {output_path}")


def save_model(pipeline: Pipeline, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    log.info(f"Saved model: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified k-fold RBF + Linear SVM classification pipeline."
    )
    p.add_argument("features", help="Path to parquet.")
    p.add_argument("--id-col",    default="id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--output-dir", default="output_classification")
    p.add_argument("--test-size",  type=float, default=0.2,
                   help="Held-out test fraction (default: 0.2).")
    p.add_argument("--n-folds",    type=int,   default=5,
                   help="Number of CV folds (default: 5).")
    p.add_argument("--pca-variance", type=float, default=0.95,
                   help="PCA variance to retain (default: 0.95).")
    p.add_argument("--subsample", default=False, action="store_true",
                   help="Whether to subsample classes within each fold's training split.")
    p.add_argument("--rbf-C",      type=float, default=1.0)
    p.add_argument("--rbf-gamma",  default="scale")
    p.add_argument("--linear-C",   type=float, default=0.1)
    p.add_argument("--save-models", action="store_true",
                   help="Persist models as pickle files.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    X, y, ids, class_names = load_data(args.features, id_col=args.id_col, label_col=args.label_col)

    # Pipelines
    models = {
        "RBF_SVM": make_rbf_pipeline(C=args.rbf_C, gamma=args.rbf_gamma),
        "Linear_SVM": make_linear_pipeline(C=args.linear_C),
    }

    all_difficulty = []
    summary_rows = []

    for model_name, pipeline in models.items():
        log.info(f"\n{'='*60}")
        log.info(f"  {model_name}")
        log.info(f"{'='*60}")

        # Cross validation runs on the full dataset (train + test) to maximise training data
        log.info(f"Running {args.n_folds}-fold CV ...")
        
        cv_results = run_cross_valid(pipeline, X, y, n_splits=args.n_folds, 
                                     class_names=class_names, random_seed=args.seed,
                                     subsample=args.subsample)

        log.info(f"CV  macro-F1 = {cv_results['mean_f1_macro']:.4f} ~ {cv_results['std_f1_macro']:.4f}")


        # Save text report
        report_path = out / f"{model_name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Model: {model_name}\n\n")
            for i, report_str in enumerate(cv_results["report_strs"]):
                f.write(f"Fold {i+1}:\n{report_str}\n\n")

        # Confusion matrices (raw counts + normalised)
        for i, cm in enumerate(cv_results["confusion_matrices"]):
            plot_confusion_matrix(cm, 
                                  class_names,
                                  title=f"{model_name} — CV Fold Confusion Matrix (row-normalised recall)",
                                  output_path=str(out / f"{model_name}_cv_fold_confusion_matrix_normalised_fold_{i}.png"),
                                  normalise=True
            )
            plot_confusion_matrix(cm, 
                                  class_names,
                                  title=f"{model_name} — CV Fold Confusion Matrix (counts)",
                                  output_path=str(out / f"{model_name}_cv_fold_confusion_matrix_counts_fold_{i}.png"),
                                  normalise=False
            )

        # One confusion matrix aggregating all folds (by concatenating predictions)
        plot_confusion_matrix(confusion_matrix(cv_results["agg_y_true"], cv_results["agg_y_pred"]), 
                              class_names,
                              title=f"{model_name} — Aggregate CV Confusion Matrix (row-normalised recall)",
                              output_path=str(out / f"{model_name}_cv_aggregate_confusion_matrix_normalised.png"),
                              normalise=True
        )

        # Per-label difficulty
        for i,  report_dict in enumerate(cv_results["report_dicts"]):
            difficulty = analyse_label_difficulty(
                cm, class_names, report_dict, model_name,
            )
            difficulty.to_csv(out / f"{model_name}_label_difficulty.csv", index=False)
            plot_label_difficulty(
                difficulty,
                title=f"{model_name} — Per-label F1 & Confusion (test set)",
                output_path=str(out / f"{model_name}_label_difficulty.png"),
            )
            all_difficulty.append(difficulty)

        summary_rows.append({
            "model": model_name,
            "cv_macro_f1_mean": round(cv_results["mean_f1_macro"], 4),
            "cv_macro_f1_std":  round(cv_results["std_f1_macro"], 4),
            "cv_weighted_f1_mean": round(cv_results["mean_f1_weighted"], 4),
            "cv_n_pca_components_mean": round(np.mean(cv_results["pca_n_components"]), 1),
        })

        if args.save_models:
            model_path = out / f"{model_name}_model.pkl"
            save_model(pipeline, model_path)


    # Compare models
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "complete_summary.csv", index=False)
    log.info(f"Summary:\n{summary_df.to_string(index=False)}")
 
    # Combined difficulty comparison
    if len(all_difficulty) == 2:
        combined = pd.merge(
            all_difficulty[0][["label", "f1", "top_confused_with", "top_confused_pct"]],
            all_difficulty[1][["label", "f1", "top_confused_with", "top_confused_pct"]],
            on="label", suffixes=("_RBF", "_Linear"),
        ).sort_values("f1_RBF")
        combined.to_csv(out / "label_difficulty_comparison.csv", index=False)
        log.info(f"Saved cross-model difficulty comparison : {out / 'label_difficulty_comparison.csv'}")
    
    
    log.info(f"All outputs written to: {out}")
    log.info("Classification pipeline complete!")


if __name__ == "__main__":
    main()