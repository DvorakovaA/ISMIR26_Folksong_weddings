"""
PCA Pipeline (Dimensionality Reduction for BERTopic probabilities)

Reads a BERTopic probability distribution parquet file, applies PCA
retaining a target explained variance (default 95 %), and writes a new
parquet file with the reduced features.

Dependencies:
    pip install pandas numpy scikit-learn pyarrow tqdm

Expected call:
    python pca_pipeline.py bertopic_output/probabilities.parquet --variance 0.95 --whiten
"""

import os
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_parquet(path: str):
    """
    Load the probability parquet and split into metadata and feature columns.

    Args:
        path: Path to the input parquet file containing BERTopic probabilities.
    Returns:
        DataFrame with non-topic columns (id, assigned_topic, topic_label, etc.)
        DataFrame with only topic_* probability columns
        ordered list of topic column names
    """
    log.info(f"Loading parquet: {path}")
    df = pd.read_parquet(path)
    log.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")

    topic_cols = [
        c for c in df.columns
        if c.startswith("topic_") and c[6:].lstrip("-").isdigit()
    ]
    meta_cols = [c for c in df.columns if c not in topic_cols]

    log.info(f"Metadata columns : {meta_cols}")
    log.info(f"Number of topic columns: {len(topic_cols)}")

    return df[meta_cols].copy(), df[topic_cols].copy(), topic_cols


def fit_pca(probs: np.ndarray, variance_threshold: float = 0.95, whiten: bool = False, random_seed: int = 42):
    """
    Fit PCA with `variance_threshold` of total explained variance.

    The probability matrix is L1-normalised per row before PCA so that
    the decomposition operates on a proper probability simplex rather than
    raw floating-point scale differences.

    Args:
        probs: (N, T) float32 probability matrix.
        variance_threshold: Fraction of variance to retain (0 < v ≤ 1).
        whiten: Normalise each PC to unit variance (useful if downstream classifier is distance-based, e.g. SVM).
        random_seed

    Returns:
        fitted sklearn PCA object.
        reduced matrix
    """
    log.info(f"Input shape: {probs.shape}.")
    log.info(f"Fitting PCA (target variance={variance_threshold}) ...")

    # L1-normalise rows so each document's probabilities sum to 1
    # (approximate_distribution output can have very small deviations from 1.0)
    probs_norm = normalize(probs, norm="l1", axis=1)

    pca = PCA(
        n_components=variance_threshold,   
        svd_solver="full",                 
        whiten=whiten,
        random_state=random_seed,
    )
    reduced = pca.fit_transform(probs_norm).astype(np.float32)

    log.info(f"PCA fit complete.")
    log.info(f"  Final number of components: {pca.n_components_}")

    return pca, reduced



def save_reduced_parquet(meta: pd.DataFrame, reduced: np.ndarray, output_path: str):
    """
    Write the reduced feature matrix alongside original metadata columns.

    Output columns: <meta_cols>, pc_0, pc_1, ... , pc_K
    """
    n_components = reduced.shape[1]
    pc_cols = {f"pc_{i}": reduced[:, i] for i in range(n_components)}

    out = pd.concat([meta.reset_index(drop=True), pd.DataFrame(pc_cols)], axis=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    log.info(f"Saved reduced parquet ({len(out):,} rows × {len(out.columns)} cols) into {output_path}")



def save_variance_report(pca: PCA, output_dir: str):
    """
    Write a CSV showing cumulative explained variance per component.
    """
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    report = pd.DataFrame({
        "component": np.arange(1, len(cumvar) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumvar,
    })
    path = os.path.join(output_dir, "pca_variance_report.csv")
    report.to_csv(path, index=False)
    log.info(f"Saved variance report: {path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PCA reduction of BERTopic probability distributions."
    )
    p.add_argument("input", help="Path to input probabilities.parquet")
    p.add_argument(
        "--output", default=None,
        help="Output parquet path (default: same dir as input, suffix _pcaVAR.parquet).",
    )
    p.add_argument(
        "--variance", type=float, default=0.95,
        help="Target explained variance fraction (default: 0.95).",
    )
    p.add_argument(
        "--whiten", action="store_true",
        help="Whiten PCA components.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve output path
    input_path = Path(args.input)
    if args.output is None:
        suffix = f"_pca{int(args.variance * 100)}.parquet"
        output_path = str(input_path.with_name(input_path.stem + suffix))
    else:
        output_path = args.output

    output_dir = str(Path(output_path).parent)

    # Run pipeline
    meta, probs_df, topic_cols = load_parquet(args.input)
    probs = probs_df.to_numpy(dtype=np.float32)

    pca, reduced = fit_pca(
        probs,
        variance_threshold=args.variance,
        whiten=args.whiten,
        random_seed=args.seed,
    )

    save_reduced_parquet(meta, reduced, output_path)
    save_variance_report(pca, output_dir)

    log.info("PCA reduction complete.")
    log.info(f"  {len(topic_cols)} topic features: {pca.n_components_} PCs")
    log.info(f"  Output: {output_path}")


if __name__ == "__main__":
    main()