"""
BERTopic Inference
Loads pretrained MaartenGr/BERTopic_Wikipedia from huggingface,
processes batches of (ID, text) pairs from CSV files, computes
per-document topic probability distributions, and stores results.

Dependencies:
    pip install bertopic datasets pandas numpy tqdm

Expected call:
    python bertopic_pipeline.py ../../translation/translated/
"""

import os
import glob
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from umap  import UMAP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_csvs(input_path: str, id_col: str = "item_id", text_col: str = "translation") -> pd.DataFrame:
    """
    Load one or more CSV files and return a merged DataFrame.

    Args:
        input_path: Path to a single CSV file or directory to be used by glob.
        id_col: Name of the document-ID column.
        text_col: Name of the text column.

    Returns:
        DataFrame with at least [id_col, text_col].
    """
    if os.path.isdir(input_path):
        pattern = os.path.join(input_path, "*.csv")
    else:
        pattern = input_path

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found matching: {pattern}")

    log.info(f"Found {len(files)} CSV file(s). Loading...")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={id_col: str})
        if id_col not in df.columns or text_col not in df.columns:
            raise ValueError(
                f"File '{f}' must contain columns '{id_col}' and '{text_col}'. "
                f"Found: {list(df.columns)}"
            )
        frames.append(df[[id_col, text_col]])

    merged = pd.concat(frames, ignore_index=True)
    log.info(f"Total documents loaded: {len(merged):,}")

    # Drop rows with missing text
    before = len(merged)
    merged = merged.dropna(subset=[text_col])
    merged[text_col] = merged[text_col].astype(str).str.strip()
    merged = merged[merged[text_col] != ""]
    if len(merged) < before:
        log.warning(f"Dropped {before - len(merged)} rows with empty text.")

    return merged.reset_index(drop=True)


def load_model(model_name: str = "MaartenGr/BERTopic_Wikipedia") -> BERTopic:
    """
    Download (or load from cache) a pretrained BERTopic model from HuggingFace.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded BERTopic instance.
    """
    log.info(f"Loading BERTopic model: {model_name}")
    model = BERTopic.load(model_name)

    # Reproducibility: set a fixed UMAP model for dimensionality reduction
    umap_model = UMAP(n_neighbors=50, n_components=5, metric='cosine', random_state=42)
    model.umap_model = umap_model

    log.info(f"Model loaded. Number of topics: {len(model.get_topic_info())}")
    return model


def infer_probabilities(model: BERTopic, texts: list[str], batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Run BERTopic approximate_distribution over texts in mini-batches.

    The approximate_distribution returns soft topic probabilities for every
    document without requiring a full re-fit.

    Args:
        model: Loaded BERTopic model.
        texts: List of document strings.
        batch_size: Number of documents per batch (tune to your RAM/GPU).

    Returns:
        topics: 1-D array of shape (N,) – most probable topic per doc.
        probabilities: 2-D array of shape (N, num_topics) – full distribution.
    """
    all_probs_distr = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Inference batches"):
        batch = texts[start : start + batch_size]

        # approximate_distribution gives a full soft distribution
        topic_prob_distr, topic_token_distr = model.approximate_distribution(batch, calculate_tokens=True)
        
        all_probs_distr.append(topic_prob_distr)

    probs_arr = np.vstack(all_probs_distr).astype(np.float32)   # (N, T)
    topics_arr = probs_arr.argmax(axis=1).astype(np.int32)  # (N,)

    log.info(f"Inference done. topics shape: {topics_arr.shape}, probabilities shape: {probs_arr.shape}")
    return topics_arr, probs_arr


def save_results(ids: pd.Series, topics: np.ndarray, probs: np.ndarray, model: BERTopic, output_dir: str = "output"):
    """
    Persist the probability distributions and simpler summary CSV.

    Output files:
    <output_dir>/probabilities.parquet
        Columns: id, assigned_topic, topic_label, topic_0, topic_1, ..., topic_T

    <output_dir>/summary.csv
        id | assigned_topic | topic_label | top_topic_prob
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build topic labels for columns
    topic_info = model.get_topic_info()
    # topic_info index is arbitrary; the "Topic" column holds the integer IDs
    # Columns present: Topic, Count, Name
    topic_id_to_name: dict[int, str] = dict(
        zip(topic_info["Topic"], topic_info["Name"])
    )

    # Derive the column order from topic_info.
    topic_ids_sorted = sorted(topic_id_to_name.keys())  # includes -1 
    topic_ids_sorted.remove(-1)  # Remove the outlier topic if present
    topic_labels = [topic_id_to_name[t] for t in topic_ids_sorted]
    ids_arr = ids.values.astype(str)

    # Save probs
    topic_cols = {f"topic_{tid}": probs[:, i] for i, tid in enumerate(topic_ids_sorted)}
    df_out = pd.DataFrame({"id": ids_arr, "assigned_topic": topics})
    df_out["topic_label"] = df_out["assigned_topic"].map(topic_id_to_name)
    df_out = pd.concat([df_out, pd.DataFrame(topic_cols)], axis=1)

    out_path = os.path.join(output_dir, "probabilities.parquet")
    df_out.to_parquet(out_path, index=False)
    log.info(f"Saved Parquet: {out_path}")

    # Save summary
    best_prob = probs.max(axis=1)
    summary = pd.DataFrame(
        {
            "id": ids_arr,
            "assigned_topic": topics,
            "topic_label": [topic_id_to_name.get(t, "unknown") for t in topics],
            "top_topic_prob": best_prob,
        }
    )
    summary_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved summary CSV: {summary_path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BERTopic inference pipeline — Wikipedia pretrained model"
    )
    p.add_argument(
        "input",
        help="Path to a CSV file or a directory of CSVs."
    )
    p.add_argument(
        "--id-col", default="item_id",
        help="Name of the document-ID column (default: 'id')."
    )
    p.add_argument(
        "--text-col", default="translation",
        help="Name of the text column (default: 'translation')."
    )
    p.add_argument(
        "--model", default="MaartenGr/BERTopic_Wikipedia",
        help="HuggingFace model ID for the pretrained BERTopic model."
    )
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="Documents per inference batch (default: 256)."
    )
    p.add_argument(
        "--output-dir", default="output",
        help="Directory for result files (default: 'output')."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load data
    df = load_csvs(args.input, id_col=args.id_col, text_col=args.text_col)

    # 2. Load model
    model = load_model(args.model)

    # 3. Infer
    topics, probs = infer_probabilities(
        model,
        texts=df[args.text_col].tolist(),
        batch_size=args.batch_size,
    )

    # 4. Save
    save_results(
        ids=df[args.id_col],
        topics=topics,
        probs=probs,
        model=model,
        output_dir=args.output_dir,
    )

    log.info("Pipeline complete!")


if __name__ == "__main__":
    main()