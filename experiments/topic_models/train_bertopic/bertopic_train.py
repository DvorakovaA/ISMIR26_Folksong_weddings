"""
BERTopic Training Pipeline — Song Lyrics
==========================================
Trains a BERTopic model on give text corpus,
saves the model locally, and produces probability-distribution outputs.

Dependencies:
    bertopic sentence-transformers umap-learn hdbscan
    pandas numpy tqdm pyarrow
"""

import os
import logging
import argparse
from pathlib import Path
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Load data

def load_csvs(input_path: str, id_col: str = "id", text_col: str = "text") -> pd.DataFrame:
    pattern = os.path.join(input_path, "*.csv") if os.path.isdir(input_path) else input_path
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found: {pattern}")
    frames = [pd.read_csv(f, dtype={id_col: str})[[id_col, text_col]] for f in files]
    df = pd.concat(frames, ignore_index=True).dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)
    log.info(f"Loaded {len(df):,} documents from {len(files)} file(s).")
    return df



# Build the BERTopic model

def build_model(min_topic_size: int = 30, n_neighbors: int = 15, n_components: int = 5, 
                nr_topics: int | str | None = "auto", embedding_model_name: str = "all-MiniLM-L6-v2",
                external_embeddings : bool = False, seed: int = 41) -> tuple[BERTopic, SentenceTransformer]:
    """
    Assemble a BERTopic model.

    Embedding model — "all-MiniLM-L6-v2":
        Fast and accurate for short texts.

    UMAP (n_neighbors=15, n_components=5):
        For finding lyrics structures

    HDBSCAN (min_cluster_size = min_topic_size):
        Controls the minimum number of songs per topic.
        prediction_data=True is required for approximate_distribution later.

    CountVectorizer:
        ngram_range=(1,2) lets bigrams like "broken heart" emerge as topic keywords.

    c-TF-IDF with BM25 weighting:
        reduce_frequent_words=True down-weights words that appear in many topics

    KeyBERTInspired representation:
        Re-ranks the c-TF-IDF keywords using embedding similarity to the topic
        centroid — produces more semantically coherent topic labels.
    """

    #~ Embedding model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    if not external_embeddings:
        embedding_model = SentenceTransformer(embedding_model_name)
    else:
        embedding_model = embedding_model_name  # path to pre-computed embeddings

    #~ UMAP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        low_memory=False,
        random_state=seed,
    )

    #~ HDBSCAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    #~ CountVectorizer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lyric_stopwords = [
        # Generic filler that floods every lyric topic
        "yeah", "oh", "ooh", "ah", "uh", "hey", "na", "la", "da", "ho"
        "gonna", "wanna", "gotta", "ain", "ain't", "chorus", "verse",
        "like", "know", "just", "got", "get", "let", "said", "say",
        "come", "go", "going", "came",
    ]
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2), # unigrams + bigrams
        min_df=5, 
        max_df=0.85,
        vocabulary=None,
    )
    # Append lyric-specific stopwords on top of sklearn's list
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    vectorizer_model = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS) + lyric_stopwords,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
    )

    #~ c-TF-IDF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True, # down-weight cross-topic common words
        bm25_weighting=True, # BM25 handles short-doc length bias
    )

    #~ Topic representation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # KeyBERTInspired: re-scores c-TF-IDF keywords by embedding similarity
    # MMR: diversifies the keyword list (avoids near-duplicate keywords)
    representation_model = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3),
    ]

    #~ Assemble BERTopic 
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        nr_topics=nr_topics,            # "auto" merges very similar topics
        calculate_probabilities=False,  # approximate_distribution instead
        verbose=True,
    )

    return topic_model, embedding_model


# Embed + fit

def embed_texts(texts: list[str], embedding_model: SentenceTransformer, batch_size: int = 128,
    cache_path: str | None = None) -> np.ndarray:
    """
    Embed all texts, with optional numpy cache so we don't re-embed on reruns.
    """
    if cache_path and os.path.exists(cache_path):
        log.info(f"Loading cached embeddings from {cache_path}")
        print(f"Embeddings shape: {np.load(cache_path).shape}")
        return np.load(cache_path)
    if isinstance(embedding_model, str):
        if os.path.exists(embedding_model):
            log.info(f"Loading cached embeddings from {embedding_model} folder.")
            # Load pre-computed embeddings from a CSV files
            paths = glob.glob(os.path.join(embedding_model, "*.csv"))
            if not paths:
                raise FileNotFoundError(f"No CSV files found in {embedding_model}")
            embeddings_list = []
            for path in paths:
                df = pd.read_csv(path, header=None)
                embeddings_list.append(df.values)
            embeddings = np.vstack(embeddings_list)
            log.info(f"Loaded {embeddings.shape[0]:,} embeddings from {len(paths)} CSV files.")

            print(f"Embeddings shape: {embeddings.shape}")
            return embeddings

    log.info(f"Embedding {len(texts):,} texts with batch_size={batch_size} ...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_path:
        np.save(cache_path, embeddings)
        log.info(f"Saved embeddings cache to {cache_path}")

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def fit_model(topic_model: BERTopic, texts: list[str], embeddings: np.ndarray,) -> tuple[list[int], np.ndarray]:
    """
    Fit BERTopic on pre-computed embeddings.
    """
    log.info("Fitting BERTopic ...")
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    n_topics = len(topic_model.get_topic_info()) - 1  # subtract outlier topic
    n_outliers = sum(1 for t in topics if t == -1)
    log.info(f"Fit complete. Topics found: {n_topics}. Outlier docs: {n_outliers:,}")
    return topics


# Compute full probability distributions (same as inference pipeline)

def infer_probabilities(model: BERTopic, texts: list[str], batch_size: int = 256,) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer topic probabilities for all texts using approximate_distribution in batches.
    """
    all_probs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="approximate_distribution"):
        batch = texts[start : start + batch_size]
        probs, _ = model.approximate_distribution(batch, calculate_tokens=False)
        all_probs.append(probs)

    probs_arr = np.vstack(all_probs).astype(np.float32)
    topics_arr = probs_arr.argmax(axis=1).astype(np.int32)
    log.info(f"Probabilities shape: {probs_arr.shape}")
    return topics_arr, probs_arr



# Save model

def save_model(model: BERTopic, output_dir: str) -> None:
    """
    Save the BERTopic model using safetensors serialization (fast and lightweight).
    """
    model_path = os.path.join(output_dir, "bertopic_model")
    model.save(model_path, serialization="safetensors", save_ctfidf=True)
    log.info(f"Model saved to {model_path}")


# Save results

def save_results(ids: pd.Series, topics: np.ndarray, probs: np.ndarray, 
                 model: BERTopic, output_dir: str) -> None:
    """
    
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    topic_info = model.get_topic_info()
    topic_id_to_name = dict(zip(topic_info["Topic"], topic_info["Name"]))
    topic_ids_sorted = sorted(tid for tid in topic_id_to_name if tid != -1)
    topic_labels = [topic_id_to_name[t] for t in topic_ids_sorted]
    ids_arr = ids.values.astype(str)

    # Probabilities
    out_path = os.path.join(output_dir, "probabilities.npz")
    np.savez_compressed(
        out_path,
        ids=ids_arr,
        topics=topics,
        probs=probs,
        topic_labels=np.array(topic_labels)
    )
    log.info(f"Saved NPZ probs to {out_path}")

    # Summary CSV 
    best_prob = probs.max(axis=1)
    summary = pd.DataFrame({
        "id": ids_arr,
        "assigned_topic": topics,
        "topic_label": [topic_id_to_name.get(tid, "unknown") for tid in
                        [topic_ids_sorted[t] if t < len(topic_ids_sorted) else -1 for t in topics]],
        "top_topic_prob": best_prob
    })
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    # Topic overview
    topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)
    log.info("Saved summary.csv and topic_info.csv")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def parse_args():
    p = argparse.ArgumentParser(description="Train BERTopic on your own lyrics corpus")
    p.add_argument("input", help="CSV file, directory, or glob pattern")
    p.add_argument("--id-col", default="item_id")
    p.add_argument("--text-col", default="translation")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--min-topic-size", type=int, default=30,
                   help="Minimum songs per topic (default: 30)")
    p.add_argument("--nr-topics", default="auto",
                   help="Target number of topics, or 'auto' to merge similar ones.")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                   help="SentenceTransformer model name.")
    p.add_argument("--embedding-cache", default="embeddings.npy",
                   help="Path to cache embeddings (avoids re-embedding on reruns).")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Embedding batch size.")
    p.add_argument("--seed", type=int, default=41)
    return p.parse_args()


def main():
    args = parse_args()

    df = load_csvs(args.input, id_col=args.id_col, text_col=args.text_col)

    texts = df[args.text_col].tolist()

    topic_model, embedding_model = build_model(
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics if args.nr_topics == "auto"
                  else int(args.nr_topics),
        embedding_model_name=args.embedding_model,
        seed=args.seed
    )

    embeddings = embed_texts(
        texts,
        embedding_model,
        batch_size=args.batch_size,
        cache_path=args.embedding_cache
    )

    fit_model(topic_model, texts, embeddings)

    topics_arr, probs_arr = infer_probabilities(
        topic_model, texts, batch_size=256
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_model(topic_model, args.output_dir)
    save_results(
        ids=df[args.id_col],
        topics=topics_arr,
        probs=probs_arr,
        model=topic_model,
        output_dir=args.output_dir
    )

    log.info("Training complete.")


if __name__ == "__main__":
    main()
