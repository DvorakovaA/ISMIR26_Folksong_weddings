"""
Statistics on the probability of assigned topics for each song in the dataset.

Input parquet file with the following columns:
  id, assigned_topic, topic_label, label, topic_0 ... topic_K
  (possibly lang column)
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def get_topic_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("topic_") and c[6:].isdigit()],
        key=lambda c: int(c[6:])
    )

def get_topic_ids_to_labels() -> dict[int, str]:
    df = pd.read_csv("bertopic_output/all_topics_info.csv")
    topic_labels = df[["Topic", "Name"]].set_index("Topic")["Name"].to_dict()
    return topic_labels

def soft_topic_stats(df: pd.DataFrame, prob_matrix: np.ndarray, topic_cols: list[str], out) -> None:
    """
    Global soft statistics:
    - Mean probability mass per topic (how dominant each topic is across the corpus)
    - Entropy per document (how spread / concentrated each song's distribution is)
    - Top-K topic coverage (how many topics needed to cover X% of probability mass per doc)
    """
    topic_lables = get_topic_ids_to_labels()

    out.write("GLOBAL SOFT TOPIC STATISTICS\n\n")

    # Mean probability mass per topic — soft equivalent of topic size
    mean_probs = prob_matrix.mean(axis=0)
    topic_ids = [int(c[6:]) for c in topic_cols]
    mean_prob_df = pd.DataFrame({
        "topic_id": topic_ids,
        "mean_prob": mean_probs,
    }).sort_values("mean_prob", ascending=False)

    out.write("Top 20 topics by mean probability mass across corpus:\n")
    out.write(f"{'topic':>10}  {'mean_prob':>10}\n")
    for _, row in mean_prob_df.head(20).iterrows():
        out.write(f"{topic_lables[(int(row.topic_id))]:>10}  {row.mean_prob:>10.5f}\n")
    out.write("\n")

    # Per-document entropy: low = concentrated on few topics, high = spread out
    eps = 1e-10
    entropy = -(prob_matrix * np.log(prob_matrix + eps)).sum(axis=1)
    out.write(f"Per-document entropy (nats):\n")
    out.write(f"  mean={entropy.mean():.4f}  std={entropy.std():.4f}  "
              f"min={entropy.min():.4f}  max={entropy.max():.4f}\n\n")

    # Top-K coverage: how many topics needed to cover 80% / 95% of probability mass
    sorted_probs = np.sort(prob_matrix, axis=1)[:, ::-1]
    cumsum = np.cumsum(sorted_probs, axis=1)
    for threshold in [0.5, 0.8, 0.95]:
        k_needed = (cumsum < threshold).sum(axis=1) + 1  # +1 for the topic that crosses threshold
        out.write(f"Topics needed to cover {threshold:.0%} of prob mass per doc: "
                  f"mean={k_needed.mean():.1f}  median={np.median(k_needed):.0f}  "
                  f"max={k_needed.max()}\n")
    out.write("\n")


def per_label_soft_stats(df: pd.DataFrame, prob_matrix: np.ndarray, topic_cols: list[str], out) -> None:
    """
    Per label:
    - Mean probability mass per topic (soft topic profile of each label)
    - Top 5 most characteristic topics per label
    - Mean entropy (are songs of this label thematically focused or diffuse?)
    - Between-label topic overlap (cosine similarity of mean topic profiles)
    """
    out.write("PER-LANG SOFT TOPIC PROFILES\n\n")
    topic_lables = get_topic_ids_to_labels()

    labels = df["lang"].values
    unique_labels = sorted(df["lang"].unique())
    topic_ids = [int(c[6:]) for c in topic_cols]

    eps = 1e-10
    entropy = -(prob_matrix * np.log(prob_matrix + eps)).sum(axis=1)

    label_profiles = {}

    for label in unique_labels:
        mask = labels == label
        sub_probs = prob_matrix[mask]
        mean_profile = sub_probs.mean(axis=1)  # shape: (n_docs_in_label,) - fix below
        mean_topic_profile = sub_probs.mean(axis=0)  # shape: (n_topics,) - mean over docs
        label_profiles[label] = mean_topic_profile

        label_entropy = entropy[mask]

        out.write(f"Label: {label}  (n={mask.sum()})\n")
        out.write(f"  Mean entropy: {label_entropy.mean():.4f} ± {label_entropy.std():.4f}\n")

        # Top 5 topics by mean probability mass
        top5_idx = np.argsort(mean_topic_profile)[::-1][:5]
        out.write("  Top 5 topics (soft):\n")
        for rank, idx in enumerate(top5_idx, 1):
            out.write(f"    {rank}. topic_{topic_lables[topic_ids[idx]]}  mean_prob={mean_topic_profile[idx]:.5f}\n")
        out.write("\n")

    # Between-label similarity matrix (cosine on mean topic profiles)
    out.write("Between-label cosine similarity of mean topic profiles:\n")
    profile_matrix = np.vstack([label_profiles[l] for l in unique_labels])
    norms = np.linalg.norm(profile_matrix, axis=1, keepdims=True) + eps
    normed = profile_matrix / norms
    sim_matrix = normed @ normed.T

    # Header
    max_label_len = max(len(l) for l in unique_labels)
    pad = max(max_label_len, 8)
    out.write(f"{'':>{pad}}  " + "  ".join(f"{l:>8}" for l in unique_labels) + "\n")
    for i, label_i in enumerate(unique_labels):
        row = "  ".join(f"{sim_matrix[i, j]:>8.3f}" for j in range(len(unique_labels)))
        out.write(f"{label_i:>{pad}}  {row}\n")
    out.write("\n")

    # Flag the most similar pairs (potential confusion candidates)
    out.write("Most similar label pairs (by topic profile):\n")
    pairs = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            pairs.append((unique_labels[i], unique_labels[j], sim_matrix[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for l1, l2, sim in pairs[:10]:
        out.write(f"  {l1} <-> {l2}:  {sim:.4f}\n")
    out.write("\n")


def dominant_topic_concentration(df: pd.DataFrame, prob_matrix: np.ndarray, out) -> None:
    """
    For each document, compare max probability vs second-max (dominance gap).
    Documents with a small gap are thematically ambiguous — interesting for error analysis.
    """
    out.write("TOPIC DOMINANCE ANALYSIS\n\n")
    topic_labels = get_topic_ids_to_labels()

    sorted_probs = np.sort(prob_matrix, axis=1)[:, ::-1]
    max_prob = sorted_probs[:, 0]
    second_prob = sorted_probs[:, 1]
    second_index = np.argsort(prob_matrix, axis=1)[:, -2]  # index of second-max topic
    second_topic_label = [topic_labels[int(c[6:])] for c in df.columns[2:][second_index]]
    dominance_gap = max_prob - second_prob

    out.write(f"Max topic probability per doc:    mean={max_prob.mean():.4f}  std={max_prob.std():.4f}\n")
    out.write(f"2nd topic probability per doc:    mean={second_prob.mean():.4f}  std={second_prob.std():.4f}\n")
    out.write(f"Dominance gap (max - 2nd):        mean={dominance_gap.mean():.4f}  std={dominance_gap.std():.4f}\n\n")

    # Most ambiguous documents (smallest dominance gap)
    df_dom = df[["id", "lang", "topic_label"]].copy()
    df_dom["max_prob"] = max_prob
    df_dom["second_prob"] = second_prob
    df_dom["second_topic_label"] = second_topic_label
    df_dom["dominance_gap"] = dominance_gap
    most_ambiguous = df_dom.nsmallest(20, "dominance_gap")

    out.write("20 most thematically ambiguous documents (smallest dominance gap):\n")
    out.write(f"{'id':>20}  {'label':>20}  {'top_topic':>30} {'second_topic':>30} {'gap':>8}\n")
    for _, row in most_ambiguous.iterrows():
        out.write(f"{str(row.id):>20}  {str(row.lang):>20}  "
                  f"{str(row.topic_label):>30} | {str(row.second_topic_label):>30} {row.dominance_gap:>8.4f}\n")
    out.write("\n")

    # Per-label mean dominance gap
    out.write("Mean dominance gap per lang:\n")
    df_dom["lang"] = df["lang"].values
    per_label_gap = df_dom.groupby("lang")["dominance_gap"].agg(["mean", "std"]).sort_values("mean")
    for label, row in per_label_gap.iterrows():
        out.write(f"  {label:<30}  mean={row['mean']:.4f}  std={row['std']:.4f}\n")
    out.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_parq", type=str, help="Path to the input PARQUET file.")
    parser.add_argument("output_txt", type=str, help="Path to the output TXT file.")
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parq)
    topic_cols = get_topic_cols(df)
    log = lambda msg: print(msg)

    log(f"Loaded {len(df):,} documents, {len(topic_cols)} topic dimensions.")

    prob_matrix = df[topic_cols].to_numpy(dtype=np.float32)

    with open(args.output_txt, "w", encoding="utf-8") as out:
        out.write(f"Input: {args.input_parq}\n")
        out.write(f"Documents: {len(df):,}   Topics: {len(topic_cols)}\n\n")

        soft_topic_stats(df, prob_matrix, topic_cols, out)
        per_label_soft_stats(df, prob_matrix, topic_cols, out)
        dominant_topic_concentration(df, prob_matrix, out)

    log(f"Done. Results written to {args.output_txt}")


if __name__ == "__main__":
    main()