"""
Compare two hard assignments from BERTopic training
"""
import pandas as pd
import argparse
import glob
import os

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score


def parse_args():
    parser = argparse.ArgumentParser(description="Compare typology vs BERTopic labels")
    parser.add_argument("--i1", required=True, help="CSV file with BERTopic labels (columns: id, topic_label)")
    parser.add_argument("--i2", required=True, help="Second CSV file with BERTopic labels (columns: id, topic_label)")
    parser.add_argument("--output", default="stability_comparison.txt", help="Output TXT file for comparison results")
    return parser.parse_args()


def main():
    args = parse_args()

    df1 = pd.read_csv(args.i1)
    df2 = pd.read_csv(args.i2)

    set1 = list(df1['assigned_topic'].values)
    set2 = list(df2['assigned_topic'].values)
    with open(args.output, 'w') as f:
        f.write(f"Comparing {args.i1} and {args.i2}\n\n")
        f.write(f"Adjusted Rand Index: {adjusted_rand_score(set1, set2)}\n")
        f.write(f"Normalized Mutual Information: {normalized_mutual_info_score(set1, set2)}\n")
        f.write(f"Jaccard Score (average weighted): {jaccard_score(set1, set2, average='weighted')}\n")



if __name__ == "__main__":
    main()