"""
Script to compare typology vs BERTopic labels for the same dataset.
For each BERTopic label, it returns the most common typology labels and their counts.
"""
import pandas as pd
import argparse
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Compare typology vs BERTopic labels")
    parser.add_argument("--bertopic_input", required=True, help="CSV file with BERTopic labels (columns: id, topic_label)")
    parser.add_argument("--typology_input", required=True, help="Path to CSV files with typology labels (columns: item_id, label)")
    parser.add_argument("--output", default="label_comparison.txt", help="Output TXT file for comparison results")
    return parser.parse_args()



def main():
    args = parse_args()
    # Read BERTopic input file - get list of unique topic labels with associated ids
    bertopic_df = pd.read_csv(args.bertopic_input)
    bertopic_lables = bertopic_df.groupby('topic_label')['id'].apply(list).to_dict()
    
    # Collect typology labels from multiple CSV files in given directory
    typology_labels = {}
    files = sorted(glob.glob(os.path.join(args.typology_input, "*.csv")))
    for file in files:
        typology_df = pd.read_csv(file)
        for _, row in typology_df.iterrows():
            item_id = row['item_id']
            label = str(row['label']) + '_' + str(os.path.basename(file)[:2])
            typology_labels[item_id] = label
    
    with open(args.output, 'w') as f:
        for blabel in bertopic_lables:
            ids = bertopic_lables[blabel]
            typology_counts = {}
            lang_counts = {'cs' : 0, 'et' : 0, 'uk' : 0, 'ko' : 0, 'nl' : 0}
            for id in ids:
                if id in typology_labels:
                    tlabel = typology_labels[id]
                    if tlabel not in typology_counts:
                        typology_counts[tlabel] = 0
                    typology_counts[tlabel] += 1
                    lang_counts[tlabel[-2:]] += 1

            # Sort typology counts by frequency
            sorted_typology_counts = sorted(typology_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"BERTopic label: {blabel} ({len(ids)} songs)" , file=f)
            # Print sorted language counts
            print(f"  Language counts: {sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)}", file=f)
            for tlabel, count in sorted_typology_counts:
                print(f"  Typology label: {tlabel}, Count: {count}", file=f)
            print(file=f)


if __name__ == "__main__":
    main()