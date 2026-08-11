#!/usr/bin/env python
"""
Read the topic model output and transform it to a format suitable for
tensorboard vector (embedding) projection.
"""
import pandas
import numpy as np
import os
import glob
import argparse


def main(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--i', help='Path to the input parquet file with topic model output.')
    parser.add_argument('--o_dir', help='Directory where the output tsv files will be stored.')

    args = parser.parse_args(args)

    df = pandas.read_parquet(args.i)
    # We want to have the columns in the same order as the original data, so we
    # need to sort them by their numeric suffix.
    #df = df.reindex(sorted(df.columns, key=lambda x: int(x.split('_')[1])), axis=1)

    # We want to store the input file as two tsv files
    os.makedirs(args.o_dir, exist_ok=True)

    # Drop metadata columns and save the topic distributions
    topic_cols = [col for col in df.columns if col.startswith('topic_') and col != 'topic_label']

    # Add lang label
    lang_dict = {}
    pattern = os.path.join("../../data_selection", "*.csv")
    files = sorted(glob.glob(pattern))
    for file in files:
        lang = os.path.basename(file)[:2]
        id_df = pandas.read_csv(file)
        for _, row in id_df.iterrows():
            lang_dict[row['item_id']] = lang
            
    df['lang'] = df['id'].map(lang_dict, na_action='ignore')
    df.fillna({'lang' : 'et'}, inplace=True)

    # Save
    df.drop(columns=topic_cols).to_csv(os.path.join(args.o_dir, 'metadata.tsv'), sep='\t', index=False)

    # Save the metadata columns separately
    df[topic_cols].to_csv(os.path.join(args.o_dir, 'topics.tsv'), sep='\t', index=False, header=False)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])