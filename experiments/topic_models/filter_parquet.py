#!/usr/bin/env python
"""
Script to filter the parquet files based on given list of item_ids.
"""
import pandas as pd
import argparse
import os

def filter_parquet(input_path: str, output_path: str, item_ids: list[int]):
    # Read the parquet file into a DataFrame
    df = pd.read_parquet(input_path)

    print(len(item_ids))
    print(len(set(item_ids)))
    
    # Filter the DataFrame to keep only rows with item_id in the given list
    filtered_df = df[df['id'].isin(item_ids)]
    print(f"Filtered parquet has {len(filtered_df)} rows.")
    print(filtered_df['label'].value_counts())

    # Save the filtered DataFrame back to a new parquet file
    filtered_df.to_parquet(output_path, index=False)
    print(f"Filtered data saved to {output_path}")

def build_argument_parser():
    parser = argparse.ArgumentParser(description='Filter parquet files based on a list of item_ids.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='Path to the input parquet file.')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Path where the filtered parquet file will be saved.')
    parser.add_argument('-ids', '--item_ids', type=str, required=True,
                        help='Path to csv with item_ids to keep in the filtered parquet file.')
    parser.add_argument('-c', '--column', type=str, default='item_id',
                        help='Name of the column in the csv file that contains the item_ids (default: item_id).')
    return parser


def main():    
    parser = build_argument_parser()
    args = parser.parse_args()

    # Read the item_ids from the provided csv file
    print(f"Reading item_ids from {args.item_ids}...")
    item_ids_df = pd.read_csv(args.item_ids)
    item_ids = item_ids_df[args.column].tolist()

    filter_parquet(args.input_path, args.output_path, item_ids)

    print("Filtering completed.")



if __name__ == "__main__":
    main()