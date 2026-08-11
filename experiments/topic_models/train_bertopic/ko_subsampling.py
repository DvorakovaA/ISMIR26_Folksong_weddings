"""

"""
import pandas as pd
import random


if "__main__" == __name__:
    original_df = pd.read_csv("../../../translation/translated/ko_translated.csv")
    print(f"Loaded {len(original_df)} rows from ko_translated.csv")

    # Randomly subsample biggest class to the size of the second biggest class
    class_counts = original_df['label'].value_counts()
    if len(class_counts) > 1:
        biggest_class = class_counts.idxmax()
        print(f"Biggest class: {biggest_class} with {class_counts[biggest_class]} samples")
        second_biggest_class_count = class_counts.nlargest(2).iloc[1]
        print(f"Second biggest class count: {second_biggest_class_count}")
        # subsample the biggest class
        biggest_class_df = original_df[original_df['label'] == biggest_class]
        subsampled_biggest_class_df = biggest_class_df.sample(n=second_biggest_class_count, random_state=41)

    # Concatenate the subsampled biggest class with the rest of the data
    rest_of_data_df = original_df[original_df['label'] != biggest_class]

    # Concatenate
    df = pd.concat([subsampled_biggest_class_df, rest_of_data_df], ignore_index=True)

    # Stratified (by label) subsampling to overall 4500 songs
    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.75, random_state=41)).reset_index(drop=True)

    print(df.groupby('label').size())
    print(len(df))
    df.to_csv("../translation/translated/ko_translated_subsampled.csv", index=False)

