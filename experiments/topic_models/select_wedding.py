"""
Reads the parquet file with BERTopic probabilities
and selects only rows with item_id in the wedding subset specified by
csv files and add language label based on the csv file name.
Writes a new parquet file with the filtered data.
"""
import os
import pandas as pd

WEDDING_CSV_DIR = "../../wedding_set"
PROBABILITIES_PARQUET = "bertopic_output/probabilities.parquet"
OUTPUT_PARQUET = "bertopic_output/probabilities_wedding.parquet"

def main():
    # Load the parquet file with probabilities
    prob_df = pd.read_parquet(PROBABILITIES_PARQUET)
    
    # Collect all item_ids from the wedding CSV files
    wedding_item_ids = {}
    number_of_wedding_item_ids = 0
    for filename in os.listdir(WEDDING_CSV_DIR):
        if filename.endswith(".csv"):
            csv_path = os.path.join(WEDDING_CSV_DIR, filename)
            wedding_df = pd.read_csv(csv_path)
            wedding_item_ids[filename[0:2]] = list(wedding_df["item_id"].unique())
            number_of_wedding_item_ids += len(wedding_item_ids[filename[0:2]])
            print(len(wedding_item_ids[filename[0:2]]))
    
    print(f"Total wedding item_ids collected: {number_of_wedding_item_ids}")
    
    # Filter the original dataframe to keep only rows with item_id in the wedding subset
    filtered_dfs = []
    for lang, ids in wedding_item_ids.items():
        filtered_df = prob_df[prob_df["id"].isin(ids)].copy()
        filtered_df["lang"] = lang  # Add language label based on the CSV file name
        filtered_dfs.append(filtered_df)

    # Concat
    result_df = pd.concat(filtered_dfs, ignore_index=True)

   
    result_df.drop_duplicates(subset=['id'], inplace=True)  # Ensure no duplicate item_ids
    # Write the filtered dataframe to a new parquet file
    result_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved filtered parquet with {len(result_df)} rows to {OUTPUT_PARQUET}")
    print(len(result_df.drop_duplicates(subset=['id'])))  # Check for duplicates

if __name__ == "__main__":
    main()