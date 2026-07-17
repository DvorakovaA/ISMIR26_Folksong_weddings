"""
Do statistics on topic assignments.

Input columns: fold,id,true_label,pred_label,correct,assigned_topic,topic_label
"""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Do statistics on topic assignments.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_txt", type=str, help="Path to the output TXT file.")
    args = parser.parse_args()

    # Read the input CSV file
    df = pd.read_csv(args.input_csv)
    with open(args.output_txt, "w") as f:
        # Overall statistics
        print("Overall statistics:", file=f)
        print('  Number of songs:', len(df), file=f)
        print('  Topics assigned:', len(df['assigned_topic'].unique()), file=f)
        print('\n20 most prominent hard topics (sorted):', file=f)
        for topic, group in sorted(df.groupby("topic_label"), key=lambda x: len(x[1]), reverse=True)[:20]:
            print(f'  Topic {topic}: {len(group)} songs', file=f)

        # By language
        print('\n\nStatistics by language:', file=f)
        for lang, group in df.groupby("true_label"):
            print(f'\nLanguage: {lang}', file=f)
            print('  Number of songs:', len(group), file=f)
            print('  Topics assigned:', len(group['assigned_topic'].unique()), file=f)
            print('  Most prominent hard topics (sorted):', file=f)
            for topic, sub_group in sorted(group.groupby("topic_label"), key=lambda x: len(x[1]), reverse=True)[:20]:
                print(f'    Topic {topic}: {len(sub_group)} songs', file=f)
            
        


if __name__ == "__main__":
    main()