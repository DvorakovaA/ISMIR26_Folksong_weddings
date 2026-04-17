#!/usr/bin/env python
"""This is a script that evalautes IAA of the machine translation evaluation results
from collaborators.

The evaluation results JSON looks like this:

{
  "language": "Dutch",
  "exportedAt": "2026-04-01T22:42:43.617Z",
  "evaluations": [
    {
      "textId": 1,
      "title": "Title 3",
      "overallRating": 5,
      "hallucination": "no",
      "unusualLanguage": "barely",
      "comment": "",
      "words": [
        {
          "word": "Een",
          "status": "not_important"
        },
        {
          "word": "eigen",
          "status": "correct"
        },
        {
          "word": "tehuis",
          "status": "correct"
        },
        ...
      },
      {
        "textID": 2,
        "title": ...
   ]
}
"""

import argparse
import logging
import json
import os

import numpy as np

from sklearn.metrics import cohen_kappa_score

iaa_pairs = {
    'cs' : ('folksong_mt_czech_SU.json', 'folksong_mt_czech_OV.json'),
    'uk' : ('folksong_mt_ukrainian_DB.json', 'folksong_mt_ukrainian_IL.json'),
}

__version__ = "0.0.1"
__author__ = {"0.0.1" : "Anna Dvorakova"}

DUPLICATE_SONGS_MAP = {
    'cs': [4, 14],
    'et': [3, 14],
    'ko': [3, 11],
    'nl': [3, 15],
    'uk': [3, 16]
}
# Used ids refer to existing ids in the data, that go from 1!



def main(args):
    for l, pair in iaa_pairs.items():
        with open(os.path.join(args.input_dir, pair[0]), 'r', encoding='utf-8') as f1, \
             open(os.path.join(args.input_dir, pair[1]), 'r', encoding='utf-8') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)

        # Extract the ratings for overallRating and hallucination
        ratings1 = []
        ratings2 = []
        hallucinations1 = []
        hallucinations2 = []
        songs1 = {}        
        songs2 = {}
        # Collect labels for words in songs
        for eval1, eval2 in zip(data1['evaluations'], data2['evaluations']):
            if eval1['textId'] == DUPLICATE_SONGS_MAP[l][1]:
                logging.info(f"Skipping duplicate song with textId {eval1['textId']} for language {l}")
                continue
            if eval1['textId'] != eval2['textId']:
                logging.warning(f"Text IDs do not match: {eval1['textId']} vs {eval2['textId']}")
                continue
            ratings1.append(eval1['overallRating'])
            ratings2.append(eval2['overallRating'])
            hallucinations1.append(eval1['hallucination'])
            hallucinations2.append(eval2['hallucination'])

            # Collect word-level labels
            songs1[eval1['textId']] = [w['status'] for w in eval1['words']]
            songs2[eval2['textId']] = [w['status'] for w in eval2['words']]

        print(f"Ratings 1 {l}: {len(ratings1)} {ratings1}")
        print(f"Ratings 2 {l}: {len(ratings2)} {ratings2}")
        # Drop in both lists where one of the annotators has missing rating
        for i in range(len(ratings1)):
            if ratings1[i] is None or ratings2[i] is None:
                ratings1[i] = None
                ratings2[i] = None
        for i in range(len(hallucinations1)):
            if hallucinations1[i] is None or hallucinations2[i] is None:
                hallucinations1[i] = None
                hallucinations2[i] = None
        ratings1 = [r for r in ratings1 if r is not None]
        ratings2 = [r for r in ratings2 if r is not None]
        hallucinations1 = [h for h in hallucinations1 if h is not None]
        hallucinations2 = [h for h in hallucinations2 if h is not None]
        
        # Compute Cohen's kappa for overallRating and hallucination
        kappa_rating = cohen_kappa_score(ratings1, ratings2)
        kappa_hallucination = cohen_kappa_score(hallucinations1, hallucinations2)
        # Compute Cohen's kappa for word-level labels
        kappa_words = []
        kappa_words_important = []
        important_acc = []
        for textId in songs1.keys():
            if textId in songs2:
                kappa_word = cohen_kappa_score(songs1[textId], songs2[textId])
                kappa_words.append(kappa_word)

                # Compute kappa only for important words (correct/wrong)
                important_indices1 = [i for i, status in enumerate(songs1[textId]) if status in ['correct', 'wrong']]
                important_indices2 = [i for i, status in enumerate(songs2[textId]) if status in ['correct', 'wrong']]
                important_indices = list(set(important_indices1) & set(important_indices2))
                print(f"Text ID {textId}: important indices {important_indices}")
                if important_indices:
                    words1 = [songs1[textId][i] for i in important_indices]
                    words2 = [songs2[textId][i] for i in important_indices]
                    if words1 == words2:
                        print(f"Text ID {textId}: perfect agreement on important words")
                        kappa_word_important = 1.0
                    else:
                        kappa_word_important = cohen_kappa_score(
                            words1,
                            words2,
                            labels=['correct', 'wrong']
                        )
                    kappa_words_important.append(kappa_word_important)
                    important_acc.append(sum(1 for w1, w2 in zip(words1, words2) if w1 == w2) / len(words1))

        mean_kappa_words = np.mean(kappa_words) if kappa_words else 0
        mean_kappa_words_important = np.mean(kappa_words_important) if kappa_words_important else 0
        logging.info(f"Language: {l}")
        # Save the results to a file
        with open(os.path.join(args.output_dir, f'iaa_{l}_{pair[0][-7:-5]}_{pair[1][-7:-5]}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Cohen's kappa for overallRating: {kappa_rating:.4f}\n")
            f.write(f"Cohen's kappa for hallucination: {kappa_hallucination:.4f}\n")
            f.write(f"Mean Cohen's kappa for word-level labels: {mean_kappa_words:.4f}\n")
            f.write(f"Kappa for word-level labels per song: {kappa_words}\n")
            f.write(f"Mean accuracy for important word-level labels: {np.mean(important_acc):.4f}\n")
            f.write(f"Mean Cohen's kappa for important word-level labels: {mean_kappa_words_important:.4f}\n")
            f.write(f"Kappa for important word-level labels per song: {kappa_words_important}\n")

def build_argument_parser():
    parser = argparse.ArgumentParser(description='Evaluate IAA of machine translation evaluation results.')
    parser.add_argument('-i', '--input_dir', type=str, default='data/evaluation_results/',
                        help='Directory containing the evaluation results JSON files.')
    parser.add_argument('-o', '--output_dir', type=str, default='data/iaa_results/',
                        help='Directory where the IAA results will be saved.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging.')
    return parser


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    #print(cohen_kappa_score(['correct', 'correct'], ['correct', 'correct']))
    #print(cohen_kappa_score(['wrong', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'wrong', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct'], ['correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct']))
    main(args)