#!/usr/bin/env python
"""This is a script that evalautes the machine translation evaluation results
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
import time
import json
import os

import numpy as np

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


JSONS_MAP = {
    'cs': 'folksong_mt_czech_OV.json',
    'et': 'folksong_mt_estonian_AA.json',
    'ko': 'folksong_mt_korean_DH.json',
    'nl': 'folksong_mt_dutch_PvK.json',
    'uk': 'folksong_mt_ukrainian_IL.json'
}

DUPLICATE_SONGS_MAP = {
    'cs': [4, 14],
    'et': [3, 14],
    'ko': [3, 11],
    'nl': [3, 15],
    'uk': [3, 16]
}
# Used ids refer to existing ids in the data, that go from 1!

def compute_mean_correctness(evaluations):
    n_correct = 0
    n_incorrect = 0
    n_not_important = 0
    n_total = 0
    for e in evaluations:
        for w in e['words']:
            if w['status'] == 'correct':
                n_correct += 1
            elif w['status'] == 'wrong':
                n_incorrect += 1
            elif w['status'] == 'not_important':
                n_not_important += 1
            n_total += 1
    n_important = n_correct + n_incorrect
    mean_correctness = n_correct / n_important if n_important > 0 else 0
    mean_importance = n_important / n_total if n_total > 0 else 0
    return mean_correctness, mean_importance, n_total


def evaluate_language(evaluations):
    mean_overall_rating = np.mean([e['overallRating'] for e in evaluations])
    mean_correctness, mean_importance, n_total_words = compute_mean_correctness(evaluations)
    return mean_overall_rating, mean_correctness, mean_importance, n_total_words


def report_language_evaluation(language, mean_overall_rating, mean_correctness, mean_importance, n_total_words):
    print('Language: {}'.format(language))
    print('  Mean overall rating: {:.2f}'.format(mean_overall_rating))
    print('  Mean correctness: {:.2f}'.format(mean_correctness))
    print('  Mean importance: {:.2f}'.format(mean_importance))
    print('  Total words evaluated: {}'.format(n_total_words))


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--input_jsons_root', type=str, required=True,
                        help='Path to the folder with input JSON files with the MT evaluation results.')
    parser.add_argument('-l', '--languages', nargs='+', default=['cs', 'et', 'ko', 'nl', 'uk'],
                        help='Language(s) to evaluate. If not specified, all languages with available '
                             ' JSONs will be evaluated.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.process_time()

    evaluations_per_language = {}

    # Load evaluations per language.
    # For now assumes one evaluation JSON per language.
    for l in args.languages:
        input_json = JSONS_MAP.get(l)
        if not input_json:
            logging.warning('No input JSON found for language {}. Skipping.'.format(l))
            continue
        input_json_path = os.path.join(args.input_jsons_root, input_json)

        with open(input_json_path) as f:
            mt_results = json.load(f)
            evaluations = mt_results.get('evaluations', [])
            evaluations_per_language[l] = evaluations

            if not evaluations:
                logging.error('No evaluations found in the input JSON: {}.'.format(args.input_json))
                return

    # Evaluate per language and report.
    for l, evaluations in evaluations_per_language.items():
        mean_overall_rating, mean_correctness, mean_importance, n_total_words = evaluate_language(evaluations)
        report_language_evaluation(l, mean_overall_rating, mean_correctness, mean_importance, n_total_words)

    _end_time = time.process_time()
    logging.info('scrape_cantus_db_sources.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
