#!/usr/bin/env python
"""This is a script that runs a topic model on a given corpus."""

import argparse
import logging
import time

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input data
    parser.add_argument('-i', '--input', required=True, help='Path to the input corpus.'
                        ' The corpus should be in a CSV file with a "lyrics" column.')
    parser.add_argument('-l', '--lyrics_column', default='lyrics',
                        help='Name of the column in the input CSV file that contains the lyrics. Default: "lyrics".')
    parser.add_argument('-L', '--labels_column', default=None,
                        help='If set, uses this as the label column. Used for stratification in cross-validation.'
                             ' Used for computing similarity of resulting topic models after they are run.')

    # Will have -m, --model for choice of LDA, DMM, GTM.
    # Parameters for other model choices here?

    # Topic number tuning
    parser.add_argument('--min_topics', type=int, default=2, help='Minimum number of topics to try. Default: 2.')
    parser.add_argument('--max_topics', type=int, default=20, help='Maximum number of topics to try. Default: 20.')
    parser.add_argument('--n_models_in_topic_tuning', type=int, default=1,
                        help='Number of times to run cross-validation for each number of topics during tuning.')

    # Topic model runs --- preparation for hypothesis testing.
    parser.add_argument('-N', '--num_models', type=int, default=100, help='Number of topic models to run with the selected number of topics. Default: 5.')


    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.process_time()

    # Load corpus.
    # Each corpus has to be a CSV file with a column that is titled "lyrics".

    # Extract lyrics.

    # Preprocess lyrics (tokenization, UDPipe).

    # Tune number of topics.
    #  - Make 5x 80:20 splits for cross-validation. (Optionally: stratify by typology.)
    #  - For each number of topics between min_topics and max_topics:
    #      - For each split:
    #         - Initialise selected topic model
    #          - Fit on the training set of each split
    #      - Compute avg. perplexity with 5x cross-val.
    #  - Select no. of topics with best perplexity.
    #
    # (Optional: if many are tied, run a goodness-of-fit test against uniform topic distribution.)

    # Run the topic model with the selected number of topics on the whole corpus.
    # Do this N_MODELS times.
    # For each run, compute perplexity (optionally: and goodness-of-fit).
    # (Optional: discard outlier models, e.g. those with perplexity > 2x the median perplexity,
    #  and rerun until we have N_MODELS non-outlier models.)




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
