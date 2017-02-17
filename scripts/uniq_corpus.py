#!/usr/bin/env python3
"""
Uniq's the sentences of the corpus. Some corpora (such as the Hungarian
Webcorpus) contain a considerable number of duplicate sentences, which
skews LM evaluation.

This script has a few shortcomings: first, all data must fit into memory;
second, the order of the sentences is not preserved. However, this is not a
problem for the kind of per-sentence modeling we want to do.
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from operator import itemgetter
import os

from emLam.utils import (
    openall, read_conll, setup_stream_logger,
    source_target_file_list, write_conll
)


def parse_arguments():
    def parse_output(output):
        name, _, percent = output.partition(':')
        return (name, float(percent)) if percent else (name, None)

    def validate_outputs(outputs):
        if len(outputs) > 1:
            ps = [p for _, p in outputs]
            if None in ps or sum(ps) != 100:
                return False
        return True

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', '-s', required=True,
                        help='the source directory.')
    parser.add_argument('--target-dir', '-t', required=True,
                        help='the target directory.')
    parser.add_argument('--field', '-f', type=int, default=0,
                        help='the (zero-based) index of the column used for '
                             'determining which lines match [0].')
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    os.nice(20)  # Play nice
    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    logger = setup_stream_logger(args.log_level)
    source_target_files = source_target_file_list(
        args.source_dir, args.target_dir)

    seen = set()
    for sf, tf in source_target_files:
        logger.info('Processing {}...'.format(sf))
        s_read, s_written = 0, 0
        with openall(sf) as inf, openall(tf, 'wt') as outf:
            for sentence in read_conll(inf):
                s_read += 1
                key = u' '.join(map(itemgetter(args.field), sentence))
                if key not in seen:
                    seen.add(key)
                    write_conll(sentence, outf)
                    s_written += 1
        # Delete target file if it consists entirely of duplicate sentences
        if s_written == 0:
            os.remove(tf)
        logger.info('Processed {}: read {} sentences; written {}.'.format(
            sf, s_read, s_written))

    logger.info('Done; found {} unique sentences.'.format(len(seen)))


if __name__ == '__main__':
    main()
