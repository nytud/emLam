#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Splits a corpus into a train/valid/test split, keeping the original sentence
order intact.
"""

from argparse import ArgumentParser
from contextlib import ExitStack
from itertools import islice
import os

from emLam.utils import openall


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', '-s', required=True,
                        help='The source directory.')
    parser.add_argument('--target-dir', '-t', required=True,
                        help='The target directory.')
    parser.add_argument('--percent', '-p', type=int, default=5,
                        help='What percent of the data should be used for both '
                             'validation and testing. The default is 5.')
    parser.add_argument('--where', '-w', choices=['front', 'back', 'both'],
                        default='both',
                        help='Where should the data of the valid and test '
                             'splits come from each file. The default is '
                             'both, which takes them from the beginning of '
                             'each odd numbered file and from the end of '
                             'each even numbered file.')
    args = parser.parse_args()

    if not (1 <= args.percent <= 20):
        parser.error('The percent value should fall between 1 and 20.')
    if args.source_dir == args.target_dir:
        parser.error('Source and target directories must differ.\n')

    return args


def copy_stream(inf, outf, lines):
    for line in islice(inf, lines):
        outf.write(line)


def wc(file):
    with openall(file) as inf:
        for line_no, _ in enumerate(inf, start=1):
            pass
    return line_no


def main():
    args = parse_arguments()

    os.nice(20)  # Play nice

    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    input_files = [os.path.join(args.source_dir, f)
                   for f in sorted(os.listdir(args.source_dir))]
    inputs = [(f, wc(f)) for f in input_files]

    with ExitStack() as stack:
        train = stack.enter_context(
            openall(os.path.join(args.target_dir, 'train.txt'), 'wt'))
        valid = stack.enter_context(
            openall(os.path.join(args.target_dir, 'valid.txt'), 'wt'))
        test = stack.enter_context(
            openall(os.path.join(args.target_dir, 'test.txt'), 'wt'))

        for fid, (f, lines) in enumerate(inputs, start=1):
            with openall(f) as inf:
                valid_sents = test_sents = int(lines * args.percent / 100)
                train_sents = lines - valid_sents - test_sents
                if args.where == 'back' or args.where == 'both' and (fid % 2 == 0):
                    copy_stream(inf, train, train_sents)
                copy_stream(inf, valid, valid_sents)
                copy_stream(inf, test, test_sents)
                if args.where == 'front' or args.where == 'both' and (fid % 2 == 1):
                    copy_stream(inf, train, train_sents)


if __name__ == '__main__':
    main()
