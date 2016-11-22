#!/usr/bin/env python3
"""
Preprocesses the Szeged Treebank. Does not do much, just rearranges the
columns and splits the last into two (lemma, pos).
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from functools import partial
import os

from emLam import WORD, LEMMA, LEMMA_POS
from emLam.utils import openall, source_target_file_list


def parse_arguments():
    parser = ArgumentParser(
        description='Preprocesses the Szeged Treebank files to the same format '
                    'required by the LM scripts. Creates tsv files where the '
                    'first column is the word form, the second the lemma and '
                    'the third the POS. The latter two is extracted from the '
                    'last column of the input files. The rest of the columns '
                    'are dropped or optionally appended to the first three.')
    parser.add_argument('--source-dir', '-s', required=True,
                        help='the source directory.')
    parser.add_argument('--target-dir', '-t', required=True,
                        help='the target directory.')
    parser.add_argument('--keep-columns', '-k', action='store_true',
                        help='keep all columns. By default, the output files '
                             'will only have 3 columns: word, lemma, POS.')

    args = parser.parse_args()
    if args.source_dir == args.target_dir:
        parser.error('Source and target directories must differ.')
    return args


def preprocess_szeged(source_target_file, keep_columns):
    source_file, target_file = source_target_file
    with openall(source_file) as inf, openall(target_file, 'wt') as outf:
        for line_no, line in enumerate(inf):
            if line == '\n':
                print(u'', file=outf)
            else:
                fields = line.rstrip('\n').split('\t')
                pos_start = fields[LEMMA_POS].find('[')
                if pos_start >= 0:
                    lemma = fields[LEMMA_POS][:pos_start]
                    pos = fields[LEMMA_POS][pos_start:]
                else:
                    # OTHER
                    lemma = fields[LEMMA]
                    pos = fields[LEMMA_POS]
                out_fields = [fields[WORD], lemma, pos]
                if keep_columns:
                    out_fields.extend(fields[2:-1])
                print(u'\t'.join(out_fields), file=outf)


def main():
    args = parse_arguments()
    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    os.nice(20)  # Play nice

    for t in source_target_file_list(args.source_dir, args.target_dir):
        partial(preprocess_szeged, keep_columns=args.keep_columns)(t)


if __name__ == '__main__':
    main()
