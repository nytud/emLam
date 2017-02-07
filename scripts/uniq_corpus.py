#!/usr/bin/env python3
"""
Uniq's the sentences of the corpus. Some corpora (such as the Hungarian
Webcorpus) contain a considerable number of duplicate sentences, which
skews LM evaluation.

This script has a few shortcomings: first, all data must fit into memory;
second, the order of the sentences is preserved. However, this is not a problem
for the kind of per-sentence modeling we want to do.
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from operator import itemgetter
import os
import os.path as op

from emLam.utils import openall, read_conll


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
    parser.add_argument('--chunk', '-c', type=int, required=True,
                        help='how many sentences are written to an output file.')
    parser.add_argument('--field', '-f', type=int, default=0,
                        help='the (zero-based) index of the column used for '
                             'determining which lines match [0].')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    os.nice(20)  # Play nice
    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    source_files = [op.join(args.source_dir, f) for f in
                    os.listdir(args.source_dir)]
    seen = set()
    for sf in source_files:
        with openall(sf) as inf:
            for sentence in read_conll(inf):
                key = u' '.join(map(itemgetter(args.field), sentence))
                if key not in seen:
                    seen.add(key)
                    pass  # TODO

if __name__ == '__main__':
    main()
