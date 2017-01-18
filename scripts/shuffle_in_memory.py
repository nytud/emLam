#!/usr/bin/env python3
"""
Shuffles a SRILM-format dataset in memory and optionally splits the result into
smaller datasets.
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
import os
import random

from emLam.utils import openall


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

    parser = ArgumentParser(
        description='Shuffles a SRILM-format dataset in memory.')
    parser.add_argument('--source-dir', '-s', required=True,
                        help='the source directory.')
    parser.add_argument('--target-dir', '-t', required=True,
                        help='the target directory.')
    parser.add_argument('--output', '-o', action='append', required=True,
                        help='the name of the output dataset(s). It is '
                             'possible to specify more than one, in which case '
                             'the format becomes -o dataset_name:percent, '
                             'where percent controls what slice of data ends '
                             'up in that particular dataset.')
    parser.add_argument('--seed', type=int, default=42,
                        help='the random seed.')

    args = parser.parse_args()
    outputs = [parse_output(o) for o in args.output]
    if not validate_outputs(outputs):
        parser.error('Invalid output specification.')
    return args.source_dir, args.target_dir, outputs, args.seed


def read_data(source_dir):
    def _read():
        for f in os.listdir(source_dir):
            with openall(os.path.join(source_dir, f)) as inf:
                for line in inf:
                    yield line.strip()
    return [l for l in _read()]


def main():
    source_dir, target_dir, outputs, seed = parse_arguments()
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    data = read_data(source_dir)
    random.seed(seed)
    random.shuffle(data)

    borders = [p if p is not None else 100 for _, p in outputs]
    for i in range(len(borders) - 1, -1, -1):
        for j in range(len(borders) - 1, i, -1):
            borders[j] += borders[i]
    borders = [0] + [int(len(data) * p / 100) for p in borders]

    for i, dataset in enumerate(ds for ds, _ in outputs):
        with openall(os.path.join(target_dir, dataset + '.gz'), 'wt') as outf:
            for l in data[borders[i]:borders[i + 1]]:
                print(l, file=outf)


if __name__ == '__main__':
    main()
