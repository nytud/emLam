#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Counts the vocabulary. Used by the other input formatting scripts."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from collections import Counter

from emLam.utils import openall


def parse_arguments():
    parser = ArgumentParser(
        description='Counts the vocabulary. Used by the other input '
                    'formatting scripts.')
    parser.add_argument('input_files', nargs='+',
                        help='the tokenized input text files.')
    parser.add_argument('--vocab-file', '-v', required=True,
                        help='the output file.')
    return parser.parse_args()


def count_input(input_files):
    """Counts the token types in the input files."""
    vocab = Counter()
    eos = 0
    for input_file in input_files:
        with openall(input_file) as inf:
            for line in inf:
                eos += 1
                tokens = line.strip().split()
                vocab.update(tokens)
    vocab['</s>'] = eos
    return vocab


def write_vocab(vocab, vocab_file):
    """Writes the vocabulary to file."""
    with openall(vocab_file, 'wt') as outf:
        for token, freq in sorted(vocab.items(), key=lambda tf: (-tf[1], tf[0])):
            print('{}\t{}'.format(token, freq), file=outf)


def main():
    args = parse_arguments()
    vocab = count_input(args.input_files)
    write_vocab(vocab, args.vocab_file)


if __name__ == '__main__':
    main()
