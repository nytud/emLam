#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Prepares data for training."""

from __future__ import absolute_import, division, print_function
from future import standard_library
from argparse import ArgumentParser
from itertools import chain
import os
import os.path as op
import subprocess

import numpy as np

from emLam.nn.data_input import digits_format_str
from emLam.utils import openall

standard_library.install_aliases()


class CorpusCompiler(object):
    def __init__(self, input_files, output_prefix):
        self.input_files = input_files
        self.output_prefix = output_prefix
        self.input_size = self.input_length(input_files)

    @staticmethod
    def input_length(input_files):
        sum_len = 0
        for input_file in input_files:
            if input_file.endswith('.gz'):
                cmd = 'zcat'
            elif input_file.endswith('.bz2'):
                cmd = 'bzcat'
            else:
                cmd = 'cat'
            s = subprocess.getoutput('{} "{}" | wc -wl'.format(cmd, input_file))
            fields = s.strip().rsplit('\n', 1)[-1].strip().split()
            sum_len += int(fields[0]) + int(fields[1])
        return sum_len

    @staticmethod
    def read_input(input_files):
        """Reads the input files one-by-one and yields tokens line-by-line."""
        for input_file in input_files:
            with openall(input_file) as inf:
                for line in inf:
                    yield line.strip().split() + ['</s>']


class TxtDiskCompiler(CorpusCompiler):
    def __init__(self, input_files, output_prefix, batch_size):
        super(TxtDiskCompiler, self).__init__(input_files, output_prefix)
        self.batch_size = batch_size

    def __call__(self):
        """Does the work."""
        num_outs = self.batch_size  # The modulo is dropped
        out_size = self.input_size // num_outs
        out_ext = digits_format_str(num_outs)

        with openall(self.output_prefix, 'wt') as header:
            print('TXT_DISK\t{}\t{}'.format(
                num_outs, self.input_size), file=header)

        input_iter = self.read_input(self.input_files)
        tokens = []
        for outi in range(num_outs):
            written = 0
            with openall(self.output_prefix + out_ext.format(outi), 'wt') as outf:
                while written < out_size:
                    tokens.extend(next(input_iter))
                    tokens_to_write = tokens[:out_size - written]
                    print(('\n'.join(tokens_to_write)), file=outf)
                    written += len(tokens_to_write)
                    tokens = tokens[len(tokens_to_write):]


class IntMemCompiler(CorpusCompiler):
    def __init__(self, input_files, output_prefix, vocab_file):
        super(IntMemCompiler, self).__init__(input_files, output_prefix)
        self.read_vocab(vocab_file)

    def read_vocab(self, vocab_file):
        with openall(vocab_file) as inf:
            self.vocab = {w: i for i, w in enumerate(l.strip().split('\t')[0]
                          for l in inf)}

    def __call__(self):
        """Does the work."""
        with openall(self.output_prefix, 'wt') as header:
            print('INT_MEM\t{}'.format(self.input_size), file=header)

        arr = np.zeros((self.input_size), dtype=np.int32)
        index = 0
        for tokens in self.read_input(self.input_files):
            for token in tokens:
                arr[index] = self.vocab[token]
                index += 1

        np.savez(self.output_prefix, data=arr)


class BatchIntMemCompiler(IntMemCompiler):
    """
    A kind of a mix between the two above. It is still an IntMemCompiler, but
    the sentences are sorted by length and there is a batch parameter. Useful
    for validation / test sets.
    """
    def __init__(self, input_files, output_prefix, vocab_file, batch_size):
        super(BatchIntMemCompiler, self).__init__(
            input_files, output_prefix, vocab_file)
        self.batch_size = batch_size

    def __call__(self):
        """Does the work."""
        with openall(self.output_prefix, 'wt') as header:
            print('BATCH_INT_MEM\t{}\t{}'.format(self.batch_size, self.input_size),
                  file=header)

        data = [[self.vocab[token] for token in tokens]
                for tokens in self.read_input(self.input_files)]
        # data = sorted(data, key=lambda s: -len(s))

        out_lists = [[] for _ in range(self.batch_size)]
        lens = np.zeros((self.batch_size), dtype=int)
        for s in data:
            i = lens.argmin()
            out_lists[i].append(s)
            lens[i] += len(s)
        print(lens)

        # So that we don't keep the data 3 times in memory
        del data
        min_length = lens.min()
        out_array = np.zeros((self.batch_size, min_length), dtype=np.int32)
        for i in range(self.batch_size):
            out_array[i] = np.fromiter(chain(*out_lists[i]), np.int32)[:min_length]

        np.savez(self.output_prefix, data=out_array)


def parse_arguments():
    parser = ArgumentParser(
        description='Prepares data for training.')
    parser.add_argument('input_files', nargs='+',
                        help='the tokenized input text files.')
    parser.add_argument('--output-prefix', '-o', required=True,
                        help='the prefix of the output files\' names. It can '
                             'be a full path, in which case the directory '
                             'structure will be constructed, if needed.')
    parser.add_argument('--format', '-f',
                        choices=['txt_disk', 'int_mem', 'batch_int_mem'],
                        help='the data format. The two choices are txt_disk, '
                             'where the data is on disk in tokenized text '
                             'format, and int_mem, where the data is in '
                             'memory as an array of ints.')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='the training batch size. Only valid for the '
                             'txt_disk format.')
    parser.add_argument('--vocab-file', '-v', default=None,
                        help='the vocabulary file, created by count_vocab.py. '
                             'Only needed for the int_mem format.')
    args = parser.parse_args()

    if args.format == 'txt_disk' and not args.batch_size:
        parser.error('The number of batches is a required argument of the '
                     'txt_disk format.')
    if args.format == 'int_mem' and not args.vocab_file:
        parser.error('The vocabulary file is a required argument of the '
                     'int_mem format.')

    return args


def create_dir(output_prefix):
    """
    Creates the directory structure for output_prefix (if it has one -- if
    there are not /'s in it, it's just the prefix to the file name.
    """
    dir_prefix = op.dirname(output_prefix)
    if dir_prefix and not op.isdir(dir_prefix):
        os.makedirs(dir_prefix)


def main():
    args = parse_arguments()
    create_dir(args.output_prefix)
    if args.format == 'txt_disk':
        c = TxtDiskCompiler(args.input_files, args.output_prefix, args.batch_size)
    elif args.format == 'int_mem':
        c = IntMemCompiler(args.input_files, args.output_prefix, args.vocab_file)
    else:
        c = BatchIntMemCompiler(args.input_files, args.output_prefix,
                                args.vocab_file, args.batch_size)
    c()


if __name__ == '__main__':
    main()
