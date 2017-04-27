#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Prepares data for training."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
import inspect
from itertools import chain
import os
import os.path as op
import random
import sys

import numpy as np

from emLam.nn.data_input import digits_format_str
from emLam.utils import openall


class CorpusCompiler(object):
    def __init__(self, input_files, batch_size, *args):
        """*args is there so that further base classes can be initialized."""
        super(CorpusCompiler, self).__init__(*args)
        self.input_files = input_files
        self.batch_size = batch_size
        self.batches = [[] for _ in range(batch_size)]
        self.ctype = get_name_stem(self.__class__, 'CorpusCompiler')

    def compile_corpus(self):
        """The function that does the work; to be implemented by subclasses."""
        raise NotImplementedError('compile_corpus must be implemented')

    def read_input(self):
        """Reads all the input into memory."""
        # TODO back to the iterator
        ret = []
        for input_file in self.input_files:
            with openall(input_file) as inf:
                ret.extend([line.split() + ['</s>'] for line in inf])
        return ret

    def balance_batches(self, sentences):
        """
        Balances the batches: always puts a new sentence into the shortest
        batch.
        """
        lens = np.zeros((self.batch_size), dtype=int)
        for sentence in sentences:
            i = lens.argmin()
            self.batches[i].append(sentence)
            lens[i] += len(sentence)

    def num_sentences(self):
        return sum(map(len, self.batches))

    def aggregate_tokens(self, fun=sum):
        return fun(map(lambda b: sum(map(len, b)), self.batches))


class DefaultCorpusCompiler(CorpusCompiler):
    """
    Creates batches from the data.

    Note that this algorithm is very rudimentary, and the chunks will be shorter
    than what might be possible with the optimal (perhaps DP-based) one.
    """
    def compile_corpus(self):
        sentences = self.read_input()
        data_len = sum(map(len, sentences))
        cut_len = data_len / float(self.batch_size)
        i, lenb, curr_cut = 0, 0, cut_len
        for sentence in sentences:
            lens = len(sentence)
            if lenb + lens <= curr_cut:
                lenb += lens
                self.batches[i].append(sentence)
            elif i + 1 < len(self.batches):
                i += 1
                lenb += lens
                curr_cut += cut_len
                self.batches[i].append(sentence)
            else:
                break


class ShuffledCorpusCompiler(CorpusCompiler):
    """Creates shuffled batches from the data."""
    def compile_corpus(self, seed=42):
        sentences = self.read_input()
        random.seed(seed)
        random.shuffle(sentences)
        self.balance_batches(sentences)
        del sentences


class SortedCorpusCompiler(CorpusCompiler):
    """Creates sorted batches from the data."""
    def compile_corpus(self):
        # The sentences are reversed twice, because balance_batches works better
        # it it can start with the longer sentences, but ultimately we want the
        # shorter sentences at the beginning
        sentences = self.read_input()
        sentences.sort(key=len, reverse=True)
        self.balance_batches(sentences)
        del sentences
        for batch in self.batches:
            batch.reverse()


class OutputWriter(object):
    """
    For writing the output into file(s). Note that this class does not work on
    its own: it has to be mingled with CorpusCompiler.
    """
    def __init__(self, output_prefix):
        super(OutputWriter, self).__init__()
        self.output_prefix = output_prefix
        self.otype = get_name_stem(self.__class__, 'OutputWriter')

    def write_output(self, *args):
        raise NotImplementedError('write_output must be implemented')

    def write_header(self, num_tokens=None):
        with openall(self.output_prefix, 'wt') as header:
            print(
                '{}\t{}\t{}\t{}\t{}'.format(
                    self.otype, self.ctype, self.batch_size,
                    self.num_sentences(), num_tokens or self.aggregate_tokens()),
                file=header
            )


class IntOutputWriter(OutputWriter):
    def __read_vocab(self, vocab_file):
        with openall(vocab_file) as inf:
            return {w: i for i, w in enumerate(l.strip().split('\t')[0]
                    for l in inf)}

    def write_output(self, vocab_file):
        vocab = self.__read_vocab(vocab_file)
        min_length = self.aggregate_tokens(min)
        self.write_header(min_length * self.batch_size)
        out_array = np.zeros((self.batch_size, min_length), dtype=np.int32)
        for i, batch in enumerate(self.batches):
            out_array[i] = np.fromiter(
                map(vocab.get, chain(*batch)), np.int32)[:min_length]
        np.savez(self.output_prefix, data=out_array)


class TxtOutputWriter(OutputWriter):
    def write_output(self, *args):
        self.write_header()
        out_ext = digits_format_str(self.batch_size)
        for i in range(self.batch_size):
            with openall(self.output_prefix + out_ext.format(i), 'wt') as outf:
                for sentence in self.batches[i]:
                    print(' '.join(sentence), file=outf)


def parse_arguments(compilers, formats):
    parser = ArgumentParser(
        description='Prepares data for training.')
    parser.add_argument('input_files', nargs='+',
                        help='the tokenized input text files.')
    parser.add_argument('--output-prefix', '-o', required=True,
                        help='the prefix of the output files\' names. It can '
                             'be a full path, in which case the directory '
                             'structure will be constructed, if needed.')
    parser.add_argument('--compiler', '-c', choices=compilers.keys(),
                        required=True,
                        help='how the sentences are compiled.')
    parser.add_argument('--format', '-f', choices=formats.keys(),
                        required=True,
                        help='the data format. The two choices are txt, '
                             'where the data is on disk in tokenized text '
                             'format, and int, where the data is in '
                             'memory as an array of ints.')
    parser.add_argument('--batch-size', '-b', type=int, required=True,
                        help='the training batch size. Only valid for the '
                             'txt_disk format.')
    parser.add_argument('--vocab-file', '-v', default=None,
                        help='the vocabulary file, created by count_vocab.py. '
                             'Only needed for the int format.')
    args = parser.parse_args()

    if args.format == 'int' and not args.vocab_file:
        parser.error('The vocabulary file is a required argument of the '
                     'int format.')

    return args


def create_dir(output_prefix):
    """
    Creates the directory structure for output_prefix (if it has one -- if
    there are not /'s in it, it's just the prefix to the file name.
    """
    dir_prefix = op.dirname(output_prefix)
    if dir_prefix and not op.isdir(dir_prefix):
        os.makedirs(dir_prefix)


def get_subclasses(cls):
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    return [c for _, c in clsmembers if issubclass(c, cls) and c != cls]


def get_name_stem(cls, suffix):
    return next(
        filter(lambda c: suffix in c.__name__, inspect.getmro(cls))
    ).__name__[:-len(suffix)].lower()


def main():
    compilers = {get_name_stem(c, 'CorpusCompiler'): c
                 for c in get_subclasses(CorpusCompiler)}
    formats = {get_name_stem(c, 'OutputWriter'): c
               for c in get_subclasses(OutputWriter)}
    args = parse_arguments(compilers, formats)
    create_dir(args.output_prefix)
    Worker = type('Worker', (compilers[args.compiler], formats[args.format]), {})
    worker = Worker(args.input_files, args.batch_size, args.output_prefix)
    worker.compile_corpus()
    worker.write_output(args.vocab_file)


if __name__ == '__main__':
    main()
