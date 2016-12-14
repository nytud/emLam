#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Data readers that processes the output of prepare_input.py."""

import math
import sys

import numpy as np

from emLam.utils import openall

class DataLoader(object):
    def __init__(self, header, batch_size, num_steps,
                 data_len, data_batches, one_hot=False,
                 data_type=np.int32, vocab_file=None):
        self.header = header
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.data_len = data_len
        self.data_batches = data_batches
        self.one_hot = one_hot
        self.data_type = data_type
        self.vocab = self._read_vocab(vocab_file) if vocab_file else None
        # Subclass-specific to avoid copying all the arguments another N times
        self.__init()

    def _read_vocab(self, vocab_file):
        with openall(vocab_file) as inf:
            return {token_freq.split('\t')[0]: i for i, token_freq in
                    enumerate(inf.read().strip().split('\n'))}

    def __iter__(self):
        raise NotImplementedError('__iter__ must be implemented.')

    def __init(self):
        """The subclass must initialize epoch_size and the data setup here."""
        raise NotImplementedError('__init must be implemented.')


class TxtDiskLoader(DataLoader):
    """Reads the text-files-per-batch format."""
    def __init(self):
        if not self.vocab:
            raise ValueError('TxtDiskLoader requires a vocabulary file.')
        self.queues = self._setup_queues(self.data_batches)
        self.epoch_size = (
            ((self.data_len // self.data_batches - 1) // self.num_steps) *
            len(self.queues[0])
        )  # -1 because targets are shifted right by 1 step

    def _setup_queues(self, data_batches):
        div, mod = divmod(data_batches, self.batch_size)
        if div == 0:
            raise ValueError('Not enough batch files ({} instead of {})'.format(
                data_batches, self.batch_size))
        elif mod != 0:
            print('The number of data files ({}) '.format(data_batches) +
                  'is not compatible with the batch size ' +
                  '({}). Only using the first '.format(self.batch_size) +
                  '{} files.'.format(self.batch_size * div), file=sys.stderr)

        ext_str = digits_format_str(data_batches)
        queues = [[] for _ in range(self.batch_size)]
        for i in range(div * self.batch_size):
            queues[i % self.batch_size].append(self.header + ext_str.format(i))
        return queues

    def __iter__(self):
        for q_step in range(len(self.queues[0])):
            infs = [openall(self.queues[i][q_step]) for i in range(self.batch_size)]
            arr = np.zeros((self.batch_size, self.num_steps + 1),
                           dtype=self.data_type)
            arr[:, -1:] = np.array(self._read_from_infs(infs, 1))
            for i in range(self.epoch_size // len(self.queues[0])):
                arr[:, 0] = arr[:, -1]
                arr[:, 1:] = np.array(
                    self._read_from_infs(infs, self.num_steps))
                if self.one_hot:
                    ret = np.zeros((self.batch_size, self.num_steps, len(self.vocab)),
                                   dtype=self.data_type)
                    ret[list(np.indices(ret.shape[:-1])) + [arr]] = 1
                    # for i in range(ret.shape[0]):
                    #     for j in range(ret.shape[1]):
                    #         ret[i, j, arr[i, j]] = 1
                else:
                    ret = arr
                yield ret[:, :self.num_steps], ret[:, 1:]
            for inf in infs:
                inf.close()

    def _read_from_infs(self, infs, num_tokens):
        return [[self.vocab[inf.readline().strip()] for _ in range(num_tokens)]
                for inf in infs]


class IntMemLoader(DataLoader):
    """Reads the int-array-in-memory format."""
    def __init(self):
        self.epoch_size = (self.data_len // self.batch_size - 1) // self.num_steps
        cropped_len = self.data_len // self.batch_size * self.batch_size
        self.data = np.load(self.header + '.npz')['data'][:cropped_len].reshape(
            self.batch_size, -1)

    def __iter__(self):
        num_steps = self.num_steps
        for i in range(self.epoch_size):
            start = i * num_steps
            end = start + num_steps
            yield self.data[:, start:end], self.data[:, start + 1:end + 1]


def digits_format_str(number):
    """Creates the format string for 0-padded integer printing up to number."""
    return '.{{:0{}}}.gz'.format(int(math.ceil(math.log10(number))))


def data_loader(header, batch_size, num_steps, one_hot=False,
                data_type=np.int32, vocab_file=None):
    with openall(header) as inf:
        fields = inf.readline().strip().split('\t')
        if fields[0] == 'TXT_DISK':
            cls = TxtDiskLoader
            data_batches = int(fields[1])
        else:
            cls = IntMemLoader
            data_batches = 0
        data_len = int(fields[-1])
    return cls(header, batch_size, num_steps, data_len, data_batches, one_hot,
               data_type, vocab_file)
