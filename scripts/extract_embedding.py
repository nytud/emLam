#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Extracts the embedding from a model trained by lstm_lm.py."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
import os

import numpy as np
import tensorflow as tf

from emLam.utils import openall


def parse_arguments():
    parser = ArgumentParser(
        description='Extracts the embedding from a model trained by lstm_lm.py.')
    parser.add_argument('vocab_file',
                        help='the vocabulary file.')
    parser.add_argument('output_file',
                        help='the output file, to which the embedding is saved.')
    parser.add_argument('--model-name', '-m', default='RNN CLM',
                        help='the name of the model [RNN CLM].')
    return parser.parse_args()


def read_vocab(vocab_file):
    with openall(vocab_file) as inf:
        return [line.split('\t')[0] for line in inf]


def main():
    args = parse_arguments()
    vocab = read_vocab(args.vocab_file)

    with tf.Session() as session:
        save_dir = os.path.join('saves', args.model_name)
        checkpoint_path = tf.train.latest_checkpoint(save_dir)
        if checkpoint_path:
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_path))
            saver.restore(session, checkpoint_path)
        else:
            raise ValueError('No saved model exists.')
        # TODO change to GLOBAL_VARIABLES in 0.12+
        embedding = tf.get_collection(tf.GraphKeys.VARIABLES,
                                      scope='Model/embedding:0')[0]
        em = session.run(embedding)
        np.savez(args.output_file + '.npz', vocab=vocab, embedding=em)


if __name__ == '__main__':
    main()
