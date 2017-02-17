#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Extracts the embedding from a model trained by lstm_lm.py."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
import os

import tensorflow as tf

from emLam.utils import AttrDict, setup_stream_logger
from emLam.nn.data_input import DataLoader
from emLam.nn.lstm_model import LSTMModel
from emLam.nn.rnn import get_cell_types
from emLam.nn.softmax import get_loss_function
from emLam.nn.utils import init_or_load_session


def parse_arguments():
    parser = ArgumentParser(
        description='Extracts the embedding from a model trained by lstm_lm.py.')
    parser.add_argument('vocab_file',
                        help='the vocabulary file.')
    parser.add_argument('--model-name', '-m', default='RNN CLM',
                        help='the name of the model [RNN CLM].')
    parser.add_argument('--num-nodes', '-N', type=int, default=200,
                        help='use how many RNN cells [200].')
    parser.add_argument('--rnn-cell', '-C', default='lstm',
                        help='the RNN cell to use {{{}}} [lstm].'.format(
                            get_cell_types().keys()))
    parser.add_argument('--layers', '-L', type=int, default=1,
                        help='the number of RNN laercell to use [lstm].')
    parser.add_argument('--log-level', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    args = parser.parse_args()

    # TODO Used in lstm_lm.py too; to nn.utils.
    if args.rnn_cell.split(',')[0] not in get_cell_types().keys():
        parser.error('Cell type must be one of {{{}}}'.format(
            get_cell_types().keys()))

    return args


def main():
    args = parse_arguments()
    vocab = DataLoader.read_vocab(args.vocab_file)

    params = AttrDict(
        rnn_cell=args.rnn_cell,
        hidden_size=args.num_nodes,
        num_layers=args.layers,
        batch_size=1,
        num_steps=1,
        keep_prob=1,
        vocab_size=len(vocab),
        max_grad_norm=None,
        embedding='yes',
        data_type=tf.float32,
    )

    # TODO: is this needed?
    testsm = get_loss_function(
        'Softmax', params.hidden_size, params.vocab_size, 1, 1, params.data_type)

    with tf.Graph().as_default() as graph:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            model = LSTMModel(params, is_training=False, softmax=None)
        with tf.variable_scope("Model", reuse=True):
            embedding = tf.get_variable(
                'embedding', [params.vocab_size, params.hidden_size],
                dtype=params.data_type)
        with tf.name_scope('Global_ops'):
            saver = tf.train.Saver(name='saver')

    with tf.Session(graph=graph) as session:
        save_dir = os.path.join('saves', args.model_name)
        init_or_load_session(session, save_dir, saver)
        np_embedding = session.run(embedding)

if __name__ == '__main__':
    main()
