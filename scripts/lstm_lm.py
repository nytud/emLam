#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Generic language modeling with RNN."""

from __future__ import absolute_import, division, print_function
import argparse
from builtins import range
from functools import partial
import glob
import os
import time

import numpy as np
import tensorflow as tf

from emLam.utils import AttrDict, setup_stream_logger
from emLam.utils.config import handle_errors, load_config
from emLam.nn.data_input import data_loader
from emLam.nn.lstm_model import LSTMModel
from emLam.nn.rnn import get_cell_types
from emLam.nn.softmax import get_loss_function
from emLam.nn.utils import init_or_load_session, load_session

TEST_STEPS = 1

logger = None


def get_sconfig(gpu_params):
    """
    Returns a session configuration object with the specified GPU parameters.
    """
    params = {}
    # params = {'log_device_placement': True}
    if gpu_params:
        params['gpu_options'] = tf.GPUOptions(*gpu_params)
    return tf.ConfigProto(**params)


def config_pp(config, warnings, errors, args):
    """Postprocessing function for the configuration."""
    if args.model_name:
        config['Network']['model_name'] = args.model_name
    if not config['Network']['model_name']:
        errors.append('Network.model_name must be specified.')
    if config['Training']['decay_delay'] == -1:
        config['Training']['decay_delay'] = config['Training']['epochs'] // 4 + 1
    if args.train and not args.valid and config['Training']['early_stopping']:
        config['Training']['early_stopping'] = 0
        warnings.append('No validation set specified: no early stopping.')
    cell_types = get_cell_types().keys()
    if config['Network']['rnn_cell'].split(',')[0] not in cell_types:
        errors.append('Cell type must be one of {{{}}}'.format(cell_types))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Character-based language modeling with RNN.')
    parser.add_argument('--train', '-t', help='the text file to train on.')
    parser.add_argument('--valid', '--dev', '-d',
                        help='the text file to use as the validation set.')
    parser.add_argument('--test', '--eval', '-e',
                        help='the text file to use as the test set.')
    parser.add_argument('--vocab', '-v', help='the vocabulary file.')
    parser.add_argument('--configuration', '-c', required=True,
                        help='the configuration file (ConfigParser/INI format).')
    parser.add_argument('--model-name', '-m',
                        help='the name of the model [RNN CLM].')
    parser.add_argument('--log-level', '-l', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='print the perplexity how many times in an '
                             'epoch [10].')
    args = parser.parse_args()

    # Config file
    pp = partial(config_pp, args=args)
    config, warnings, errors = load_config(
        args.configuration, 'lstm_lm_conf.schema', pp)
    handle_errors(warnings, errors)

    if not args.train and not args.test:
        parser.error('At least one of the train or test sets must be specified.')

    return args, config


def run_epoch(session, model, data, epoch_size=0, verbose=0,
              global_step=0, writer=None):
    """
    Runs an epoch on the network.
    - epoch_size: if 0, it is taken from data
    - data: a DataLoader instance
    """
    # TODO: these two should work together better, i.e. keep the previous
    #       iteration state intact if epoch_size != 0; also loop around
    epoch_size = data.epoch_size if epoch_size <= 0 else epoch_size
    data_iter = iter(data)
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = [model.cost, model.final_state, model.train_op]
    fetches_summary = fetches + [model.summaries]
    if verbose:
        log_every = epoch_size // verbose

    for step in range(epoch_size):
        x, y = next(data_iter)

        # feed_dict = {model.sequence: batch}
        # for i, (c, h) in enumerate(model.initial_state):
        #     feed_dict[c] = state[i].c  # CEC for layer i
        #     feed_dict[h] = state[i].h  # hidden for layer i
        feed_dict = {
            model.input_data: x,
            model.targets: y,
            model.initial_state: state
        }

        if verbose and step % log_every == log_every - 1:
            cost, state, _, summary = session.run(fetches_summary, feed_dict)
            if writer:
                writer.add_summary(summary, global_step=global_step)
            if model.is_training:
                global_step += 1
        else:
            cost, state, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.params.num_steps
        if verbose and step % log_every == log_every - 1:
            logger.debug(
                "%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 iters * model.params.batch_size / (time.time() - start_time))
            )

    # global_step is what the user sees, i.e. if the output is verbose, it is
    # increased, otherwise it isn't
    if not verbose and model.is_training:
        global_step += 1

    return np.exp(costs / iters), global_step


def stop_early(valid_ppls, early_stop, save_dir):
    """
    Stops early, i.e.
    - checks if we want early stopping and if the PPL of the validation set
      has been detoriating
    - deletes all checkpoints later than the best performing one.
    - return True if we stopped early; False otherwise
    """
    if (
        early_stop > 0 and
        np.argmin(valid_ppls) < len(valid_ppls) - early_stop
    ):
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        all_checkpoints = checkpoint.all_model_checkpoint_paths
        tf.train.update_checkpoint_state(
            save_dir, all_checkpoints[-early_stop - 1],
            all_checkpoints[:-early_stop])
        for checkpoint_to_delete in all_checkpoints[-early_stop:]:
            for file_to_delete in glob.glob(checkpoint_to_delete + '*'):
                os.remove(file_to_delete)
        logger.info('Stopping training due to overfitting; deleted models ' +
                    'after {}'.format(
                        all_checkpoints[-early_stop - 1].rsplit('-', 1)[-1]))
        return True
    else:
        return False


def main():
    args, config = parse_arguments()

    network_params = AttrDict(config['Network'])
    network_params.data_type = tf.float32
    train_params = AttrDict(config['Training'])
    train_params.update(network_params)
    valid_params = AttrDict(config['Validation'])
    valid_params.update(network_params)
    test_params = AttrDict(config['Evaluation'])
    test_params.update(network_params)

    train_data = valid_data = test_data = None
    if args.train:
        train_data = data_loader(args.train, train_params.batch_size,
                                 train_params.num_steps, vocab_file=args.vocab)
        train_params.vocab_size = len(train_data.vocab)
        trainsm = get_loss_function(
            train_params.softmax, train_params.hidden_size,
            train_params.vocab_size, train_params.batch_size,
            train_params.num_steps, train_params.data_type)
    if args.valid:
        valid_data = data_loader(args.valid, args.batch_size, args.num_steps,
                                 vocab_file=args.vocab)
        valid_params.vocab_size = len(valid_data.vocab)
        validsm = get_loss_function(
            valid_params.softmax, valid_params.hidden_size,
            valid_params.vocab_size, valid_params.batch_size,
            valid_params.num_steps, valid_params.data_type)
    if args.test:
        test_data = data_loader(args.test, test_params.batch_size,
                                test_params.num_steps, vocab_file=args.vocab)
        test_params.vocab_size = len(test_data.vocab)
        testsm = get_loss_function(
            test_params.testsm, test_params.hidden_size,
            test_params.vocab_size, test_params.batch_size,
            test_params.num_steps, test_params.data_type)

    global logger
    logger = setup_stream_logger(args.log_level)

    with tf.Graph().as_default() as graph:
        # init_scale = 1 / math.sqrt(args.num_nodes)
        initializer = tf.random_uniform_initializer(
            -network_params.init_scale, network_params.init_scale)

        if args.train:
            with tf.name_scope('Train'):
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    mtrain = LSTMModel(train_params, is_training=True, softmax=trainsm)
        if args.valid:
            with tf.name_scope('Valid'):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mvalid = LSTMModel(valid_params, is_training=False, softmax=validsm)
        if args.test:
            with tf.name_scope('Test'):
                with tf.variable_scope("Model", reuse=not args.test_only,
                                       initializer=initializer):
                    mtest = LSTMModel(test_params, is_training=False, softmax=testsm)
        with tf.name_scope('Global_ops'):
            saver = tf.train.Saver(
                name='saver',
                max_to_keep=max(10, config['Training']['early_stopping'] + 1))
            init = tf.initialize_all_variables()

    # TODO: look into Supervisor
    # The training itself
    with tf.Session(graph=graph, config=get_sconfig(config.get('GPU'))) as sess:
        save_dir = os.path.join('saves', network_params.model_name)

        if args.train:
            boards_dir = os.path.join('boards', network_params.model_name)
            writer = tf.train.SummaryWriter(boards_dir, graph=graph)
            last_epoch = init_or_load_session(sess, save_dir, saver, init)
            global_step = 0  # TODO not if we load the model...
            logger.info('Starting...')
            if args.valid:
                logger.info('Epoch {:2d}-                 valid PPL {:6.3f}'.format(
                    last_epoch, run_epoch(sess, mvalid, valid_data, 0, verbose=10)[0]))

            valid_ppls = []
            for epoch in range(last_epoch, args.epochs + 1):
                lr_decay = args.lr_decay ** max(epoch - args.decay_delay, 0.0)
                mtrain.assign_lr(sess, args.learning_rate * lr_decay)

                train_perplexity, global_step = run_epoch(
                    sess, mtrain, train_data, 0, verbose=args.verbose,
                    global_step=global_step, writer=writer)
                if args.valid:
                    valid_perplexity, _ = run_epoch(sess, mvalid, valid_data)
                    valid_ppls.append(valid_perplexity)
                    valid_ppl = '{:6.3f}'.format(valid_perplexity)
                else:
                    valid_ppl = 'N/A'
                logger.info('Epoch {:2d} train PPL {:6.3f} valid PPL {:6.3f}'.format(
                    epoch, train_perplexity, valid_ppl))
                saver.save(sess, os.path.join(save_dir, 'model'), epoch)

                # Check for overfitting
                if stop_early(valid_ppls, train_params.early_stopping, save_dir):
                    break

            writer.close()

        if args.test:
            load_session(sess, save_dir, saver)
            logger.info('Running evaluation...')
            test_perplexity, _ = run_epoch(sess, mtest, test_data)
            logger.info('Test perplexity: {:.3f}'.format(test_perplexity))


if __name__ == '__main__':
    main()
