#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Utility functions for the NN code."""

import logging
import os
import re

import tensorflow as tf


def load_session(sess, save_dir, saver):
    """Loads a session, returns the last completed epoch."""
    logger = logging.getLogger('emLam.nn')
    checkpoint = tf.train.get_checkpoint_state(save_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        if not saver:
            raise ValueError('A Saver must be specified to load model.')
        path = checkpoint.model_checkpoint_path
        logger.info('Load checkpoint {}'.format(path))
        saver.restore(sess, path)
        epoch = int(re.search(r'-(\d+)$', path).group(1)) + 1
        return epoch
    else:
        return None


def init_session(sess, save_dir, init):
    """Initiates a session. Returns the epoch, which is 0 in this case."""
    logger = logging.getLogger('emLam.nn')
    if not init:
        raise ValueError('"init" must be specified to initialize the model.')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    logger.info('Randomly initialize variables')
    sess.run(init)
    return 0


def init_or_load_session(sess, save_dir, saver=None, init=None):
    """
    Initiates or loads a session. In the first case, init must be specified;
    in the second, saver. Returns the last completed epoch.
    """
    if saver:
        epoch = load_session(sess, save_dir, saver)
    if epoch is None and init:
        epoch = init_session(sess, save_dir, init)
    if epoch is None:
        raise ValueError('Could not load or initialize session. Make sure '
                         'a saver or an initializer (resp.) is available')
    return epoch
