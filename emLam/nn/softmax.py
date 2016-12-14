#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Implementation of softmax and (some of) its alternatives for lstm_model.py.
All functions return a (shared) loss and a prediction tensor.
"""
from functools import partial

import tensorflow as tf


class LossAndPrediction(object):
    """
    Creates a loss and a prediction tensor (the latter only for evaluation).
    """
    def __init__(self, hidden_size, vocab_size, batch_size, num_steps,
                 data_type, kwargs):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.data_type = data_type
        self.kwargs = kwargs

    def __call__(self):
        raise NotImplementedError('Not implemented; use a subclass')


class Softmax(LossAndPrediction):
    def __call__(self, outputs, targets, need_prediction=False):
        """Computes the shared loss over all time-steps."""
        flat_output = tf.reshape(outputs, [-1, self.hidden_size])
        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden_size, self.vocab_size],
            dtype=self.data_type)
        softmax_b = tf.get_variable(
            "softmax_b", [self.vocab_size], dtype=self.data_type)
        logits = tf.matmul(flat_output, softmax_w) + softmax_b

        cost = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(targets, [-1])],
            [tf.ones([self.batch_size * self.num_steps],
                     dtype=self.data_type)])
        loss = tf.reduce_sum(cost) / self.batch_size
        if need_prediction:
            prediction = tf.reshape(
                tf.nn.softmax(logits), [-1, self.num_steps, self.vocab_size])
            return loss, prediction
        else:
            return loss


class SamplingError(LossAndPrediction):
    """Common ancestor for the sampling error classes."""
    def loss_and_prediction(self, outputs, targets, loss_func):
        try:
            num_sampled = int(self.kwargs['num_sampled'])
        except ValueError:
            raise ValueError('The number of samples must be an integer.')
        except KeyError:
            raise ValueError('The parameter num_samples is missing.')
        flat_output = tf.reshape(outputs, [-1, self.hidden_size])
        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden_size, self.vocab_size], dtype=self.data_type)
        softmax_b = tf.get_variable(
            "softmax_b", [self.vocab_size], dtype=self.data_type)
        cost = loss_func(
            tf.transpose(softmax_w), softmax_b,         # .T for some reason
            flat_output, tf.reshape(targets, [-1, 1]),  # Column vector
            num_sampled, self.vocab_size)
        loss = tf.reduce_sum(cost) / self.batch_size
        return loss


class NceLoss(SamplingError):
    def __call__(self, outputs, targets):
        return super().loss_and_prediction(outputs, targets, tf.nn.nce_loss)


class SampledSoftmax(SamplingError):
    def __call__(self, outputs, targets):
        return super().loss_and_prediction(outputs, targets,
                                           tf.nn.sampled_softmax_loss)


def get_loss_function(loss_definition, hidden_size, vocab_size, batch_size,
                      num_steps, data_type):
    defs = loss_definition.split(',')
    def_params = dict(map(partial(str.split, sep=':'), defs[1:]))
    return globals()[defs[0]](hidden_size, vocab_size, batch_size, num_steps,
                              data_type, def_params)


# def differentiated_softmax(outputs, targets, hidden_size, vocab_size,
#                            batch_size, num_steps, data_type, **kwargs):
#     sizes = kwargs['sizes']
#     if vocab_size != sum(sizes):
#         raise ValueError('The embedding partition sizes do not sum up.')
#     flat_output = tf.reshape(outputs, [-1, hidden_size])
#     softmax_ws = [tf.get_variable("softmax_w_{}".format(i),
#                                   [hv_sizes[0], hv_sizes[1]], dtype=data_type)
#                   for i, hv_sizes in sizes]
#     softmax_bs = [tf.get_variable("softmax_b_{}".format(i),
#                                   [hv_sizes[1]], dtype=data_type)
#                   for i, hv_sizes in sizes]
