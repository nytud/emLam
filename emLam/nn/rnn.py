#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Creates various types of RNN cells."""

from functools import partial
import inspect
import sys

import tensorflow as tf


def lstm(num_cells, **kwargs):
    return tf.nn.rnn_cell.BasicLSTMCell(
        num_cells, forget_bias=float(kwargs.get('forget_bias', 1.0)),
        state_is_tuple=True)


def lstmp(num_cells, **kwargs):
    forget_bias = float(kwargs.get('forget_bias', 1.0))
    num_proj = int(kwargs['num_proj'])
    proj_clip = float(kwargs['forget_bias']) if 'proj_clip' in kwargs else None
    activation = getattr(tf, kwargs.get('activation', 'tanh'))
    return tf.nn.rnn_cell.LSTMCell(num_units=num_cells, num_proj=num_proj,
                                   proj_clip=proj_clip, forget_bias=forget_bias,
                                   state_is_tuple=True, activation=activation)


def gru(num_cells, **kwargs):
    activation = getattr(tf, kwargs.get('activation', 'tanh'))
    return tf.nn.rnn_cell.GRUCell(num_cells, activation=activation)


def get_rnn(rnn_definition, num_cells):
    cell_type_params = rnn_definition.split(',')
    cell_type = cell_type_params[0]
    params = cell_type_params[1:]
    cell_params = dict(map(partial(str.split, sep=':'), params))
    try:
        cell_fn = get_cell_types()[cell_type]
    except KeyError as ke:
        raise ValueError('No cell configuration named {} exists'.format(
            cell_type))
    try:
        return cell_fn(num_cells=num_cells, **cell_params)
    except KeyError as ke:
        raise ValueError('The parameter {} is required for cell type {}'.format(
            ke.args[0], cell_type))


def get_cell_types():
    return {nam: obj for nam, obj in inspect.getmembers(sys.modules[__name__])
            if inspect.isfunction(obj) and nam not in ['get_rnn',
                                                       'get_cell_types']}
