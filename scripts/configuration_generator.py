#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Configuration file generator for the other scripts. Useful for experiments.
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import str as text
from itertools import product
from operator import itemgetter
import os
import json

import configobj

from emLam.utils.config import handle_errors, load_config, set_dot_value
from emLam.utils import openall


def parse_arguments():
    parser = ArgumentParser(
        description='Configuration file generator for the other scripts. '
                    'Useful for experiments. Note that the schema checking '
                    'is very general, so the configuration files should be '
                    '"as valid as possible".')
    parser.add_argument('--configuration', '-c', required=True,
                        help='the configuration file to use as a base.')
    parser.add_argument('--schema', '-s', required=True,
                        help='the schema file.')
    parser.add_argument('--replacements', '-r', required=True,
                        help='the replacement mappings. A JSON dictionary, '
                             'where the keys are the (full) key paths, '
                             'and the values are either lists, or '
                             'a string that contains a '
                             'single %% character, which gets replaced by the '
                             'current values of the list-valued parameters '
                             'separated by hyphens.')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='the output directory.')
    return parser.parse_args()


def read_mappings(mapping_file):
    """
    Reads the replacement mappings. I know JSON is ugly, but for (usually)
    int and float parameters it can be written nicely.
    """
    string_mappings, list_mappings = [], []
    with openall(mapping_file) as inf:
        d = json.load(inf)
        for k, v in sorted(d.items()):
            if isinstance(v, list):
                list_mappings.append((k, v))
            elif isinstance(v, text):
                string_mappings.append((k, v))
            else:
                raise ValueError(
                    'Unsupported value type ({}: {}) for key {}'.format(
                        v, type(v), k))
    if len(list_mappings) == 0:
        raise ValueError('No list replacements found in file {}.'.format(
            mapping_file))
    return string_mappings, list_mappings


def new_file_name(file_name, values_str):
    """
    Returns the file name of the new configuration file: the original name, with
    values_str appended to it. If the original file had an extension (the part
    of the name after the last dot), it will be kept at the end of the file.
    """
    base_name, dot, ext = os.path.basename(file_name).rpartition('.')
    return base_name + '-' + values_str + dot + ext


def delete_nones(section):
    """
    Deletes all None members from a section. This should be called as-is,
    not via walk(), because the latter does not allow deletion in its contract.
    """
    to_remove = set()
    for k, v in section.items():
        if v is None:
            to_remove.add(k)
        elif isinstance(v, dict):
            delete_nones(v)
            if len(v) == 0:
                to_remove.add(k)
    for k in to_remove:
        del section[k]


def main():
    args = parse_arguments()
    config, warnings, errors = load_config(args.configuration, args.schema)
    handle_errors(warnings, errors)
    string_mappings, list_mappings = read_mappings(args.replacements)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    keys = list(map(itemgetter(0), list_mappings))
    for values in product(*map(itemgetter(1), list_mappings)):
        new_config = configobj.ConfigObj(config)
        values_str = u'-'.join(text(v) for v in values)
        for i, key in enumerate(keys):
            set_dot_value(new_config, key, values[i])
        for skey, svalue in string_mappings:
            set_dot_value(new_config, skey, svalue.replace('%', values_str))
        new_config.filename = os.path.join(
            args.output_dir, new_file_name(args.configuration, values_str))
        delete_nones(new_config)
        new_config.write()


if __name__ == '__main__':
    main()
