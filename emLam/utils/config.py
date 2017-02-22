#!/usr/bin/env python3
"""Configuration-related functionality."""

from __future__ import absolute_import, division, print_function
from itertools import chain
import os
from pkg_resources import resource_exists, resource_filename
import sys

import configobj
from validate import Validator


def get_config_file(config_file):
    """
    Returns the path to the configuration file specified. If there is a file at
    the path specified, it is returned as is; if not, the conf/ directory of the
    installed package is checked. If that fails as well, ValueError is raised.
    """
    if os.path.isfile(config_file):
        return config_file
    elif resource_exists('conf', config_file):
        return resource_filename('conf', config_file)
    else:
        raise ValueError('Could not find configuration file {}'.format(config_file))


def handle_errors(warnings, errors, stream=sys.stderr, exit=True):
    """Handles errors and warnings."""
    for warning in warnings:
        print(warning, file=stream)
    if errors:
        for error in errors:
            print(error, file=stream)
        if exit:
            sys.exit(1)


def load_config(config_file, schema, postprocessing=None):
    """
    Loads a configobj-type configuration file. It is also
    possible to specify an application-specific postprocessing function which
    has three arguments: the configuration object and the warnings and errors
    lists. These are also the return values from the function.
    """
    config = configobj.ConfigObj(get_config_file(config_file),
                                 configspec=get_config_file(schema))
    warnings, errors = [], []
    v = Validator()
    results = config.validate(v, preserve_errors=True)
    if postprocessing:
        postprocessing(config, warnings, errors)

    for sections, key in configobj.get_extra_values(config):
        warnings.append('Undefined key {}'.format(
            '.'.join((chain((*sections, key))))))
    for sections, key, error in configobj.flatten_errors(config, results):
        errors.append('{}: {}'.format('.'.join((chain((*sections, key)))), error))
    return config, warnings, errors
