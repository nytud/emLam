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
    """
    Handles errors and warnings. Both are written to the stream specified; if
    exit is True, the program is stopped on any error.
    """
    if warnings:
        print('Warnings:', file=stream)
        for warning in warnings:
            print(warning, file=stream)
    if errors:
        print('Errors:', file=stream)
        for error in errors:
            print(error, file=stream)
        if exit:
            sys.exit(1)


def load_config(config_file, schema, postprocessing=None, retain=None, drop=None):
    """
    Loads a configobj-type configuration file. It is also
    possible to specify an application-specific postprocessing function which
    has three arguments: the configuration object and the warnings and errors
    lists. These are also the return values from the function.

    It is possible to specify a list of section titles that are retained (in
    this case, or others are dropped) or dropped. This allows more fine grain
    control over which sections need to be validated and which do not.
    """
    def keep_section(sections, retained, dropped):
        """Tells if the section path in question should be kept."""
        secs = set(sections)
        if retained and not secs & retained:
            return False
        if dropped and secs & dropped:
            return False
        return True

    config = configobj.ConfigObj(get_config_file(config_file),
                                 configspec=get_config_file(schema))
    warnings, errors = [], []
    v = Validator()
    results = config.validate(v, preserve_errors=True)
    if postprocessing:
        postprocessing(config, warnings, errors)

    retained = set(retain) if retain else set()
    dropped = set(drop) if drop else set()
    for sections, key in configobj.get_extra_values(config):
        if keep_section(sections, retained, dropped):
            warnings.append('Undefined key {}'.format(
                '.'.join((chain(sections, [key])))))
    for sections, key, error in configobj.flatten_errors(config, results):
        if keep_section(sections, retained, dropped):
            errors.append('{}: {}'.format(
                '.'.join((chain(sections, [key]))), error))
    return config, warnings, errors


def __find_section(configobj, section, path=[]):
    """Finds the path in the section tree to a specific section."""
    for k, v in configobj.items():
        if isinstance(v, dict):
            found_path = __find_section(v, section, path + [k])
            if found_path[-1] == section:
                return found_path
    else:
        return path


def cascade_section(configobj, section):
    """
    Returns the dictionary of a section, with all keys from the parents
    "cascaded" down to it.
    """
    path = __find_section(configobj, section)
    if path:
        ret = {}
        for sec in path:
            configobj = configobj[sec]
            for k, v in configobj.items():
                if not isinstance(v, dict):
                    ret[k] = v
        return ret
    else:
        raise ValueError('No section {} in the configuration.'.format(section))


def get_value(configobj, path):
    """
    Returns the value specified by the full path to the key (specified as an
    iterable). Returns None if the key does not exist.
    """
    for i in range(0, len(path) - 1):
        configobj = configobj.get(path[i], {})
    return configobj.get(path[-1], None)


def get_dot_value(configobj, key):
    """
    Same as get_value(), but key is in the dot-separated format; e.g.
    section1.section2.key.
    """
    return get_value(configobj, key.split('.'))


def set_value(configobj, path, value):
    """Sets the value. Path is the same as for get_value."""
    for i in range(0, len(path) - 1):
        configobj = configobj.setdefault(path[i], {})
    configobj[path[-1]] = value


def set_dot_value(configobj, key, value):
    """Same as set_value(), but key is a dot-separated format."""
    set_value(configobj, key.split('.'), value)
