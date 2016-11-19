#!/usr/bin/env python3
"""Generic utility functions."""

from __future__ import absolute_import, division, print_function
from builtins import map
import bz2
import gzip
from io import open
from multiprocessing import Pool
import os
import os.path as op
import re
import sys

__allname_p = re.compile(r'^(.+?)(?:\.gz|\.bz2)?$')

if sys.version_info.major == 3:
    def openall(
        filename, mode='rt', encoding=None, errors=None, newline=None,
        buffering=-1, closefd=True, opener=None,  # for open()
        compresslevel=5,  # faster default compression
    ):
        """
        Opens all file types known to the Python SL. There are some differences
        from the stock functions:
        - the default mode is 'rt'
        - the default compresslevel is 5, because e.g. gzip does not benefit a lot
          from higher values, only becomes slower.
        """
        if filename.endswith('.gz'):
            return gzip.open(filename, mode, compresslevel,
                             encoding, errors, newline)
        elif filename.endswith('.bz2'):
            return bz2.open(filename, mode, compresslevel,
                            encoding, errors, newline)
        else:
            return open(filename, mode, buffering, encoding, errors, newline,
                        closefd, opener)
else:
    def openall(
        filename, mode='rt', encoding=None, errors=None, newline=None,
        buffering=-1, closefd=True, opener=None,  # for open()
        compresslevel=5,  # faster default compression
    ):
        """
        Opens all file types known to the Python SL. There are some differences
        from the stock functions:
        - the default mode is 'rt'
        - the default compresslevel is 5, because e.g. gzip does not benefit a lot
          from higher values, only becomes slower.
        TODO: make gzip and bz2 work at least with encoding
        """
        if filename.endswith('.gz'):
            return gzip.open(filename, mode, compresslevel)
        elif filename.endswith('.bz2'):
            return bz2.open(filename, mode, compresslevel)
        else:
            return open(filename, mode, buffering, encoding, errors, newline,
                        closefd)


def allname(fn):
    """
    Returns the "base" name of a file, removing the extensions openall()
    handles.
    """
    return __allname_p.match(fn).group(1)


def source_target_file_list(source_dir, target_dir):
    """
    Reads the list of files in source_dir, and returns a list of tuples, whose
    first field is the path to the source file, and the second is the path to a
    file with the same name in target_dir.
    """
    files = filter(op.isfile,
                   (op.join(source_dir, f) for f in os.listdir(source_dir)))
    tuples = [(f, op.join(target_dir, op.basename(f))) for f in files]
    return tuples


def run_function(fn, params, processes=1):
    """
    Runs fn with parameters params. If the number of processes is 1, it is
    equivalent to fn(params); otherwise, the function is run in a
    multiprocessing pool.
    """
    if processes < 1:
        raise ValueError('Number of processes must be at least 1.')
    elif processes == 1:
        return list(map(fn, params))
    else:
        p = Pool(processes)
        return p.map(fn, params)
