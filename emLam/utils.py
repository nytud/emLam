#!/usr/bin/env python3
"""Generic utility functions."""

from __future__ import absolute_import, division, print_function
from builtins import map
import bz2
from functools import partial
import gzip
from io import open, TextIOWrapper
from multiprocessing import Manager, Pool
import os
import os.path as op
from queue import Queue
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
    class PatchedGzipFile(gzip.GzipFile):
        """
        GzipFile does not implement read1(), which is need by TextIOWrapper.
        Good, consistent API design! :/ See https://bugs.python.org/issue10791.
        """
        def read1(self, n):
                return self.read(n)

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
            f = PatchedGzipFile(filename, mode, compresslevel)
        elif filename.endswith('.bz2'):
            f = bz2.open(filename, mode, compresslevel)
        else:
            return open(filename, mode, buffering, encoding, errors, newline,
                        closefd)
        return TextIOWrapper(f, encoding, errors, newline)


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
        ret = p.map(fn, params)
        p.close()
        p.join()
        return ret


def run_queued(fn, params, processes=1, queued_params=None):
    """
    Same as run_function, but the function should be such that it accepts a
    parameter queue and reads its inputs from there; params still contains
    options for the function.
    """
    if processes < 1:
        raise ValueError('Number of processes must be at least 1.')
    queue = Queue() if processes == 1 else Manager().Queue()
    f = partial(fn, queue=queue)
    if processes == 1:
        return list(map(f, params))
    else:
        p = Pool(processes)
        ret = p.map(f, params)
        p.close()
        p.join()
        return ret
