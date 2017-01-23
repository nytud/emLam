#!/usr/bin/env python3
"""Generic utility functions."""

from __future__ import absolute_import, division, print_function
from builtins import map
from contextlib import contextmanager
import bz2
from functools import partial
import gzip
from io import open, TextIOWrapper
import logging
from multiprocessing import Manager, Pool
import os
import os.path as op
from queue import Queue
import re
import sys

from emLam.logging import QueueListener, QueueHandler

__allname_p = re.compile(r'^(.+?)(\.gz|\.bz2)?$')

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
        GzipFile in Python 2.7 does not implement read1(), which is needed by
        TextIOWrapper.  Good, consistent API design! :/
        See https://bugs.python.org/issue10791.
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
    Returns the "base" name and the extension of a file separately, Only the
    extensions openall() handles are considered.
    """
    return __allname_p.match(fn).groups()


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


@contextmanager
def __configure_logging(fn, processes, logging_level, **kwargs):
    if logging_level:
        logging_queue = Queue() if processes == 1 else Manager().Queue()
        sh = logging.StreamHandler()
        sh.setLevel(logging_level)
        process_log = '(%(process)d)' if processes > 1 else ''
        f = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s{} %(message)s'.format(process_log))
        sh.setFormatter(f)
        ql = QueueListener(logging_queue, sh)
        ql.start()
        f = partial(fn, logging_level=logging_level,
                    logging_queue=logging_queue, **kwargs)
    else:
        f = partial(fn, **kwargs)
        ql = None

    yield f

    if ql:
        ql.stop()


def run_function(fn, params, processes=1, logging_level=None):
    """
    Runs fn with parameters params. If the number of processes is 1, it is
    equivalent to fn(params); otherwise, the function is run in a
    multiprocessing pool.
    """
    if processes < 1:
        raise ValueError('Number of processes must be at least 1.')

    with __configure_logging(fn, processes, logging_level) as f:
        if processes == 1:
            return list(map(f, params))
        else:
            p = Pool(processes)
            ret = p.map(f, params)
            p.close()
            p.join()
            return ret


def run_queued(fn, params, processes=1, queued_params=None, logging_level=None):
    """
    Same as run_function, but the function should be such that it accepts a
    parameter queue and reads its inputs from there; params still contains
    options for the function.

    If logging_level is not None, a QueueHandler -- QueueListener pair is set
    up so that the function can forward its logging to the stderr of the main
    process. In this case, the params sent to the function are extended by
    this logging level and the logging queue.
    """
    if processes < 1:
        raise ValueError('Number of processes must be at least 1.')

    queue = Queue() if processes == 1 else Manager().Queue()
    for qp in queued_params:
        queue.put_nowait(qp)

    with __configure_logging(fn, processes, logging_level, queue=queue) as f:
        # f = partial(fn, queue=queue)
        if processes == 1:
            ret = list(map(f, params))
        else:
            p = Pool(processes)
            ret = p.map(f, params)
            p.close()
            p.join()

    return ret


def setup_logger(logging_level, logging_queue, name='script'):
    """Setups logging for scripts."""
    logger = logging.getLogger('emLam')
    # Remove old handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[-1])

    if logging_level:
        # Set up root logger
        logger.setLevel(logging_level)
        qh = QueueHandler(logging_queue)
        qh.setLevel(logging_level)
        logger.addHandler(qh)
    else:
        # Don't log anything
        logger.setLevel(logging.CRITICAL + 1)

    logger = logging.getLogger('emLam.' + name)
    logger.setLevel(logger.parent.level)
    return logger


class AttrDict(dict):
    """Makes our life easier."""
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError('key {} missing'.format(key))
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError('key {} missing'.format(key))
        self[key] = value
