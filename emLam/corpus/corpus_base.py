#!/usr/bin/env python3
"""Base class for corpus preprocessors."""

from __future__ import absolute_import, division, print_function
import inspect

from emLam.utils import openall


class Preprocessing(object):
    """Base class for corpus preprocessors."""
    def initialize(self):
        """
        Initializes any resources the preprocessor might use. This should be
        done here so that it is only run once in a multiprocessing setting.
        """
        pass

    def cleanup(self):
        """The opposite of initialize()."""
        pass

    def preprocess(self, input_stream, output_stream):
        """
        (Pre)processes a corpus read from input_stream, and writes the output
        to output_stream.
        """
        raise NotImplementedError('preprocess() must be implemented')

    def preprocess_files(self, input_file, output_file):
        """Same as preprocess(), but on files."""
        with openall(input_file) as inf, openall(output_file, 'wt') as outf:
            return self.preprocess(inf, outf)

    @classmethod
    def parser(cls, subparsers):
        """
        This method adds a (n ArgumentParser) subparser to the group specified
        in the argument.
        """
        raise NotImplementedError('parser() must be implemented')

    @classmethod
    def instantiate(cls, process_id=0, **kwargs):
        """
        Instantiates the class from keyword arguments. The process_id (not a
        real pid, but an ordinal starting from 0) is there so that preprocessors
        that use external resources can "plan" accordingly.
        """
        argspec = inspect.getargspec(cls.__init__).args
        corpus_args = {k: kwargs[k] for k in argspec[1:] if k in kwargs}
        return cls(**corpus_args)
