#!/usr/bin/env python3
"""Base class for corpus preprocessors."""

from __future__ import absolute_import, division, print_function
import inspect

from emLam.utils import openall


class Preprocessing(object):
    """Base class for corpus preprocessors."""
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
    def instantiate(cls, **kwargs):
        """Instantiates the class from keyword arguments."""
        argspec = inspect.getargspec(cls.__init__).args
        corpus_args = {k: kwargs[k] for k in argspec[1:] if k in kwargs}
        return cls(**corpus_args)
