#!/usr/bin/env python3
"""Base class for corpus preprocessors."""

from __future__ import absolute_import, division, print_function

class Preprocessing(object):
    """Base class for corpus preprocessors."""
    def preprocess(self, input_stream, output_stream):
        """
        (Pre)processes a corpus read from input_stream, and writes the output
        to output_stream.
        """
        raise NotImplementedError('preprocess() must be implemented')

    @classmethod
    def parser(cls, subparsers):
        """
        This method adds a (n ArgumentParser) subparser to the group specified
        in the argument.
        """
        raise NotImplementedError('parser() must be implemented')
