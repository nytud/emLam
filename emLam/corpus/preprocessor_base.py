#!/usr/bin/env python3
"""Base class for corpus preprocessors."""

from __future__ import absolute_import, division, print_function

from emLam.corpus.component import Component


class Preprocessor(Component):
    """Base class for corpus preprocessors."""
    def initialize(self):
        """
        Initializes any resources the preprocessor might use. This should be
        done here so that it is only run once in a multiprocessing setting.
        """
        pass

    def preprocess(self, input_stream, output_stream):
        """
        (Pre)processes a corpus read from input_stream, and writes the output
        to output_stream.
        """
        raise NotImplementedError(
            'preprocess() must be implemented in class {}'.format(cls.__name__))

    def cleanup(self):
        """The opposite of initialize()."""
        pass


class CopyPreprocessor(Preprocessor):
    """This class simply copies the input to the output."""
    NAME = 'copy'
    DESCRIPTION = 'copies input lines to the output'

    def __init__(self):
        pass

    def preprocess(self, input_stream, output_stream):
        for line in input_stream:
            print(line, file=output_stream, end=u'')

    @classmethod
    def parser(cls, subparsers):
        parser = subparsers.add_parser(
            cls.NAME, help='Copies input lines to the output.')
