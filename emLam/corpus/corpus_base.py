#!/usr/bin/env python3
"""Base class for corpus class."""

from __future__ import absolute_import, division, print_function

from emLam.corpus.component import Component
from emLam.corpus.multi_file_writer import MultiFileWriter
from emLam.utils import openall


class Corpus(Component):
    """Base class for corpus objects."""
    def __init__(self, max_lines=None):
        super(Corpus, self).__init__()
        self.max_lines = max_lines

    def instream(self, input_file):
        return openall(input_file)

    def outstream(self, output_file):
        if self.max_lines:
            return MultiFileWriter(output_file, self.max_lines)
        else:
            return openall(output_file, 'wt')

    def files_to_streams(self, input_file, output_file):
        """Yields streams for the input and output file."""
        with self.instream(input_file) as inf, self.outstream(output_file) as outf:
            yield inf, outf


class GoldCorpus(Corpus):
    """Corpus that requires no analysis, only formatting."""
    def files_to_streams(self, input_file, output_file):
        """Converts the input stream into the expected format."""
        for inf, outf in super(GoldCorpus, self).files_to_streams(input_file,
                                                                  output_file):
            yield self.convert_input(inf), outf

    def convert_input(self, input_stream):
        """Converts the input to tsv format."""
        raise NotImplementedError(
            'convert_input() must be implemented in class {}'.format(
                self.__class__.__name__))


class RawCorpus(Corpus):
    """Corpus that requires analysis. For code organization purposes."""
    pass
