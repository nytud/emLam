#!/usr/bin/env python3
"""Base class for corpus class."""

from __future__ import absolute_import, division, print_function

from emLam.corpus import get_all_preprocessors
from emLam.corpus.component import Component
from emLam.corpus.multi_file_writer import MultiFileWriter
from emLam.utils import openall


class Corpus(Component):
    """Base class for corpus objects."""
    def __init__(self, max_lines):
        super(Corpus, self).__init__()
        self.max_lines = max_lines

    def instream(self, input_file):
        return openall(input_file)

    def outstream(self, output_file):
        if self.max_lines:
            self.logger.info('MultiFileWriter for {}, {} lines / file.'.format(
                output_file, self.max_lines))
            return MultiFileWriter(output_file, self.max_lines)
        else:
            self.logger.info('Regular output file {}.'.format(output_file))
            return openall(output_file, 'wt')

    def files_to_streams(self, input_file, output_file):
        """Yields streams for the input and output file."""
        with self.instream(input_file) as inf, self.outstream(output_file) as outf:
            yield inf, outf


class GoldCorpus(Corpus):
    """Corpus that require no analysis, only formatting."""
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

    @classmethod
    def parser(cls, subparsers):
        """
        Adds CopyPreprocessor as the preprocessor. Descendants should not
        re-implement this method, but add their parameters in child_parser()
        instead.
        """
        parser = cls.child_parser(subparsers)
        parser.set_defaults(preprocessor='CopyPreprocessor')
        parser.set_defaults(max_lines=None)

    @classmethod
    def child_parser(cls, subparsers):
        """Takes the role of parser() in descendants."""
        raise NotImplementedError(
            'child_parser() must be implemented in class {}'.format(cls.__name__))


class RawCorpus(Corpus):
    """Corpus that require analysis. It adds a subparser for preprocessors."""
    @classmethod
    def parser(cls, subparsers):
        """
        Adds a subparser for preprocessor selection. Descendants should not
        re-implement this method, but add their parameters in child_parser()
        instead.
        """
        parser = cls.child_parser(subparsers)
        parser.add_argument('--max-lines', type=int,
                            help='if specified, the output will be written to '
                                 'a series of files, each (save the last one) '
                                 'approximately this long')
        pp_subparsers = parser.add_subparsers(
            title='Preprocessors',
            description='Lists the preprocessors available. For help on a '
                        'specific one, call the script with the '
                        '`<preprocessor> -h` arguments.',
            dest='preprocessor', help='the preprocessors available.')
        preprocessors = get_all_preprocessors()
        for _, pp_class in sorted(preprocessors.items()):
            pp_class.parser(pp_subparsers)

    @classmethod
    def child_parser(cls, subparsers):
        """Takes the role of parser() in descendants."""
        raise NotImplementedError(
            'child_parser() must be implemented in class {}'.format(cls.__name__))
