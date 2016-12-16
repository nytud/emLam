#!/usr/bin/env python3
"""Base class for corpus class."""

from __future__ import absolute_import, division, print_function

from emLam.corpus import get_all_preprocessors
from emLam.corpus.component import Component
from emLam.utils import openall


class Corpus(Component):
    """Base class for corpus objects."""
    def files_to_streams(self, input_file, output_file):
        """Yields streams for the input and output file."""
        with openall(input_file) as inf, openall(output_file, 'wt') as outf:
            yield inf, outf


class GoldCorpus(Corpus):
    """Corpus that require no analysis, only formatting."""
    def files_to_streams(self, input_file, output_file):
        """Converts the input stream into the expected format."""
        for inf, outf in super(GoldCorpus, self).files_to_streams(input_file,
                                                                  output_file):
            yield map(self.convert_input, inf), outf

    @classmethod
    def parser(cls, subparsers):
        """
        Adds CopyPreprocessor as the preprocessor. Descendants should not
        re-implement this method, but add their parameters in child_parser()
        instead.
        """
        parser = cls.child_parser(subparsers)
        parser.set_defaults(preprocessor='CopyPreprocessor')

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
