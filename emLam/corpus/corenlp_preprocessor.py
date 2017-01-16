#!/usr/bin/env python3
"""A preprocessor that invokes a Stanford CoreNLP server for analysis."""

from __future__ import absolute_import, division, print_function

from emLam.corenlp import CoreNLP
from emLam.corpus.preprocessor_base import Preprocessor


class CoreNlpPreprocessor(Preprocessor):
    """A preprocessor that invokes a Stanford CoreNLP server for analysis."""
    NAME = 'CoreNLP'

    def __init__(self, corenlp_props, max_length=10000):
        self.corenlp_props = corenlp_props
        self.max_length = max_length
        self.corenlp = None

    def initialize(self):
        """
        The CoreNLP server is initialized here so that it is only created once,
        in the processing process, not in the main one.
        """
        if not self.corenlp:
            self.corenlp = CoreNLP(self.corenlp_props)

    def cleanup(self):
        if self.corenlp:
            del self.corenlp
        self.corenlp = None

    def preprocess(self, input_stream, output_stream):
        for chunk, parsed in enumerate(self.__parse_with_corenlp(input_stream)):
            if chunk > 0:
                # Preserve the empty sentence separator line between chunks
                print(u'', file=output_stream)
            print(u'\n\n'.join(u'\n'.join(u'\t'.join(token) for token in sent)
                               for sent in parsed),
                  file=output_stream)

    def __parse_with_corenlp(self, input_stream):
        """
        Parses the input with CoreNLP. This generator is called from
        preprocess(). Reads from the input_stream a batch of sentences, and
        yields the parsed data chunk.
        """
        text = ''
        for txt in input_stream:
            text += txt
            if len(text) > self.max_length:
                yield self.corenlp.parse(text)
                text = ''
        if text:
            yield self.corenlp.parse(text)

    @classmethod
    def parser(cls, subparsers):
        """Adds the GATE server parameters to the subparser."""
        # Double % because argparse uses 'str % params'-style formatting
        parser = subparsers.add_parser(cls.NAME, help='CoreNLP preprocessor')
        parser.add_argument('--corenlp-props', '-G', required=True,
                            help='a Python ConfigParser-style configuration '
                                 'file that contains the parameters passed to '
                                 'the CoreNLP server. If there is a "%%" in '
                                 'the file name, it will be replaced by the '
                                 'id of the current process. This feature '
                                 'should be used in a multiprocessing '
                                 'setting.')
        parser.add_argument('--max-length', '-l', type=int, default=10000,
                            help='the length of a text chunk sent to the server '
                                 '[10000].')

    @classmethod
    def instantiate(cls, process_id=1, **kwargs):
            mod_args = dict(kwargs)
            mod_args['corenlp_props'] = kwargs['corenlp_props'].replace(
                '%', str(process_id))
            if process_id > 1 and mod_args['corenlp_props'] == kwargs['corenlp_props']:
                raise ValueError('At least as many gate servers must be '
                                 'specified as there are processes.')
            return super(CoreNlpPreprocessor, cls).instantiate(process_id,
                                                               **mod_args)
