#!/usr/bin/env python3
"""Base class for preprocessors that invoke a GATE server for analysis."""

from __future__ import absolute_import, division, print_function

from emLam.gate import parse_with_gate
from emLam.corpus.corpus_base import Preprocessing


class GATEPreprocessing(Preprocessing):
    def __init__(self, gate_url, max_length=10000):
        self.gate_url = gate_url
        self.max_length = max_length

    def preprocess(self, input_stream, output_stream):
        for parsed in self.parse_with_gate(input_stream):
            print(u'\n'.join(u'\t'.join(sent) for sent in parsed),
                  file=output_stream)

    def _parse_with_gate(self, input_stream):
        """
        Parses the input with GATE. This generator should be called from
        preprocess(). Reads from the input_stream via read_input(), and
        yields each parsed data chunk.
        """
        text = ''
        for txt in self.read_input(input_stream):
            text += txt
            if len(text) > self.max_length:
                yield parse_with_gate(text, self.gate_url)
                text = ''
        if text:
            yield parse_with_gate(text, self.gate_url)

    def _read_input(self, input_stream):
        """A generator that returns a chunk of text at a time."""
        raise NotImplementedError('read_input() must be implemented')

    @classmethod
    def _parser(cls, subparser):
        """
        Adds the GATE server parameters to the subparser. Should be called by
        subclasses in parser().
        """
        # TODO Use ArgumentParser(parents=)
        subparser.add_argument('--gate-url', '-G', required=True,
                               help='the url to the GATE server.')
        subparser.add_argument('--max-length', '-l', type=int, default=10000,
                               help='the length of a text chunk to send to GATE '
                                    '[10000].')
