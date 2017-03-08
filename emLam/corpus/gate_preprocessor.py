#!/usr/bin/env python3
"""A preprocessor that invokes a GATE server for analysis."""

from __future__ import absolute_import, division, print_function

from emLam.gate import Gate
from emLam.corpus.preprocessor_base import Preprocessor


class GATEPreprocessor(Preprocessor):
    """A preprocessor that invokes a GATE server for analysis."""
    NAME = 'GATE'
    DESCRIPTION = 'GATE preprocessor'

    def __init__(self, gate_props, max_length=10000, restart_every=0, anas='no'):
        super(GATEPreprocessor, self).__init__()
        self.gate_props = gate_props
        self.max_length = max_length
        self.restart_every = restart_every
        self.anas = anas
        self.gate = None

    def initialize(self):
        """
        Gate is initialized here so that it is only created once, in the
        processing process, not in the main one.
        """
        if not self.gate:
            self.gate = Gate(self.gate_props, self.restart_every)

    def cleanup(self):
        if self.gate:
            del self.gate
        self.gate = None

    def preprocess(self, input_stream, output_stream):
        for chunk, parsed in enumerate(self.__parse_with_gate(input_stream)):
            if chunk > 0:
                # Preserve the empty sentence separator line between chunks
                print(u'', file=output_stream)
            print(u'\n\n'.join(u'\n'.join(u'\t'.join(token) for token in sent)
                               for sent in parsed),
                  file=output_stream)

    def __parse_with_gate(self, input_stream):
        """
        Parses the input with GATE. This generator is called from
        preprocess(). Reads from the input_stream a batch of sentences, and
        yields the parsed data chunk.
        """
        text = ''
        for txt in input_stream:
            text += txt
            if len(text) > self.max_length:
                yield self.gate.parse(text, self.anas)
                text = ''
        if text:
            yield self.gate.parse(text, anas)

    @classmethod
    def instantiate(cls, process_id=1, **kwargs):
        mod_args = dict(kwargs)
        mod_args['gate_props'] = kwargs['gate_props'].replace('%', str(process_id))
        if process_id > 1 and mod_args['gate_props'] == kwargs['gate_props']:
            raise ValueError('At least as many gate servers must be '
                             'specified as there are processes.')
        return super(GATEPreprocessor, cls).instantiate(process_id,
                                                        **mod_args)
