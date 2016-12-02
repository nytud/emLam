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
        for chunk, parsed in enumerate(self._parse_with_gate(input_stream)):
            if chunk > 0:
                # Preserve the empty sentence separator line between chunks
                print(u'', file=output_stream)
            print(u'\n\n'.join(u'\n'.join(u'\t'.join(token) for token in sent)
                               for sent in parsed),
                  file=output_stream)

    def _parse_with_gate(self, input_stream):
        """
        Parses the input with GATE. This generator should be called from
        preprocess(). Reads from the input_stream via _read_input(), and
        yields each parsed data chunk.
        """
        text = ''
        for txt in self._read_input(input_stream):
            text += txt
            if len(text) > self.max_length:
                yield parse_with_gate(text, self.gate_url)
                text = ''
        if text:
            yield parse_with_gate(text, self.gate_url)

    def _read_input(self, input_stream):
        """A generator that returns a chunk of text at a time."""
        raise NotImplementedError('_read_input() must be implemented')

    @classmethod
    def _parser(cls, subparser):
        """
        Adds the GATE server parameters to the subparser. Should be called by
        subclasses in parser().
        """
        # TODO Use ArgumentParser(parents=)
        subparser.add_argument('--gate-url', '-G', required=True, action='append',
                               help='the <host>:<port> of the GATE server. If '
                                    'multiple processes are used, at least as '
                                    'many servers should be given as there are '
                                    'processes.')
        subparser.add_argument('--max-length', '-l', type=int, default=10000,
                               help='the length of a text chunk to send to GATE '
                                    '[10000].')

    @classmethod
    def instantiate(cls, process_id=1, **kwargs):
        try:
            mod_args = dict(kwargs)
            mod_args['gate_url'] = kwargs['gate_url'][process_id]
            return super(GATEPreprocessing, cls).instantiate(process_id,
                                                             **mod_args)
        except:
            raise ValueError('At least as many gate servers must be specified '
                             'as there are processes.')
