#!/usr/bin/env python3
"""Base class for preprocessors that invoke a GATE server for analysis."""

from __future__ import absolute_import, division, print_function

from emLam.gate import Gate
from emLam.corpus.corpus_base import Preprocessing


class GATEPreprocessing(Preprocessing):
    def __init__(self, gate_props, max_length=10000, restart_every=0):
        self.gate_props = gate_props
        self.max_length = max_length
        self.restart_every = restart_every
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
                yield self.gate.parse(text)
                text = ''
        if text:
            yield self.gate.parse(text)

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
        # Double % because argparse uses 'str % params'-style formatting
        subparser.add_argument('--gate-props', '-G', required=True,
                               help='the hunlp-GATE property file used to '
                                    'start the server. If there is a "%%" in '
                                    'the file name, it will be replaced by the '
                                    'id of the current process. This feature '
                                    'should be used in a multiprocessing '
                                    'setting.')
        subparser.add_argument('--max-length', '-l', type=int, default=10000,
                               help='the length of a text chunk to send to GATE '
                                    '[10000].')
        subparser.add_argument('--restart-every', '-r', metavar='N', type=int,
                               default=0,
                               help='restart the GATE server after every N '
                                    'sentences to avoid OutOfMemoryException '
                                    '[0].')

    @classmethod
    def instantiate(cls, process_id=1, **kwargs):
            mod_args = dict(kwargs)
            mod_args['gate_props'] = kwargs['gate_props'].replace('%', str(process_id))
            if process_id > 1 and mod_args['gate_props'] == kwargs['gate_props']:
                raise ValueError('At least as many gate servers must be '
                                 'specified as there are processes.')
            return super(GATEPreprocessing, cls).instantiate(process_id,
                                                             **mod_args)
