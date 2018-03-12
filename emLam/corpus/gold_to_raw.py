#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Converts a gold corpus to a raw one: connects the tokens in the selected
field to running text.
"""

from __future__ import absolute_import, division, print_function

from emLam.corpus.preprocessor_base import Preprocessor


class GoldToRaw(Preprocessor):
    """
    Converts a gold corpus to a raw one: connects the tokens in the selected
    field to running text.
    """
    NAME = 'GoldToRaw'
    DESCRIPTION = 'Converts a gold corpus to raw'

    def __init__(self, field=0):
        self.field = field

    def preprocess(self, input_stream, output_stream):
        words = []
        for token in input_stream:
            word = token.rstrip().split('\t')[self.field]
            if word:
                words.append(word)
            else:
                print(' '.join(words), file=output_stream)
                words = []
