#!/usr/bin/env python3
"""Preprocessing steps for the Szeged corpus."""

from __future__ import absolute_import, division, print_function

from emLam import WORD, LEMMA, LEMMA_POS
from emLam.corpus.corpus_base import Preprocessing


class SzegedPreprocessing(Preprocessing):
    def __init__(self, keep_columns):
        self.keep_columns = keep_columns

    def preprocess(self, input_stream, output_stream):
        for line_no, line in enumerate(input_stream):
            if line == '\n':
                print(u'', file=output_stream)
            else:
                fields = line.rstrip('\n').split('\t')
                pos_start = fields[LEMMA_POS].find('[')
                if pos_start >= 0:
                    lemma = fields[LEMMA_POS][:pos_start]
                    pos = fields[LEMMA_POS][pos_start:]
                else:
                    # OTHER
                    lemma = fields[LEMMA]
                    pos = fields[LEMMA_POS]
                out_fields = [fields[WORD], lemma, pos]
                if self.keep_columns:
                    out_fields.extend(fields[2:-1])
                print(u'\t'.join(out_fields), file=output_stream)

    @classmethod
    def parser(cls, subparsers):
        parser = subparsers.add_parser('hu_szeged', help='Szeged Treebank')
        parser.add_argument('--keep-columns', '-k', action='store_true',
                            help='keep all columns. By default, the output files '
                                 'will only have 3 columns: word, lemma, POS.')
