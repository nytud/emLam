#!/usr/bin/env python3
"""Corpus reader for the MNSZ2."""

from __future__ import absolute_import, division, print_function
#from builtins import str
from html.parser import HTMLParser  # Needs future, too
import io
import itertools
import re
import sys
import tarfile

from lxml import etree

from emLam.corpus.corpus_base import RawCorpus
from emLam.utils import openall


binary_type = str if sys.version_info < (3,) else bytes


class MNSZ2Corpus(RawCorpus):
    NAME = 'hu_mnsz2'

    def __init__(self, foreign=False):
        self.foreign = foreign

    def files_to_streams(self, input_file, output_file):
        with openall(input_file, 'rb') as inf:
            with openall(output_file, 'wt', encoding='utf-8') as outf:
                yield self.__extract_text(inf), outf

    def __extract_text(self, input_stream):
        """Extracts the text from the stupid xml."""
        recording, poem, in_p = False, False, False
        for event, node in etree.iterparse(input_stream, huge_tree=True,
                                           events=['start', 'end']):
            if 'event' == 'start':
                if node.tag == 'body':
                    recording = True
                elif node.tag == 'poem':
                    poem = True
                elif node.tag == 'div' and node.get('type') == 'poem':
                    poem = True
                elif node.tag == 'p' or node.tag == 'note':
                    in_p = node.get('lang') != 'foreign' or self.foreign
            else:
                if node.tag == 'body':
                    recording = False
                elif node.tag == 'poem':
                    poem = False
                elif node.tag == 'div' and node.get('type') == 'poem':
                    poem = False
                elif node.tag == 'head' and node.get('type') == 'attached':
                    pass  # TODO add text
                    yield self.__to_unicode(node.text)
                elif node.tag == 'p' or node.tag == 'note':
                    pass  # TODO add text
                    yield self.__to_unicode(node.text)
                    in_p = False

    @staticmethod
    def __to_unicode(s):
        print(type(s), binary_type)
        return s.decode('iso-8859-2') if type(s) == binary_type else s


    @classmethod
    def child_parser(cls, subparsers):
        parser = subparsers.add_parser('hu_mnsz2', help='Hungarian National Corpus')
        parser.add_argument('--foreign', '-f', action='store_true',
                            help='include paragraphs marked with lang=foreign')
        return parser
