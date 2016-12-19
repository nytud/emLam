#!/usr/bin/env python3
"""Corpus reader for the MNSZ2."""

from __future__ import absolute_import, division, print_function
import re
from six.moves.html_parser import HTMLParser
import sys

from lxml import etree

from emLam.corpus.corpus_base import RawCorpus
from emLam.utils import openall


binary_type = str if sys.version_info < (3,) else bytes


class MNSZ2Corpus(RawCorpus):
    NAME = 'hu_mnsz2'
    html_parser = HTMLParser()
    spaces = re.compile(r'[ \t]+')
    empty_lines = re.compile(r'\n[ \t]+\n')

    def __init__(self, foreign=False):
        self.foreign = foreign

    def files_to_streams(self, input_file, output_file):
        with openall(input_file, 'rb') as inf:
            with openall(output_file, 'wt', encoding='utf-8') as outf:
                yield self.__extract_text(inf), outf

    def __extract_text(self, input_stream):
        """Extracts the text from the stupid xml."""
        recording, poem, in_p = False, False, False
        texts = []
        for event, node in etree.iterparse(input_stream, huge_tree=True,
                                           events=['start', 'end'],
                                           resolve_entities=False):
            if event == 'start':
                if node.tag == 'body':
                    recording = True
                elif self.__is_poem(node):
                    poem = True
                elif self.__is_content_node(node) and recording and not poem:
                    in_p = node.get('lang') != 'foreign' or self.foreign
            else:
                if node.tag == 'body':
                    recording = False
                elif self.__is_poem(node):
                    poem = False
                elif self.__is_content_node(node) and recording and not poem:
                    if node.text:
                        texts = [self.__clean_text(node.text)] + texts
                    if texts:
                        #yield u' '.join(texts) + u'\n'
                        yield self.__join(texts)
                        texts = []
                    in_p = False
                elif in_p:
                    texts.append(self.__clean_text(node.text))
                    texts.append(self.__clean_text(node.tail))

    @staticmethod
    def __join(texts):
        """
        Keeps the newlines in the text, so that quntoken doesn't choke on the
        100k character-long lines.
        """
        text = MNSZ2Corpus.empty_lines.sub(u'\n', u' '.join(texts))
        text = MNSZ2Corpus.spaces.sub(u' ', text)
        return text

    @staticmethod
    def __is_poem(node):
        if node.tag == 'poem':
            return True
        elif node.tag == 'div' and node.get('type') == 'poem':
            return True
        return False

    @staticmethod
    def __is_content_node(node):
        if node.tag == 'p' or node.tag == 'note':
            return True
        elif node.tag == 'head' and node.get('type') == 'attached':
            return True
        return False

    @staticmethod
    def __clean_text(s):
        if not s:
            return u''
        else:
            s = s.decode('iso-8859-2') if type(s) == binary_type else s
            s = MNSZ2Corpus.html_parser.unescape(s)
            #s = s.replace('\n', ' ')
            return s.strip()

    @classmethod
    def child_parser(cls, subparsers):
        parser = subparsers.add_parser('hu_mnsz2', help='Hungarian National Corpus')
        parser.add_argument('--foreign', '-f', action='store_true',
                            help='include paragraphs marked with lang=foreign')
        return parser
