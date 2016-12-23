#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Corpus reader for the MNSZ2."""

from __future__ import absolute_import, division, print_function
import os
import re
from six.moves.html_parser import HTMLParser
import sys

from lxml import etree

from emLam.corpus.corpus_base import RawCorpus
from emLam.corpus.hacks import split_for_qt
from emLam.utils import openall


binary_type = str if sys.version_info < (3,) else bytes


class MNSZ2Corpus(RawCorpus):
    NAME = 'hu_mnsz2'
    html_parser = HTMLParser()
    spaces = re.compile(r'[ \t]+')
    empty_lines = re.compile(r'\n[ \t]+\n')
    word_per_line = re.compile(r'off[/\\]jrc')

    def __init__(self, foreign=False):
        self.foreign = foreign

    def files_to_streams(self, input_file, output_file):
        if os.path.basename(input_file) == 'off_hu_jrc.xml':
            extractor = self.__extract_word_per_line
        else: 
            extractor = self.__extract_text
        with openall(input_file, 'rb') as inf:
            with openall(output_file, 'wt', encoding='utf-8') as outf:
                yield extractor(inf), outf

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
                if in_p:
                    if self.__is_content_node(node):
                        if node.text:
                            texts = [self.__clean_text(node.text)] + texts
                        if texts:
                            # LOL, __join() returns a list :D But see below
                            chunks = split_for_qt(self.__join(texts))
                            for chunk in chunks:
                                yield chunk
                                #print(chunk.encode('utf-8'))
                            texts = []
                        in_p = False
                    else:
                        texts.append(self.__clean_text(node.text))
                        texts.append(self.__clean_text(node.tail))

    def __extract_word_per_line(self, input_stream):
        """Extracts the text from the even more stupid xml."""
        recording, in_p = False, False
        texts = []
        for event, node in etree.iterparse(input_stream, huge_tree=True,
                                           events=['start', 'end'],
                                           resolve_entities=False):
            if event == 'start':
                if node.tag == 'div':
                    texts = []
            else:
                if node.tag == 'p':
                    texts += self.__clean_text(node.text).split()
                elif node.tag == 'div':
                    #print(u' '.join(texts).encode('utf-8') + '\n')
                    for chunk in split_for_qt(u' '.join(texts)):
                        yield chunk
                        #print(chunk.encode('utf-8')[:-1])

    @staticmethod
    def __join(texts):
        """
        Keeps the newlines in the text, so that quntoken doesn't choke on the
        100k character-long lines.
        """
        text = MNSZ2Corpus.empty_lines.sub(u'\n', u' '.join(texts))
        # Empty line between paragraphs to prevent QunToken from choking
        return MNSZ2Corpus.spaces.sub(u' ', text)

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
