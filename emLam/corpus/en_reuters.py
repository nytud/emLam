#!/usr/bin/env python3
"""Corpus reader for the new Reuters corpus."""

from __future__ import absolute_import, division, print_function
from six.moves.html_parser import HTMLParser
import sys

from lxml import etree

from emLam.corpus.corpus_base import RawCorpus
from emLam.utils import openall


binary_type = str if sys.version_info < (3,) else bytes


class Reuters(RawCorpus):
    NAME = 'en_reuters'
    DESCRIPTION = 'the new Reuters corpus'

    html_parser = HTMLParser()

    def __init__(self, max_lines):
        super(Reuters, self).__init__()

    def files_to_streams(self, input_file, output_file):
        """
        Reads input_file according to the corpus format (compressed / not). In
        the former case, modifies the output_file name so that the '.tar' part
        is not included in it.
        """
        with openall(input_file, 'rb') as inf:
            with openall(output_file, 'wt', encoding='utf-8') as outf:
                yield self.parse_xml(inf), outf

    def parse_xml(self, input_stream):
        in_text = False
        texts = []
        title = ''
        for event, node in etree.iterparse(input_stream, huge_tree=True,
                                           events=['start', 'end'],
                                           resolve_entities=False):
            if event == 'start':
                if node.tag == 'text':
                    in_text = True
            else:
                if node.tag == 'title':
                    title = self.__clean_text(node.text)
                elif node.tag == 'text':
                    in_text = False
                    if node.text:
                        texts = [self.__clean_text(node.text)] + texts
                else:
                    if in_text:
                        texts.append(self.__clean_text(node.text))
                        texts.append(self.__clean_text(node.tail))
        yield title + u'\n\n' + u''.join(texts).strip() + u'\n\n'

    @staticmethod
    def __clean_text(s):
        if not s:
            return u''
        else:
            s = s.decode('us-ascii') if type(s) == binary_type else s
            s = Reuters.html_parser.unescape(s)
            # s = s.replace('\n', ' ')
            return s
