#!/usr/bin/env python3
"""Corpus reader for the new Reuters corpus."""

from __future__ import absolute_import, division, print_function

from lxml import etree

from emLam.corpus.corpus_base import RawCorpus
from emLam.utils import openall


class Webcorpus(RawCorpus):
    NAME = 'en_reuters'

    def __init__(self, max_lines):
        super(Webcorpus, self).__init__()

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
                                           events=['start', 'end']):
            if event == 'start':
                if node.tag == 'text':
                    in_text = True
            else:
                if node.tag == 'title':
                    title = node.text or ''
                elif node.tag == 'text':
                    in_text = False
                    if node.text:
                        texts = [node.text or ''] + texts
                else:
                    if in_text:
                        texts.append(node.text or '')
                        texts.append(node.tail or '')
        yield title + '\n\n' + ''.join(texts).decode('us-ascii').strip() + '\n\n'

    @classmethod
    def child_parser(cls, subparsers):
        parser = subparsers.add_parser('en_reuters',
                                       help='the new Reuters corpus')
        return parser
