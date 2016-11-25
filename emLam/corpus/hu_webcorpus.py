#!/usr/bin/env python3
"""Preprocessing steps for the Szeged corpus."""

from __future__ import absolute_import, division, print_function
from html.parser import HTMLParser  # Needs future, too
import re
import tarfile

from emLam.corpus.gate_corpus import GATEPreprocessing
from emLam.utils import openall


class WebcorpusPreprocessing(GATEPreprocessing):
    rename_p = re.compile(r'\.tar(\.gz)$')

    def __init__(self, gate_url, max_length=10000, compressed=True,
                 max_entities=0.2):
        super(WebcorpusPreprocessing, self).__init__(gate_url, max_length)
        self.compressed = compressed
        self.max_entities = max_entities
        self.html_parser = HTMLParser()

    def preprocess_files(self, input_file, output_file):
        """
        Reads input_file according to the corpus format (compressed / not). In
        the former case, modifies the output_file name so that the '.tar' part
        is not included in it.
        """
        if self.compressed:
            output_file = self.rename_p.sub(r'\1', output_file)
            input_stream = self.enumerate_tar
        else:
            input_stream = self.enumerate_file
        with openall(output_file, 'wt', encoding='utf-8') as outf:
            for inf in input_stream(input_file):
                self.preprocess(inf, outf)

    def _read_input(self, input_stream):
        """A generator that returns a chunk of text at a time."""
        for line in input_stream:
            if line.startswith(u'<s>'):
                text = line[3:]
                amps = text.count(u'&')
                if amps > 0:
                    text = self.html_parser.unescape(text)
                    entities = text.count(u'&') - amps
                    if entities / float(len(text)) > self.max_entities:
                        # Skip sentence if too many entities (i.e. foreign script)
                        continue
                yield text

    @classmethod
    def parser(cls, subparsers):
        parser = subparsers.add_parser('hu_webcorpus', help='Hungarian Webcorpus')
        super(WebcorpusPreprocessing, cls)._parser(parser)
        parser.add_argument('--uncompressed', '-u', action='store_false',
                            dest='compressed',
                            help='the source directory contains the uncompressed '
                                 'corpus (1 file per document). Not recommended; '
                                 'by default, the preprocessor expects '
                                 'tar.gz files.')
        parser.add_argument('--max-entities', '-m', type=float, default=0.2,
                            help='the HTML entity / characters ratio above '
                                 'which a sentence is discarded.')

    @staticmethod
    def enumerate_tar(archive):
        if not tarfile.is_tarfile(archive):
            return  # TODO
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.isfile():
                    member_f = tf.extractfile(member.name)
                    yield (line.decode('iso-8859-2') for line in member_f)
                    member_f.close()

    @staticmethod
    def enumerate_file(corp_file):
        with openall(corp_file, encoding='iso-8859-2') as inf:
            yield inf
