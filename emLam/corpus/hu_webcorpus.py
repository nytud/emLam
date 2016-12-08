#!/usr/bin/env python3
"""Preprocessing steps for the Szeged corpus."""

from __future__ import absolute_import, division, print_function
from html.parser import HTMLParser  # Needs future, too
import io
import itertools
import re
import tarfile

from emLam.corpus import RawCorpus
from emLam.utils import openall


class Webcorpus(RawCorpus):
    NAME = 'hu_webcorpus'
    rename_p = re.compile(r'\.tar(\.gz)$')

    def __init__(self, compressed=True, max_entities=0.2):
        self.compressed = compressed
        self.max_entities = max_entities
        self.html_parser = HTMLParser()

    def files_to_streams(self, input_file, output_file):
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
            inf = itertools.chain.from_iterable(input_stream(input_file))
            return map(self._read_input, inf), outf

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
    def child_parser(cls, subparsers):
        parser = subparsers.add_parser('hu_webcorpus', help='Hungarian Webcorpus')
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
                    # This should work, but alas, only from Python 3.3
                    # yield io.TextIOWrapper(member_f, encoding='iso-8859-2')
                    yield io.TextIOWrapper(io.BytesIO(member_f.read()), encoding='iso-8859-2')
                    yield [u'<s>\n']  # To separate files
                    member_f.close()

    @staticmethod
    def enumerate_file(corp_file):
        with openall(corp_file, encoding='iso-8859-2') as inf:
            yield inf
