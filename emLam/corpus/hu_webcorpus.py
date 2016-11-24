#!/usr/bin/env python3
"""Preprocessing steps for the Szeged corpus."""

from __future__ import absolute_import, division, print_function
from html.parser import HTMLParser  # Needs future, too
import re
import tarfile

from emLam.corpus.corpus_base import Preprocessing
from emLam.utils import openall


class WebcorpusPreprocessing(Preprocessing):
    rename_p = re.compile(r'\.tar(\.gz)$')

    def __init__(self, compressed=True):
        if compressed:
            self.input_stream = self.enumerate_tar
        else:
            self.input_stream = self.enumerate_file

    def preprocess_files(self, input_file, output_file):
        """
        Reads input_file according to the corpus format (compressed / not). In
        the former case, modifies the output_file name so that the '.tar' part
        is not included in it.
        """
        if compressed:
            output_file = self.rename_p.sub(r'\1', output_file)
        with openall(output_file, 'wt') as outf:
            for inf in self.input_stream(input_file):
                self.preprocess(inf, outf)

    def preprocess(self, input_stream, output_stream):
        for line in input_stream:
            if line.startswith('<s>'):


    @classmethod
    def parser(cls, subparsers):
        parser = subparsers.add_parser('hu_webcorpus', help='Hungarian Webcorpus')
        parser.add_argument('--uncompressed', '-u', action='store_false',
                            dest='compressed',
                            help='the source directory contains the uncompressed '
                                 'corpus (1 file per document). Not recommended; '
                                 'by default, the preprocessor expects '
                                 'tar.gz files.')

    @staticmethod
    def enumerate_tar(archive):
        if not tarfile.is_tarfile(archive):
            return  # TODO
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.isfile():
                    member_f = tf.extractfile(member.name)
                    yield member_f
                    member_f.close()

    @staticmethod
    def enumerate_file(corp_file):
        with openall(corp_file) as inf:
            yield inf
