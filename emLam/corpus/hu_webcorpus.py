#!/usr/bin/env python3
"""Corpus reader for the Hungarian Webcorpus."""

from __future__ import absolute_import, division, print_function
from contextlib import contextmanager
from html.parser import HTMLParser  # Needs future, too
import io
import itertools
import re
import tarfile
from unicodedata import category

from emLam.corpus.corpus_base import RawCorpus
from emLam.utils import openall


class Webcorpus(RawCorpus):
    NAME = 'hu_webcorpus'
    DESCRIPTION = 'Hungarian Webcorpus'

    rename_p = re.compile(r'\.tar(\.gz)$')

    def __init__(self, max_lines, compressed=True, max_entities=0.2):
        super(Webcorpus, self).__init__(max_lines)
        self.compressed = compressed
        self.max_entities = max_entities
        self.html_parser = HTMLParser()

    @contextmanager
    def instream(self, input_file):
        if self.compressed:
            input_stream = self.enumerate_tar
        else:
            input_stream = self.enumerate_file
        inf = itertools.chain.from_iterable(input_stream(input_file))
        yield self.__read_sentence(inf)

    def outstream(self, output_file):
        """Removes the 'tar' from the name of the output file, if compressed."""
        if self.compressed:
            output_file = self.rename_p.sub(r'\1', output_file)
        return super(Webcorpus, self).outstream(output_file)

#    def files_to_streams(self, input_file, output_file):
#        """
#        Reads input_file according to the corpus format (compressed / not). In
#        the former case, modifies the output_file name so that the '.tar' part
#        is not included in it.
#        """
#        if self.compressed:
#            output_file = self.rename_p.sub(r'\1', output_file)
#            input_stream = self.enumerate_tar
#        else:
#            input_stream = self.enumerate_file
#        with openall(output_file, 'wt', encoding='utf-8') as outf:
#            inf = itertools.chain.from_iterable(input_stream(input_file))
#            yield self.__read_sentence(inf), outf

    def __read_sentence(self, input_stream):
        """Returns a sentence a time, cleaned of HTML entities."""
        for line in input_stream:
            if line.startswith(u'<s>'):
                text = line[3:].strip()
                orig_text = text
                amps = text.count(u'&')
                if amps > 0:
                    text = self.html_parser.unescape(text)
                    entities = amps - text.count(u'&')
                    self.logger.debug(
                        u'Entities: {}, amps: {}, len: {}; ratio: {} => {}: {} ({})'.format(
                            entities, amps, len(text), entities / float(len(text)),
                            entities / float(len(text)) > self.max_entities,
                            text.strip(), orig_text.strip()).encode('utf-8'))
                    if len(text) >entities / float(len(text)) > self.max_entities:
                        # Skip sentence if too many entities (i.e. foreign script)
                        continue
                clean_text = self.__clean_text(text)
                if clean_text != text:
                    self.logger.debug(u'Filtered text: `{}` -> `{}`'.format(
                        text, clean_text))
                yield clean_text + u'\n'

    @staticmethod
    def __clean_text(text):
        """Cleans the text of all unicode shenanigans."""
        clean_text = []
        for c in text.replace(u'\t', u' '):
            # Get rid of extended Unicode characters, which are most likely
            # there by accident
            if ord(c) > 65535:
                continue
            cat = category(c)
            # Control characters are also bugs in the corpus
            if cat.startswith('C'):
                continue
            clean_text.append(c if not cat.startswith('Z') else ' ')
        return u''.join(clean_text)

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
