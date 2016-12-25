#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Reads a regular text corpus where the relevant units (usually paragraphs)
are separated by newlines.
"""

from __future__ import absolute_import, division, print_function
from contextlib import contextmanager

from emLam.corpus.corpus_base import RawCorpus
from emLam.corpus.hacks import split_for_qt

class TextCorpus(RawCorpus):
    NAME = 'text'

    @contextmanager
    def instream(self, input_file):
        def __instream():
            with super(TextCorpus, self).instream(input_file) as inf:
                lines = []
                for l in inf:
                    line = l.rstrip('\n')
                    if len(line) > 0:
                        lines.append(line)
                    else:
                        if lines:
                            for chunk in split_for_qt(u'\n'.join(lines)):
                                yield chunk
                        lines = []
                else:
                    if lines:
                        for chunk in split_for_qt(u'\n'.join(lines)):
                            yield chunk
        yield __instream()

    @classmethod
    def child_parser(cls, subparsers):
        parser = subparsers.add_parser(cls.NAME,
                                       help='Newline-separated text corpus')
        return parser
