#!/usr/bin/env python3
"""Preprocessor that looks for input that segfaults QunToken."""

from __future__ import absolute_import, division, print_function
import collections
import os
from tempfile import NamedTemporaryFile
from subprocess import Popen, PIPE

from emLam.corpus.preprocessor_base import Preprocessor


class QunTokenErrors(Preprocessor):
    """Preprocessor that looks for input that segfaults QunToken."""
    NAME = 'qt_errors'

    def __init__(self, quntoken, max_length=50000, tmp_dir='/run/shm/'):
        self.quntoken = quntoken
        self.max_length = max_length
        self.tmp_dir = tmp_dir
        if not os.path.isfile(self.quntoken):
            raise ValueError('No quntoken executable at {}'.format(quntoken))

    def preprocess(self, input_stream, output_stream):
        for text in self.__read_chunk(input_stream):
            sentences = text.strip().split('\n')
            if not self.__run_quntoken(sentences):
                queue = collections.deque([sentences])
                while queue:
                    self.__find_culprits(queue, output_stream)

    def __find_culprits(self, queue, output_stream):
        sentences = queue.popleft()
        k = len(sentences) // 2
        found = []
        if not self.__run_quntoken(sentences[:k]):
            found.append(sentences[:k])
        if not self.__run_quntoken(sentences[k:]):
            found.append(sentences[k:])
        if found:
            for culprit in found:
                if len(culprit) == 1:
                    print(culprit[0], file=output_stream)
                elif len(culprit) > 0:
                    queue.append(culprit)
        else:
            # Since we only get here if there is a problem, but neither half
            # is at blame, we can conclude that the error spans just those two
            # lines we split at. I am not sure such errors are possible;
            # just to be on the safe side...
            print(u' '.join(found[k-1:k+1]), file=output_stream)

    def __run_quntoken(self, sentences):
        if len(sentences) == 0:
            return True
        with NamedTemporaryFile(delete=False, dir=self.tmp_dir) as infile:
            for sentence in sentences:
                print(sentence.encode('utf-8'), file=infile)
        p = Popen([self.quntoken, infile.name], stdout=PIPE, stderr=PIPE)
        _, _ = p.communicate()
        os.unlink(infile.name)
        return p.returncode == 0

    def __read_chunk(self, input_stream):
        text = ''
        for txt in input_stream:
            text += txt
            if len(text) > self.max_length:
                yield text
                text = ''
        if text:
            yield text

    @classmethod
    def parser(cls, subparsers):
        """Adds the QunToken parameters to the subparser."""
        parser = subparsers.add_parser(cls.NAME, help='QunToken error logger')
        parser.add_argument('quntoken', help='the quntoken binary.')
        parser.add_argument('--max-length', '-l', type=int, default=10000,
                            help='the length of a text chunk to send to GATE '
                                 '[10000].')
        parser.add_argument('--tmp-dir', default='/dev/shm',
                            help='the directory used to create temporary '
                                 'files [/dev/shm].')
