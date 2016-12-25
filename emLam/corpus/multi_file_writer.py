#!/usr/bin/env python3
"""Defines ways to "convert" a file name to an input/output stream."""

from __future__ import absolute_import, division, print_function
from builtins import range
from io import TextIOBase
import math
import os

from emLam.utils import allname, openall

class MultiFileWriter(TextIOBase):
    def __init__(self, file_name, max_lines, wait_for_empty=True):
        self.file_name = file_name
        self.max_lines = max_lines
        self.wait_for_empty = wait_for_empty
        self.index = 1
        self.lines = 0
        self.f = openall(self.__get_file_name(), 'wt')

    def __get_file_name(self, index=None, digits=None):
        basename, extension = allname(self.file_name)
        ext = extension if extension else ''
        num_format = '{{:{}d}}'.format(digits) if digits else '{}'
        index_str = num_format.format(self.index if index is None else index)
        return '{}-{}{}'.format(basename, index_str, ext)

    def close(self):
        self.f.close()

    def fileno(self):
        return self.f.fileno()

    def flush(self):
        return self.f.flush()

    def write(self, s):
        for line in s.splitlines():
            self.f.write(line)
            self.f.write(u'\n')
            self.lines += 1
            if self.lines >= self.max_lines and (
                    not self.wait_for_empty or line == ''):
                self.__new_file()

    def __new_file(self):
        """
        Opens the next file, resets the line counter and renames all previous
        files if we need a new digit.
        """
        self.f.close()
        digits = int(math.log10(self.index))
        self.index += 1
        new_digits = int(math.log10(self.index))
        if new_digits > digits:
            for i in range(1, self.index):
                os.rename(self.__get_file_name(self.index - 1, digits),
                          self.__get_file_name(self.index, new_digits))
        self.f = openall(self.__get_file_name(), 'wt')
        self.lines = 0

    def isatty(self):
        return False

    def readable(self):
        return False

    def seekable(self):
        return False

    def writable(self):
        return True
