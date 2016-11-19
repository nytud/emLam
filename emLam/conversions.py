#!/usr/bin/env python3
"""Conversion functions from tsv to the final token format."""

from __future__ import absolute_import, division, print_function
import inspect
from operator import itemgetter
import re
import sys

from emLam import WORD, LEMMA, POS


_univ_re = re.compile(r'([[][^[]+[]])')
_univ_pos_re = re.compile(r'/([^]]+)\]')
_univ_hyph_re = re.compile(r'^\[Hyph:[^]]+\]$')
# To fix erroneous tagging
_univ_map = {
    '[N]': '[/N]',
    '[V]': '[/V]',
    '[Num]': '[/Num]',
    '[_Mod]': '[_Mod/V]',
}


def field_word(fields):
    return [fields[WORD]]


def field_lemma(fields):
    return [fields[LEMMA]]


def field_lemma_inf(fields):
    """
    For universal dependencies returned by GATE: ~lemmad_krs.
    Fields: word, lemma, pos (, ...).
    """
    return _field_lemma_inf(fields)


def field_lemma_inf_with_pos(fields):
    """
    For universal dependencies returned by GATE: ~lemmad_krs, but the POS tag
    (in the narrow sense, i.e. /N, etc.) is appended to the lemma as well.
    Fields: word, lemma, pos (, ...).
    """
    return _field_lemma_inf(fields, True)


def _field_lemma_inf(fields, keep_pos=False):
    lemma = fields[LEMMA]
    univ = fields[POS]
    word_ana = [lemma]
    infl_ana = []
    if '[' in univ and not _univ_hyph_re.match(univ):  # not OTHER, maybe sth else too
        form = 0
        parts = _univ_re.findall(univ)
        for part in parts:
            part = _univ_map.get(part, part)
            if part == '[/Supl]':  # Delete 'leg', but remember it
                form = 2
            if part.startswith('[/') and keep_pos:  # POS tag
                word_ana[-1] += part
            elif part.startswith('[_Comp'):
                form = max(form, 1)
            elif part.startswith('[_'):
                word_ana[-1] += part
            else:
                infl_ana.append(part)
        if form == 1:
            word_ana.append('[Comp]')
        elif form == 2:
            word_ana.append('[Supl]')
    return word_ana + infl_ana


def _get_field_selectors():
    return {
        name: obj for name, obj in inspect.getmembers(sys.modules[__name__])
        if name.startswith('field_') and (
            (inspect.isfunction(obj) and obj.__module__ == __name__) or
            isinstance(obj, itemgetter)
        )
    }


def get_field_function(field):
    return _get_field_selectors()['field_' + field]


def list_field_functions():
    return [f[6:] for f in _get_field_selectors().keys()]
