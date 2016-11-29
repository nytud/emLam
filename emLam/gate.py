#!/usr/bin/env python3
"""Parses the GATE output."""

from __future__ import absolute_import, division, print_function
from io import BytesIO
import re
from future.moves.urllib.parse import urlencode

from lxml import etree
import requests


_gate_modules = 'ML3-SSTok,HFSTLemm,ML3-PosLem-hfstcode'  # Opt: 'QT,...
_anas_p = re.compile(r'{ana=([^}]+), feats=([^}]+])(?:, incorrect=[^,}]+)?, lemma=([^}]+)?}')


def parse_with_gate(text, gate_url, anas=False):
    """Parses a text with a running GATE server."""
    r = requests.get(
        'http://{}/process?{}'.format(
            gate_url, urlencode({'run': _gate_modules,
                                 'text': text.encode('utf-8')}))
    )
    assert r.status_code == 200, \
        'No error, but unsuccessful request with text {}{}'.format(
            text[:100], '...' if len(text) > 0 else '')
    return parse_gate_xml(r.content, anas)


def parse_gate_xml_file(xml_file, get_anas=False):
    """Parses a GATE response from a file."""
    dom = etree.parse(xml_file)
    root = dom.getroot()
    text, sent = [], []
    for a in root.xpath('./AnnotationSet/Annotation[@Type!="SpaceToken"]'):
        if a.attrib['Type'] == 'Token':
            word = a.find('Feature[Name="string"]').find('Value').text
            lemma = a.find('Feature[Name="lemma"]').find('Value').text
            pos = a.find('Feature[Name="hfstana"]').find('Value').text
            tup = [word, lemma if lemma else word, pos]
            if get_anas:
                anas = a.find('Feature[Name="anas"]').find('Value').text or ''
                if anas:
                    for ana in anas.split(';'):
                        try:
                            a_ana, a_pos, a_lemma = _anas_p.match(ana).groups()
                        except:
                            print(u'Strange ana {} / {} {}'.format(ana, lemma, pos))
                            raise
                        if a_pos == pos and a_lemma == lemma:
                            # This is the right analysis
                            tup.append(a_ana)
                            break
                    else:
                        tup.append('')
                        # print('Could not find the analysis for: {} / {} {}'.format(
                        #         anas, lemma, pos))
                else:
                    tup.append('')
                    # print('Could not find anas for {} {}'.format(
                    #     lemma, pos))
            sent.append(tup)
        else:
            text.append(sent)
            sent = []
    return text


def parse_gate_xml(xml, anas=False):
    """Parses a GATE response from memory."""
    return parse_gate_xml_file(BytesIO(xml), anas)
