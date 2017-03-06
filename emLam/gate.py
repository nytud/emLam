#!/usr/bin/env python3
"""Manages a GATE server."""

from __future__ import absolute_import, division, print_function
from builtins import range
from configparser import RawConfigParser  # Should work under 2.7, too
from future.moves.urllib.parse import urlencode
from io import open, BytesIO, StringIO
import logging
import os
import re
from subprocess import Popen
import time

from lxml import etree
import pyparsing as pp
import requests

from emLam import WORD, LEMMA


_anas_p = re.compile(r'{ana=([^}]+), feats=([^}]+])(?:, incorrect=[^,}]+)?, lemma=([^}]+)?}')


class GateError(Exception):
    pass


class Gate(object):
    """hunlp-GATE interface object."""
    def __init__(self, gate_props, restart_every=None,
                 modules='QT,HFSTLemm,ML3-PosLem-hfstcode'):
        """
        gate_props is the name of the GATE properties file. It is suppoesed to
        be in the hunlp-GATE directory.

        If restart_every is specified, the GATE server is restarted after that
        many sentences (counted from the parsed output). This is necessary
        because it (hunlp-)GATE leaking memory like there is no tomorrow.
        """
        # Opt: ML3-SSTok
        self.logger = logging.getLogger('emLam.GATE')
        self.gate_props = gate_props
        self.gate_dir = os.path.dirname(gate_props)
        self.gate_url = self._gate_url()
        self.restart_every = restart_every
        self.modules = modules
        self.server = None
        self.parsed = 0
        self.__start_server()

    def __del__(self):
        # See also http://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
        # for ideas on how to replace __del__ with something better (?)
        self.__stop_server()

    def _gate_url(self):
        """Assembles the GATE url from the properties."""
        cp = RawConfigParser({'host': 'localhost', 'port': 8000})
        with open(self.gate_props) as inf:
            cp.readfp(StringIO(u'[GATE]\n' + inf.read()))
        return '{}:{}'.format(cp.get('GATE', 'host'), cp.get('GATE', 'port'))

    def __start_server(self):
        self.logger.debug('Starting server {}...'.format(self.gate_props))
        # TODO eat the server's output -- in this case, there is no need to wait
        self.server = Popen(['./gate-server.sh', self.gate_props],
                            cwd=self.gate_dir)
        self.parsed = 0
        time.sleep(10)
        self.logger.info('Started server {}'.format(self.gate_props))

    def __stop_server(self):
        self.logger.debug('Stopping server {}?'.format(self.gate_props))
        if self.server:
            self.logger.debug('Stopping server {}'.format(self.gate_props))
            try:
                requests.post('http://{}/exit'.format(self.gate_url))
            except:
                pass
            self.server.wait()
            self.logger.info('Stopped server {}'.format(self.gate_props))
        self.server = None

    def __restart_server(self):
        self.logger.debug('Restarting server {}'.format(self.gate_props))
        self.__stop_server()
        self.__start_server()

    def parse(self, text, anas=False):
        """Parses a text with a running GATE server."""
        if not self.server:
            self.__start_server()
        with open('/dev/shm/text-{}'.format(os.getpid()), 'wt') as outf:
            print(text, file=outf)

        url = 'http://{}/process?{}'.format(
            self.gate_url, urlencode({'run': self.modules,
                                      'text': text.encode('utf-8')}))
        try:
            reply = self.__send_request(url)
            if reply:
                with open('/dev/shm/xml-{}'.format(os.getpid()), 'wt') as outf:
                    print(reply, file=outf)
                parsed = parse_gate_xml(reply, anas)
                if self.restart_every:
                    self.parsed += len(parsed)
                    if self.parsed >= self.restart_every:
                        self.__restart_server()
                return parsed
        except GateError as ge:
            self.__stop_server()
            raise
        except:
            self.__stop_server()
            raise

    def __send_request(self, url):
        for tries in range(3):
            try:
                r = requests.post(url, timeout=120)
                if r.status_code != 200:
                    self.logger.warning(
                        'Server {} returned an illegal status code {} (try {})'.format(
                            self.gate_props, r.status_code, tries + 1))
                    r = None
            except Exception as e:
                self.logger.warning(
                    u'Exception {} while trying to access server {} (try {})'.format(
                        e, self.gate_url, tries + 1))
                r = None
            if not r:
                self.__restart_server()
            if r:
                return r.content
        else:
            self.logger.error(
                u'Number of tries exceeded with server {}, url {}'.format(
                    self.gate_props, url))
            raise GateError(
                u'Number of tries exceeded with server {}'.format(self.gate_props))


def parse_gate_xml_file(xml_file, get_anas=False):
    """
    Parses a GATE response from a file. We use a SAX(-like?) parser, because
    only iterparse() provide the huge_tree argument, and it is needed sometimes
    if the analysis for a word is too long. Much uglier than the dom-based
    solution, but what can one do?
    """
    logger = logging.getLogger('emLam.GATE')
    text, sent = [], []
    token_feats = {'string': 0, 'lemma': 1, 'hfstana': 2}
    if get_anas:
        token_feats['anas'] = 3
    curr_token_feat = None
    tup = None
    annotation_id = None
    for event, node in etree.iterparse(xml_file, huge_tree=True, events=['start', 'end']):
        if event == 'start':
            if node.tag == 'Annotation':
                if node.get('Type') == 'Token':
                    tup = [None, None, None, None]
                    annotation_id = node.get('Id')
        else:  # end
            if node.tag == 'Annotation':
                if node.get('Type') == 'Token':
                    # The lemma might be None
                    if tup[LEMMA] is None:
                        tup[LEMMA] = tup[WORD]
                    if get_anas:
                        # If requested, find the analysis that matches lemma & POS
                        word, lemma, pos, anas = tup
                        if anas:
                            for ana in anas.split(';'):
                                try:
                                    a_ana, a_pos, a_lemma = _anas_p.match(ana).groups()
                                except:
                                    logger.exception(
                                        u'Strange ana {} / {} {} [{}]'.format(
                                            ana, lemma, pos, annotation_id))
                                    raise
                                if a_pos == pos and a_lemma == lemma:
                                    # This is the right analysis
                                    break
                            else:
                                a_ana = ''
                                # No matching analysis
                        else:
                            a_ana = ''
                            # Empty anas
                        tup[-1] = a_ana
                    sent.append(tup[:len(token_feats)])
                    tup = None
                elif node.get('Type') == 'Sentence':
                    text.append(sent)
                    sent = []
                annotation_id = None
            elif node.tag == 'Name' and tup and node.text in token_feats:
                curr_token_feat = node.text
            elif node.tag == 'Value' and curr_token_feat:
                tup[token_feats[curr_token_feat]] = node.text
                curr_token_feat = None
    return text


def parse_gate_xml_file_dom(xml_file, get_anas=False):
    """Parses a GATE response from a file."""
    logger = logging.getLogger('emLam.GATE')
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
                            logger.exception(u'Strange ana {} / {} {}'.format(
                                ana, lemma, pos))
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


class AnasParser(object):
    def __init__(self):
        def dummy(toks):
            toks.insert(0, '')

        ptag = pp.QuotedString('[', endQuoteChar=']', unquoteResults=False)
        ptext = pp.CharsNotIn('+[];')
        psurface = pp.Literal('=') + ptext
        psegment = pp.Or([
            ptext.setResultsName('segment') + ptag.setResultsName('tag') +
            pp.Suppress(psurface),
            pp.Empty().setParseAction(dummy).setResultsName('segment') +
            ptag.setResultsName('tag')
        ])
        pword = pp.delimitedList(pp.Group(psegment), '+')
        panas = pp.delimitedList(pp.Group(pword), ';')
        self.p = panas

    def parse(self, s):
        return self.p.parseString(s)
