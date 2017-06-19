#!/usr/bin/env python3
"""Manages a GATE server."""

from __future__ import absolute_import, division, print_function
from builtins import range
try:
    from configparser import RawConfigParser
except:
    from ConfigParser import RawConfigParser
from future.moves.urllib.parse import urlencode
from io import open, BytesIO, StringIO
import json
import logging
import os
import re
from subprocess import Popen
import time
import warnings

from lxml import etree
import requests

from emLam import WORD, LEMMA


_anas_p = re.compile(r'ana=([^}]+), feats=([^}]+])(?:, incorrect=([^,}]+))?, '
                     r'lemma=([^}]+)?')


class GateError(Exception):
    pass


class Gate(object):
    """hunlp-GATE interface object."""
    def __init__(self, gate_props, modules, restart_every=0, gate_version=8.4):
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
        self.gate_url = self.__gate_url()
        self.restart_every = restart_every
        self.modules = modules
        self.server = None
        self.parsed = 0
        self.parser = GateOutputParser.get_parser(gate_version)
        self.__start_server()

    def __del__(self):
        # See also http://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
        # for ideas on how to replace __del__ with something better (?)
        self.__stop_server()

    def __gate_url(self):
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

    def restart_server(self):
        self.logger.debug('Restarting server {}'.format(self.gate_props))
        self.__stop_server()
        self.__start_server()

    def parse(self, text, anas='no'):
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
                parsed = self.parser.parse_gate_xml(reply, anas)
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


class GateOutputParser(object):
    """Class for parsing the GATE output XML."""
    def __init__(self):
        self.logger = logging.getLogger('emLam.GATE')
        self.logger.debug('GATE parser class: {}'.format(self.__class__.__name__))

    @staticmethod
    def get_parser(gate_version=8.4):
        if gate_version <= 8.2:
            return GateOutputParser82()
        else:
            return GateOutputParser84()

    def parse_gate_xml_file(self, xml_file, get_anas='no'):
        """
        Parses a GATE response from a file. We use a SAX(-like?) parser, because
        only iterparse() provide the huge_tree argument, and it is needed sometimes
        if the analysis for a word is too long. Much uglier than the dom-based
        solution, but what can one do?
        """
        text, sent = [], []
        token_feats = {'string': 0, 'lemma': 1, 'hfstana': 2, 'anas': 3}
        curr_token_feat = None
        tup = None
        for event, node in etree.iterparse(xml_file, huge_tree=True, encoding='utf-8', events=['start', 'end']):
            if event == 'start':
                if node.tag == 'Annotation':
                    if node.get('Type') == 'Token':
                        tup = [None, None, None, None]
            else:  # end
                if node.tag == 'Annotation':
                    if node.get('Type') == 'Token':
                        # The lemma might be None
                        if tup[LEMMA] is None or '<incorrect_word>' in tup[LEMMA]:
                            tup[LEMMA] = tup[WORD]
                        sent.append(self.extract_anas(tup, get_anas))
                        tup = None
                    elif node.get('Type') == 'Sentence':
                        text.append(sent)
                        sent = []
                elif node.tag == 'Name' and tup and node.text in token_feats:
                    curr_token_feat = node.text
                elif node.tag == 'Value' and curr_token_feat:
                    tup[token_feats[curr_token_feat]] = node.text
                    curr_token_feat = None
        return text

    def parse_gate_xml_file_dom(self, xml_file, get_anas='no'):
        """
        Parses a GATE response from a file. Deprecated in favor of
        parse_gate_xml_file().
        """
        warnings.warn('parse_gate_xml_file_dom() is deprecated, as it '
                      'cannot handle arbirarily large large inputs. Use '
                      'parse_gate_xml_file() instead.', DeprecationWarning)
        dom = etree.parse(xml_file)
        root = dom.getroot()
        text, sent = [], []
        for a in root.xpath('./AnnotationSet/Annotation[@Type!="SpaceToken"]'):
            if a.attrib['Type'] == 'Token':
                word = a.find('Feature[Name="string"]').find('Value').text
                lemma = a.find('Feature[Name="lemma"]').find('Value').text
                pos = a.find('Feature[Name="hfstana"]').find('Value').text
                anas = a.find('Feature[Name="anas"]').find('Value').text or ''
                tup = [word, lemma if lemma else word, pos, anas]
                sent.append(self.extract_anas(tup, get_anas))
            else:
                text.append(sent)
                sent = []
        return text

    def parse_gate_xml(self, xml, anas='no'):
        """Parses a GATE response from memory."""
        return self.parse_gate_xml_file(BytesIO(xml), anas)

    def extract_anas(self, tup, get_anas):
        """Extracts the anas and writes them back to tup as a json list."""
        if get_anas != 'no':
            word, lemma, pos, anas = tup
            if anas:
                all_anas = self.parse_anas(anas, tup)
            else:
                all_anas = []
            if get_anas == 'matching':
                all_anas = [a for a in all_anas if
                            a['lemma'] == lemma and a['feats'] == pos]
            tup[-1] = json.dumps(all_anas)
        else:
            tup = tup[:-1]
        return tup

    def parse_anas(self, anas, tup):
        """
        Extracts the anas. tup is only passed to the function for better error
        reporting.
        """
        raise NotImplementedError('parse_anas() must be implemented.')


class GateOutputParser82(GateOutputParser):
    """Gate output parser for GATE version 8.2 (and below?)."""
    def parse_anas(self, anas, tup):
        """Extracts the anas from the output of GATE 8.2 or below."""
        ret = []
        if anas:
            for ana in anas[1:-1].split('};{'):
                try:
                    a_ana, a_pos, a_incorrect, a_lemma = _anas_p.match(ana).groups()
                    feats = {'ana': a_ana, 'feats': a_pos, 'lemma': a_lemma}
                    if a_incorrect:
                        feats['incorrect'] = True
                        feats['lemma'] = tup[WORD]
                    ret.append(feats)
                except:
                    self.logger.exception(
                        u'Strange ana "{}" in "{}"; {}'.format(ana, anas, tup))
                    raise
        return ret


class GateOutputParser84(GateOutputParser):
    """Gate output parser for GATE version 8.4 (and above?)."""
    def parse_anas(self, anas, tup):
        """Extracts the anas from the output of GATE 8.4 or above."""
        ret = []
        for all_ana in etree.fromstring(anas):
            for ana in all_ana:
                feats = {}
                for entry in ana:
                    feat = entry.xpath('./string')[0].text
                    value = entry.xpath('./string')[1].text
                    feats[feat] = value
                if feats.get('incorrect'):
                    feats['lemma'] = tup[WORD]
                ret.append(feats)
        return ret
