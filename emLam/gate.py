#!/usr/bin/env python3
"""Manages a GATE server."""

from __future__ import absolute_import, division, print_function
from builtins import range
try:
    from configparser import RawConfigParser
except:
    from ConfigParser import RawConfigParser
from collections import defaultdict
from future.moves.urllib.parse import urlencode
from io import open, BytesIO, StringIO
import json
import logging
import os
import re
from subprocess import Popen
import time

from lxml import etree
import requests


_anas_p = re.compile(r'ana=([^}]+), feats=([^}]+])(?:, incorrect=([^,}]+))?, '
                     r'lemma=([^}]+)?')


class GateError(Exception):
    pass


class Gate(object):
    """hunlp-GATE interface object."""

    def __init__(self, gate_props, modules, token_feats, get_anas='no',
                 restart_every=0, gate_version=8.4):
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
        self.modules = modules
        self.get_anas = get_anas
        self.restart_every = restart_every
        self.server = None
        self.parsed = 0
        if (self.get_anas == 'no') == ('anas' in token_feats):
            raise ValueError(
                'Invalid setup: anas should be "no" if and only if it is not '
                'in token_feats')
        self.parser = GateOutputParser.get_parser(token_feats, gate_version)
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

    def parse(self, text):
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
                with open('/dev/shm/xml-{}'.format(os.getpid()), 'wb') as outf:
                    outf.write(reply)
                parsed = self.parser.parse_gate_xml(reply, self.get_anas)
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


class Feature(object):
    def __init__(self, default=None):
        self.default = default
        self.value = None

    def set(self, orig_value):
        """Might modify the original value."""
        self.value = orig_value

    def get(self):
        return self.value if self.value is not None else self.default

    def __repr__(self):
        return '{}({} / {})'.format(
            self.__class__.__name__, self.value, self.default)


class DepTargetFeature(Feature):
    """Adds 1 to the depTarget, because the ML in GATE starts at 0..."""
    def set(self, orig_value):
        self.value = str(int(orig_value) + 1)


class DefaultFeature(Feature):
    """Always returns the default."""
    def __init__(self, default=None):
        super(DefaultFeature, self).__init__(default)
        self.value = self.default


class GateOutputParser(object):
    """Class for parsing the GATE output XML."""

    # Defaults for the token features (needed because stupid GATE does not
    # add the ROOT/0 dependency to the verb)
    DEFAULTS = {'depTarget': lambda: Feature('-1'),
                'depType': lambda: Feature('ROOT'),
                '_': lambda: DefaultFeature('_')}

    def __init__(self, token_feats):
        self.token_feats_list = token_feats.split(',')
        self.token_feats = {f: i for i, f in enumerate(self.token_feats_list)}
        self.logger = logging.getLogger('emLam.GATE')
        self.logger.debug('GATE parser class: {}'.format(self.__class__.__name__))
        # Whether we need to fix the ids (GATE global ids to per sentence ids)
        self.fix_ids = set(['id', 'depTarget']) & set(self.token_feats_list)

    @staticmethod
    def get_parser(token_feats, gate_version=8.4):
        if gate_version <= 8.2:
            return GateOutputParser82(token_feats)
        else:
            return GateOutputParser84(token_feats)

    def parse_gate_xml_file(self, xml_file, get_anas):
        """
        Parses a GATE response from a file. We use a SAX(-like?) parser, because
        only iterparse() provide the huge_tree argument, and it is needed sometimes
        if the analysis for a word is too long. Much uglier than the dom-based
        solution, but what can one do?
        """
        text, sent = [], []
        curr_token_feat = None
        data = None
        for event, node in etree.iterparse(
            xml_file, huge_tree=True, encoding='utf-8', events=['start', 'end']
        ):
            if event == 'start':
                if node.tag == 'Annotation':
                    if node.get('Type') == 'Token':
                        data = defaultdict(lambda: Feature())
                        data.update({tf: self.DEFAULTS[tf]()
                                     for tf in self.token_feats_list
                                     if tf in self.DEFAULTS})
            else:  # end
                if node.tag == 'Annotation':
                    if node.get('Type') == 'Token':
                        lemma = data['lemma'].value
                        # The lemma might be None
                        if lemma is None or '<incorrect_word>' in lemma:
                            data['lemma'] = data.get('string', lemma)
                        data['id'].set(node.get('Id'))
                        sent.append(data)
                        data = None
                    elif node.get('Type') == 'Sentence':
                        self.fix_id_and_dep(sent)
                        sent = [[token[tf].get() for tf in self.token_feats_list]
                                for token in sent]
                        text.append(sent)
                        sent = []
                elif node.tag == 'Name' and data is not None:
                    curr_token_feat = node.text
                elif node.tag == 'Value' and curr_token_feat:
                    data[curr_token_feat].set(node.text)
                    curr_token_feat = None
        return text

    def fix_id_and_dep(self, sentence):
        """
        Fixes the IDs of the words in the sentence so that it starts from 1 and
        the ROOT is 0. Only run if either id or dep* is a requested feature.
        """
        if self.fix_ids:
            mapping = {'-1': '0'}
            for i, data in enumerate(sentence, 1):
                mapping[data['id'].get()] = str(i)
            for data in sentence:
                data['id'].set(mapping[data['id'].get()])
                data['depTarget'].set(mapping[data['depTarget'].get()])

    def parse_gate_xml(self, xml, anas='no'):
        """Parses a GATE response from memory."""
        return self.parse_gate_xml_file(BytesIO(xml), anas)

    def extract_anas(self, data, get_anas):
        """Extracts the anas and writes them back to data as a json list."""
        if get_anas != 'no':
            anas = data['anas'].value
            if anas:
                word = data['string']
                try:
                    all_anas = self.parse_anas(anas, word)
                except:
                    self.logger.exception(
                        u'Could not parse anas "{}"; {}'.format(anas, data))
                    raise
            else:
                all_anas = []
            if get_anas == 'matching':
                lemma = data['lemma'].value
                pos = data['hfstana'].value
                all_anas = [a for a in all_anas if
                            a['lemma'] == lemma and a['feats'] == pos]
            data['anas'] = json.dumps(all_anas)
        return data

    def parse_anas(self, anas, word):
        """
        Extracts the anas. word is passed to the function to handle "incorrect"
        parses.
        """
        raise NotImplementedError('parse_anas() must be implemented.')


class GateOutputParser82(GateOutputParser):
    """Gate output parser for GATE version 8.2 (and below?)."""
    def parse_anas(self, anas, word):
        """Extracts the anas from the output of GATE 8.2 or below."""
        ret = []
        if anas:
            for ana in anas[1:-1].split('};{'):
                try:
                    a_ana, a_pos, a_incorrect, a_lemma = _anas_p.match(ana).groups()
                    feats = {'ana': a_ana, 'feats': a_pos, 'lemma': a_lemma}
                    if a_incorrect:
                        feats['incorrect'] = True
                        feats['lemma'] = word
                    ret.append(feats)
                except:
                    self.logger.exception(u'Strange ana "{}"'.format(ana))
                    raise
        return ret


class GateOutputParser84(GateOutputParser):
    """Gate output parser for GATE version 8.4 (and above?)."""
    def parse_anas(self, anas, word):
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
                    feats['lemma'] = word
                ret.append(feats)
        return ret
