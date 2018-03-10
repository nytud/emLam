#!/usr/bin/env python3

"""
Parts of the preprocessing script that are useful outside of it (e.g. for
fabric), too.
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from functools import partial
import os.path as op
import sys

from emLam.corpus import get_all_corpora, get_all_preprocessors
from emLam.corpus.corpus_base import GoldCorpus
from emLam.corpus.preprocessor_base import CopyPreprocessor
from emLam.corpus.gold_to_raw import GoldToRaw
from emLam.utils.config import handle_errors, load_config


def usage_epilog(corpora, preprocessors):
    """Describes the various Corpus and Preprocessor classes available."""
    cformat = '{{:<{}}} - {{}}'.format(max(len(name) for name in corpora.keys()))
    pformat = '{{:<{}}} - {{}}'.format(max(len(name) for name in preprocessors.keys()))
    c = '\nThe following corpora are available:\n' + '\n'.join(
        cformat.format(name, cls_path[0].description) for name, cls_path
        in corpora.items())
    p = '\nThe following preprocessors are available:\n' + '\n'.join(
        pformat.format(name, cls_path[0].description) for name, cls_path
        in preprocessors.items())
    return c + '\n' + p


def config_pp(config, warnings, errors, class_paths):
    """
    Postprocessing function for the configuration: makes sure that it has
    sections for the selected corpus and preprocessor. Not all components have
    sections in the configuration; examples include the text corpus and the copy
    preprocessor. However, since properties are inherited from the ancestor
    classes, we need to know their full path later.
    """
    for path in class_paths:
        cfg = config
        for section in path:
            cfg = cfg.setdefault(section, {})


def parse_arguments(args=None):
    if not args:
        args = sys.argv[1:]
    corpora = get_all_corpora()
    preprocessors = get_all_preprocessors()

    parser = ArgumentParser(
        description='Preprocesses the specified corpus.',
        formatter_class=RawDescriptionHelpFormatter,
        epilog=usage_epilog(corpora, preprocessors))
    parser.add_argument('--source', '-s', required=True,
                        help='the data source. Either a directory, in which '
                             'all files added to the input recursively, or a '
                             'file that lists all input files.')
    parser.add_argument('--target-dir', '-t', required=True,
                        help='the target directory.')
    parser.add_argument('--corpus', '-c', required=True,
                        choices=[c for c in corpora.keys()],
                        help='the corpus to preprocess. See below for a '
                             'description of the available corpora.')
    parser.add_argument('--preprocessor', '-p', required=True,
                        choices=[p for p in preprocessors.keys()],
                        help='the preprocessor to use. See below for a '
                             'description of the available options.')
    parser.add_argument('--configuration', '-C', required=True,
                        help='the configuration file.')
    parser.add_argument('--processes', '-P', type=int, default=1,
                        help='the number of files to process parallelly.')
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')

    args = parser.parse_args(args)
    if not op.exists(args.source):
        parser.error('Source {} does not exist.'.format(args.source))
    if args.source == args.target_dir:
        parser.error('Source and target directories must differ.')

    args.corpus, corpus_path = corpora[args.corpus]
    args.preprocessor, preprocessor_path = preprocessors[args.preprocessor]
    if (
        issubclass(args.corpus, GoldCorpus) and
        args.preprocessor not in [CopyPreprocessor, GoldToRaw]
    ):
        parser.error("Gold standard corpora can only be used with the ``copy'' "
                     "preprocessor.")

    # Config file
    config, warnings, errors = load_config(
        args.configuration, 'preprocess_corpus.schema',
        retain=[args.corpus.name, args.preprocessor.name],
        postprocessing=partial(config_pp,
                               class_paths=[corpus_path, preprocessor_path]))
    handle_errors(warnings, errors)

    return args, config
