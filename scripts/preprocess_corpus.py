#!/usr/bin/env python3
"""
Preprocesses the specified corpus. Converters should be put into the
emLam.corpus package. Currently, converters exist for the Szeged Treebank,
Webcorpus and MNSZ2.
"""

from __future__ import absolute_import, division, print_function
from builtins import range
from argparse import ArgumentParser
import logging
import os
import os.path as op
from queue import Empty

from emLam.corpus import get_all_corpora, get_all_preprocessors
from emLam.corpus.corpus_base import GoldCorpus
from emLam.corpus.preprocessor_base import CopyPreprocessor
from emLam.utils import run_queued, setup_queue_logger


def usage_epilog(corpora, preprocessors):
    """Describes the various Corpus and Preprocessor classes available."""
    cformat = '{{:<{}}} - {{}}'.format(max(len(name) for name, _ in corpora))
    pformat = '{{:<{}}} - {{}}'.format(max(len(name) for name, _ in preprocessors))
    c = '\nThe following corpora are available:\n' + '\n'.join(
        cformat.format(name, cls.description) for name, cls in corpora)
    p = '\nThe following corpora are available:\n' + '\n'.join(
        pformat.format(name, cls.description) for name, cls in corpora)
    return c + '\n' + p


def parse_arguments():
    corpora = get_all_corpora()
    preprocessors = get_all_preprocessors()

    parser = ArgumentParser(
        description='Preprocesses the specified corpus.',
        epilog=usage_epilog(corpora, preprocessors))
    parser.add_argument('--source-dir', '-s', required=True,
                        help='the source directory.')
    parser.add_argument('--target-dir', '-t', required=True,
                        help='the target directory.')
    parser.add_argument('--corpus', '-p', required=True,
                        choices=[c for c, _ in corpora],
                        help='the corpus to preprocess. See below for a '
                             'description of the available corpora.')
    parser.add_argument('--preprocessor', '-p', required=True,
                        choices=[p for p, _ in preprocessors],
                        help='the preprocessor to use. See below for a '
                             'description of the available options.')
    parser.add_argument('--configuration', '-C', required=True,
                        help='the configuration file.')
    parser.add_argument('--processes', '-P', type=int, default=1,
                        help='the number of files to process parallelly.')
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')

    args = parser.parse_args()
    if args.source_dir == args.target_dir:
        parser.error('Source and target directories must differ.')

    args.corpus = corpora[args.corpus]
    args.preprocessor = preprocessors[args.preprocessor]
    if (
        issubclass(args.corpus, GoldCorpus) and
        args.preprocessor != CopyPreprocessor
    ):
        parser.error("Gold standard corpora can only be used with the ``copy'' "
                     "preprocessor.")

    return args


def walk_non_hidden(directory):
    """Walks directory as os.walk, skipping hidden files and directories."""
    def delete_hidden(lst):
        for i in range(len(lst) - 1, -1, -1):
            if lst[i][0] == '.':
                del lst[i]

    for tup in os.walk(directory):
        dirpath, dirnames, filenames = tup
        delete_hidden(dirnames)
        delete_hidden(filenames)
        yield tup


def source_target_file_list(source_dir, target_dir):
    source_dir = op.abspath(source_dir)
    target_dir = op.abspath(target_dir)
    source_files = [op.abspath(op.join(d, f))
                    for d, _, fs in walk_non_hidden(source_dir) for f in fs]
    target_files = []
    for sf in source_files:
        sf_rel = sf[len(source_dir):].lstrip(os.sep)
        tf = op.join(target_dir, sf_rel)
        td = op.dirname(tf)
        if not op.isdir(td):
            os.makedirs(td)
        target_files.append(tf)
    return zip(source_files, target_files)


def process_file(components, queue, logging_level=None, logging_queue=None):
    corpus_cls, preprocessor_cls, pid, args = components
    # First set up the logger used by the corpus and the preprocessor
    logger = setup_queue_logger(logging_level, logging_queue)

    # Then we can instantiate the objects that do the actual work
    corpus = corpus_cls.instantiate(pid, **args)
    preprocessor = preprocessor_cls.instantiate(pid, **args)
    preprocessor.initialize()
    try:
        while True:
            try:
                infile, outfile = queue.get_nowait()
                logger.info('Started processing {}'.format(infile))
                for ins, outs in corpus.files_to_streams(infile, outfile):
                    preprocessor.preprocess(ins, outs)
                logger.info('Done processing {}'.format(infile))
            except Empty:
                logger.debug('Queue depleted.')
                break
            except:
                logger.exception('Exception in file {}'.format(
                    infile))
                preprocessor.cleanup()
                preprocessor.initialize()
    except Exception as e:
        logger.exception('Unexpected exception')
        raise
    finally:
        preprocessor.cleanup()


def main():
    args = parse_arguments()
    os.nice(20)  # Play nice

    components = [(args.corpus, args.preprocessor, p + 1, dict(args.__dict__))
                  for p in range(args.processes)]
    source_target_files = source_target_file_list(args.source_dir, args.target_dir)

    run_queued(process_file, components,
               args.processes, source_target_files, args.log_level)


if __name__ == '__main__':
    main()
