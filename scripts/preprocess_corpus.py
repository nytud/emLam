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
from emLam.logging import QueueHandler
from emLam.utils import run_queued


def parse_arguments():
    parser = ArgumentParser(
        description='Preprocesses the specified corpus.')
    parser.add_argument('--source-dir', '-s', required=True,
                        help='the source directory.')
    parser.add_argument('--target-dir', '-t', required=True,
                        help='the target directory.')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='the number of files to process parallelly.')
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    subparsers = parser.add_subparsers(
        title='Corpus selection',
        description='This section lists the corpora processor available for '
                    'selection. For help on a specific corpus, call the '
                    'script with the `<corp> -h` arguments.',
        dest='corpus', help='the corpora processors available.')
    corpora = get_all_corpora()
    for _, corpus_class in sorted(corpora.items()):
        corpus_class.parser(subparsers)

    args = parser.parse_args()
    if args.source_dir == args.target_dir:
        parser.error('Source and target directories must differ.')
    args.corpus = corpora[args.corpus]
    args.preprocessor = get_all_preprocessors()[args.preprocessor]
    if args.log_level:
        args.log_level = logging.getattr(args.log_level.upper())

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
    logger = logging.getLogger()
    if logging_level:
        logger.setLevel(logging_level)
        qh = QueueHandler(logging_queue)
        qh.setLevel(logging_level)
        logger.addHandler(qh)
    else:
        # Don't log anything
        logger.setLevel(logging.CRITICAL + 1)

    # Then we can instantiate the objects that do the actual work
    corpus = corpus_cls.instantiate(pid, **args)
    preprocessor = preprocessor_cls.instantiate(pid, **args)
    preprocessor.initialize()
    try:
        while True:
            try:
                infile, outfile = queue.get_nowait()
                logging.getLogger().info('Started processing {}'.format(infile))
                for ins, outs in corpus.files_to_streams(infile, outfile):
                    preprocessor.preprocess(ins, outs)
                logging.getLogger().info('Done processing {}'.format(infile))
            except Empty:
                break
            except:
                logging.getLogger().exception('Exception in file {}'.format(
                    infile))
                preprocessor.cleanup()
                preprocessor.initialize()
    except Exception as e:
        logging.getLogger().exception('Unexpected exception')
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
