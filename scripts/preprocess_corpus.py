#!/usr/bin/env python3
"""
Preprocesses the specified corpus. Converters should be put into the
emLam.corpus package. Currently, converters exist for the Szeged Treebank,
Webcorpus and MNSZ2.
"""

from __future__ import absolute_import, division, print_function
from builtins import range
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from functools import partial
import os
import os.path as op
from queue import Empty

from emLam.corpus import get_all_corpora, get_all_preprocessors
from emLam.corpus.corpus_base import GoldCorpus
from emLam.corpus.preprocessor_base import CopyPreprocessor
from emLam.corpus.gold_to_raw import GoldToRaw
from emLam.utils import run_queued, setup_queue_logger
from emLam.utils.config import cascade_section, handle_errors, load_config


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


def parse_arguments():
    corpora = get_all_corpora()
    preprocessors = get_all_preprocessors()

    parser = ArgumentParser(
        description='Preprocesses the specified corpus.',
        formatter_class=RawDescriptionHelpFormatter,
        epilog=usage_epilog(corpora, preprocessors))
    parser.add_argument('--source-dir', '-s', required=True,
                        help='the source directory.')
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

    args = parser.parse_args()
    if args.source_dir == args.target_dir:
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
    corpus_cls, preprocessor_cls, pid, config = components
    # First set up the logger used by the corpus and the preprocessor
    logger = setup_queue_logger(logging_level, logging_queue)

    # Then we can instantiate the objects that do the actual work
    corpus = corpus_cls.instantiate(
        pid, **cascade_section(config, corpus_cls.name))
    logger.debug('Configuration: {}'.format(cascade_section(config, corpus_cls.name)))
    preprocessor = preprocessor_cls.instantiate(
        pid, **cascade_section(config, preprocessor_cls.name))
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
    args, config = parse_arguments()
    os.nice(20)  # Play nice

    components = [(args.corpus, args.preprocessor, p + 1, config)
                  for p in range(args.processes)]
    source_target_files = source_target_file_list(args.source_dir, args.target_dir)

    run_queued(process_file, components,
               args.processes, source_target_files, args.log_level)


if __name__ == '__main__':
    main()
