#!/usr/bin/env python3

"""
Preprocesses the specified corpus. Converters should be put into the
emLam.corpus package. Currently, converters exist for the Szeged Treebank,
Webcorpus and MNSZ2.
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from builtins import range
import copy
from functools import partial
import os
import os.path as op
from queue import Empty
import sys

from emLam.corpus import get_all_corpora, get_all_preprocessors
from emLam.corpus.corpus_base import GoldCorpus
from emLam.corpus.preprocessor_base import CopyPreprocessor
from emLam.corpus.gold_to_raw import GoldToRaw
from emLam.utils import run_queued, setup_queue_logger, source_target_file_list
from emLam.utils.config import cascade_section, handle_errors, load_config
import emLam.utils.remote as remote


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
    parser.add_argument('--remote', '-R',
                        help='Runs the script remotely. The value is '
                             'in the format <cmd>:<cf>, where <cf> is '
                             'a configuration file following '
                             'remote_preprocessing.schema, and <cmd> is '
                             'the remote command: one of "setup", "run" and '
                             '"cleanup" (with the usual semantics). All other '
                             'parameters are taken from this command line. '
                             'Note that this implies that the source and '
                             'target directories must be reachable from all '
                             'remote hosts.')
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


def run_locally(args, config):
    """Runs the preprocessing locally."""
    os.nice(20)  # Play nice

    components = [(args.corpus, args.preprocessor, p + 1, config)
                  for p in range(args.processes)]
    source_target_files = source_target_file_list(args.source, args.target_dir)

    run_queued(process_file, components,
               args.processes, source_target_files, args.log_level)


def get_any_index(lst, *values):
    """Returns the index of (the first of) a set of values in lst."""
    for value in values:
        try:
            return lst.index(value)
        except ValueError:
            pass


def get_remote_params(args):
    """
    Returns the remote command and configuration. Also amends the command line:
    - removes the -R switch and its value
    - changes the configuration file to remote_dir/rconfig.conf.
    """
    cmd, cf = args.remote.rsplit(':', 1)
    remote_config = load_config(cf, 'remote_preprocessing.schema')[0]
    cmd_fields = copy.copy(sys.argv[1:])
    remote_idx = get_any_index(cmd_fields, '-R', '--remote')
    del cmd_fields[remote_idx:remote_idx + 2]
    config_idx = get_any_index(cmd_fields, '-C', '--configuration')
    cmd_fields[config_idx] = os.path.join(
        remote_config['Infrastructure']['remote_dir'], 'rconfig.conf')
    input_idx = get_any_index(cmd_fields, '-s', '--source')
    cmd_fields[input_idx] = os.path.join(
        remote_config['Infrastructure']['remote_dir'], 'input.lst')
    remote_config['cmd_line'] = ' '.join(cmd_fields)
    print(remote_config['cmd_line'])
    return cmd, remote_config


def run_remotely(args, config):
    """Runs the preprocessing remotely."""
    cmd, remote_config = get_remote_params(args)
    if cmd not in remote.commands:
        raise ValueError('Remote commands must be one of {}'.format(
            ', '.join(remote.commands)))
    getattr(remote, cmd)(remote_config, args)


def main():
    args, config = parse_arguments()
    if args.remote:
        run_remotely(args, config)
    else:
        run_locally(args, config)


if __name__ == '__main__':
    main()
