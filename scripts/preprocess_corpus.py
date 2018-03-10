#!/usr/bin/env python3

"""
Preprocesses the specified corpus. Converters should be put into the
emLam.corpus package. Currently, converters exist for the Szeged Treebank,
Webcorpus and MNSZ2.
"""

from __future__ import absolute_import, division, print_function
from builtins import range
import os
from queue import Empty

from emLam.utils import run_queued, setup_queue_logger, source_target_file_list
from emLam.utils.config import cascade_section
from emLam.utils.preprocessing import parse_arguments


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
    source_target_files = source_target_file_list(args.source, args.target_dir)

    run_queued(process_file, components,
               args.processes, source_target_files, args.log_level)


if __name__ == '__main__':
    main()
