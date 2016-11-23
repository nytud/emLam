"""Creates a corpus with UNK values from a SRILM input corpus."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from collections import Counter
from functools import partial, reduce
from multiprocessing import Pool
from operator import iadd
import os
import re
import sys

from emLam.utils import openall


def parse_arguments():
    parser = ArgumentParser(
        description='Creates a corpus with UNK values from a SRILM input corpus.')
    parser.add_argument('--source-dir', '-s', default='.',
                        help='The source directory. The default is \'.\'.')
    parser.add_argument('--target-dir', '-t', default='.',
                        help='The target directory. The default is \'.\'.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--n', '-n', type=int,
                       help='How many first occurrences to replace with '
                            '"<unk>". The default is 1.')
    group.add_argument('--min', '-m', type=int,
                       help='What is the minimum number of occurrences above '
                            'which words are kept. The default is 1.')
    parser.add_argument('--vocab-file', '-v',
                        help='A vocabulary file. If used together with --min, '
                             'the counts will be read from here, and not '
                             'computed from the corpus.')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='How many processes to use vocabulary counting or '
                             '--min.')
    args = parser.parse_args()

    if args.source_dir == args.target_dir:
        parser.error('Source and target directories must differ.\n')
    elif args.source_dir == args.pos_dir:
        parser.error('Source and POs directories must differ.\n')
    elif args.target_dir == args.pos_dir:
        parser.error('Target and POs directories must differ.\n')
    try:
        source_regex = re.compile(args.source_regex if args.source_regex else '.*')
    except:
        parser.exit(1, 'Invalid source regex.\n')
    if args.vocab_file is not None and args.min is None:
        parser.error('--vocab-file has no effect if --min is not used.')

    return (args.source_dir, args.target_dir, args.pos_dir, source_regex,
            args.n, args.min, args.vocab_file, args.processes)


def unk_first(source_files, pos_files, target_files, n):
    """
    Replaces the first n occurrences of each word. Reads the source files in
    a round-robin fashion, so that the distribution is balanced.
    """
    infs = [openall(sf) for sf in source_files]
    outfs = [openall(tf, 'wt') for tf in target_files]
    posfs = [openall(pf) for pf in pos_files]
    seen_it = Counter()
    while len(infs):
        for f in range(len(infs) - 1, -1, -1):
            il = infs[f].readline()
            pl = posfs[f].readline() if posfs[f] else None
            if not il:
                infs[f].close()
                if posfs[f]:
                    posfs[f].close()
                outfs[f].close()
                del infs[f]
                del posfs[f]
                del outfs[f]
            else:
                tokens = il.strip().split()
                ptokens = pl.strip().split() if pl else None
                for i, token in enumerate(tokens):
                    seen_it[token] += 1
                    if seen_it[token] <= n:
                        if ptokens:
                            tokens[i] = '<unk-{}>'.format(ptokens[i].lower())
                        else:
                            tokens[i] = '<unk>'
                print >>outfs[f], ' '.join(tokens)


def _replace_file(in_out_files):
    in_file, pos_file, out_file = in_out_files
    if pos_file:
        with openall(in_file) as inf, openall(pos_file) as pinf, \
             openall(out_file, 'wt') as outf:  # noqa
            for line_no, line in enumerate(inf):
                lemmas = line.strip().split()
                poss = pinf.readline().strip().lower().split()
                try:
                    print(' '.join(w if w in keep_words else '<unk-{}>'.format(poss[i])
                                   for i, w in enumerate(lemmas)),
                          file=outf)
                except:
                    raise ValueError('Error in {}/{}:{} --- {}:{} vs {}:{}'.format(in_file, pos_file, line_no, len(lemmas), lemmas, len(poss), poss))
    else:
        with openall(in_file) as inf, openall(out_file, 'wt') as outf:
            for line in inf:
                print(' '.join(w if w in keep_words else '<unk>'
                               for w in line.strip().split()), file=outf)


def unk_keep_list(source_files, pos_files, target_files, keep_words, processes):
    """Replaces all words not in keep_words with <unk>."""
    Pool(processes).map(_replace_file,
                        zip(source_files, pos_files, target_files))


def _count_file(file_name):
    c = Counter()
    with openall(file_name) as inf:
        for line in inf:
            c.update(line.strip().split())
    return c


def count_occurrences(source_files, processes):
    """Counts the occurrences of all words in the source files."""
    p = Pool(processes)
    return reduce(iadd, p.map(_count_file, source_files))


def freqs_file_to_dict(file_name):
    with openall(file_name) as inf:
        return {l[0]: int(l[1]) for l in
                map(lambda l: l.split(), map(str.strip, inf))}


def freqs_dict_to_file(file_name, freqs):
    with openall(file_name, 'wt') as outf:
        outf.write("\n".join("{}\t{}".format(word, freq) for word, freq in
                             sorted(freqs.items())))


if __name__ == '__main__':
    (source_dir, target_dir, pos_dir, source_regex,
     n, min_occ, vocab_file, processes) = parse_arguments()

    os.nice(20)  # Play nice

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    files = source_target_file_list(args.source_dir, args.target_dir):
    if n:
        unk_first(source_files, pos_files, target_files, n)
    else:
        if vocab_file is None or not os.path.isfile(vocab_file):
            freqs = count_occurrences(source_files, processes)
            print('Counted {} words.'.format(len(freqs)), file=sys.stderr)
            if vocab_file is not None:
                freqs_dict_to_file(vocab_file, freqs)
        else:
            freqs = freqs_file_to_dict(vocab_file)
            print('Read counts for {} words'.format(len(freqs)),
                  file=sys.stderr)
        keep_words = set(w for w, f in freqs.items() if f >= min_occ)
        print('Kept {} words.'.format(len(keep_words)), file=sys.stderr)
        unk_keep_list(source_files, pos_files, target_files, keep_words, processes)
