#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
Word2vec wrapper
================

Synopsis
--------
    examples:
    `````````
        ./scripts/run_w2v.py -x --export

Authors
-------
* Marc Evrard         (<marc.evrard@gmail.com>)
'''


import argparse
import sys

import numpy as np


class Option:
    def __init__(self, argp):
        self.embeds_fbasepath = argp.embeds_fbasepath


class Export:
    def __init__(self, opts):
        self.opts = opts

        self.id2word = []
        self.embeds = []

    def import_embeds(self):
        embeds, id2word = [], []
        with open(self.opts.embeds_fbasepath) as f:
            for idx, l in enumerate(f):
                if idx == 0:
                    n_words, dim = l.rstrip().split(' ')
                else:   # Skip header line
                    word, *vec = l.rstrip().split(' ')
                    id2word.append(word)
                    try:
                        embeds.append([float(el) for el in vec])
                    except ValueError:
                        print("**ERROR!**:", idx-1, word, len(vec), vec, sep='\n')
            # print(len(embeds), n_words)
            assert len(embeds) == int(n_words)

        self.id2word = id2word
        self.embeds = np.array(embeds, dtype=np.float32)    # pylint: disable=no-member

    def export_embeds(self):
        with open(self.opts.embeds_fbasepath[:-4] + '_voc.txt', 'w') as f:
            for word in self.id2word:
                f.write(word + '\n')
        np.save(self.opts.embeds_fbasepath[:-4], self.embeds)
        print("Mean | STD:", np.mean(self.embeds), '|', np.std(self.embeds))


def get_args(args=None):     # Add possibility to manually insert args at runtime (e.g. for ipynb)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-p', '--embeds-fbasepath',
                        help='Embedding name.')
    parser.add_argument('-x', '--export-embeds', action='store_true',
                        help='Export embeddings and vocabulary to file.')

    return parser.parse_args(args)


def main(argp):

    options = Option(argp)
    export = Export(options)

    if argp.export_embeds:
        export.import_embeds()
        export.export_embeds()


if __name__ == '__main__':
    try:
        main(get_args())
    except KeyboardInterrupt:
        sys.exit("\nProgram interrupted by user.\n")
