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
import json
import os
import subprocess
import sys
from math import floor, log2

import numpy as np
import psutil


PATHS_FNAME = 'paths.json'
CONF_FNAME = 'config.json'


class Option:
    def __init__(self, argp):

        self.memory = None
        self.num_threads = None

        self.argp = argp

        self.script_path = os.path.dirname(os.path.realpath(__file__))
        self.paths = paths = self._load_paths()

        (self.data_path, self.model_path, self.bin_path, self.eval_path,
         self.embeds_basename, self.eval_fname
        ) = (None,) * 6

        for (key, value) in paths.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.config = config = self._load_config()

        (self.corpus_fname, self.cbow, self.voc_min_cnt, self.embeds_dim, self.iter,
         self.win_size, self.negative, self.hs, self.binary, self.sample
        ) = (None,) * 10

        for (key, value) in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if argp.corpus_fpath:
            self.corpus_fpath = argp.corpus_fpath
        else:
            self.corpus_fpath = os.path.join(paths['data_path'], self.corpus_fname)

        if self.binary:
            self.embeds_fpath = os.path.join(self.model_path, self.embeds_basename + '.bin')
        else:
            self.embeds_fpath = os.path.join(self.model_path, self.embeds_basename + '.txt')


        self.w2v = os.path.join(self.bin_path, 'word2vec')
        self.eval = os.path.join(self.bin_path, 'compute-accuracy')

    def _load_paths(self):
        with open(os.path.join(self.script_path, PATHS_FNAME)) as f:
            return json.load(f)

    def _load_config(self):
        if self.argp.cbow:
            conf_fpath = self._get_param_tag_fpath(self.script_path, CONF_FNAME, ['cbow'])
        else:
            conf_fpath = self._get_param_tag_fpath(self.script_path, CONF_FNAME, ['sg'])
        print("Config file used:", conf_fpath)
        with open(conf_fpath) as f:
            return json.load(f)

    def set_ressources(self, num_threads=None, memory=None, num_jobs=1):
        if memory is None:
            memory = psutil.virtual_memory().available / 1024**3
        if num_threads is None:
            num_threads = os.cpu_count()
        self.memory = 2**floor(log2(memory / num_jobs))
        self.num_threads = num_threads / num_jobs

    @staticmethod
    def _get_param_tag_fpath(path, fname, params):
        assert not isinstance(params, str)
        suffix = '_'.join([join_list(param_pair, sep='') for param_pair in params])
        (basename, ext) = os.path.splitext(fname)
        new_basename = '{}_{}{}'.format(basename, suffix, ext)
        return os.path.join(path, new_basename)


class Word2vec:
    def __init__(self, opts):
        self.opts = opts

    @staticmethod
    def _run_command(command, name=None, stdin=None, stdout=None):
        print(join_list(command))
        subprocess.run(command, stdin=stdin, stdout=stdout)
        if name is None:
            name = command[0]
        print("'{}' done.\n".format(name))

    def train(self):
        opts = self.opts
        command = [opts.w2v,
                   '-train', opts.corpus_fpath,
                   '-output', opts.embeds_fpath,
                   '-cbow', opts.cbow,
                   '-size', opts.embeds_dim,
                   '-window', opts.win_size,
                   '-negative', opts.negative,
                   '-hs', opts.hs,
                   '-sample', opts.sample,
                   '-threads', opts.num_threads,
                   '-binary', opts.binary,
                   '-iter', opts.iter,
                   '-min-count', opts.voc_min_cnt]
        self._run_command(lst2str_lst(command))


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
                    assert len(vec) == int(dim)
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


def lst2str_lst(lst):
    return [str(el) for el in lst]


def join_list(lst, sep=' '):
    return sep.join(lst2str_lst(lst))


def get_args(args=None):     # Add possibility to manually insert args at runtime (e.g. for ipynb)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train the models.')
    parser.add_argument('-c', '--cbow', action='store_true', default=False,
                        help='Use CBOW algorithm for training.')
    parser.add_argument('--corpus-fpath',
                        help='Training dataset filepath.')
    parser.add_argument('-x', '--export-embeds', action='store_true',
                        help='Export embeddings and vocabulary to file.')

    return parser.parse_args(args)


def main(argp):

    options = Option(argp)
    word2vec = Word2vec(options)
    export = Export(options)

    if argp.train:
        print("\n** TRAIN MODEL **\n")
        word2vec.train()

    if argp.export_embeds:
        export.import_embeds()
        export.export_embeds()


if __name__ == '__main__':
    try:
        main(get_args())
    except KeyboardInterrupt:
        sys.exit("\nProgram interrupted by user.\n")
