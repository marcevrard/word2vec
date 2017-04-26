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

# TODO: Create parent classes for wvec and derive this module from them!

class Option:
    def __init__(self, argp, job_idx=None):

        self.embeds_fpath = ''
        self.result_fpath = ''
        self.memory = None
        self.num_threads = None

        self.argp = argp

        if argp.cbow:
            self.cbow = 1
            self.algo = 'cbow'
        else:
            self.cbow = 0
            self.algo = 'sg'

        self.script_path = os.path.dirname(os.path.realpath(__file__))
        self.paths = paths = self._load_paths()

        (self.corpus_fpath, self.model_path, self.bin_path, self.embeds_basename, self.eval_fname,
         self.quest_words_fpath, self.quest_phrases_fpath, self.result_path,
        ) = (None,) * 8

        for (key, value) in paths.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.config = config = self._load_config()

        (self.min_count, self.embeds_dim, self.iter, self.win_size, self.negative, self.hs,
         self.binary, self.sample,
        ) = (None,) * 8

        for (key, value) in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.emb_ext = '.bin' if self.binary else '.txt'

        self._build_param_tag_paths(job_idx)

        if argp.corpus_fpath:
            self.corpus_fpath = argp.corpus_fpath

        self.w2v = os.path.join(self.bin_path, 'word2vec')
        self.eval = os.path.join(self.bin_path, 'compute-accuracy')

        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

    def _load_paths(self):
        with open(os.path.join(self.script_path, PATHS_FNAME)) as f:
            return json.load(f)

    def _load_config(self):
        conf_fpath = self.get_param_tag_fpath(self.script_path, CONF_FNAME, [self.algo])
        print("Config file used:", conf_fpath)
        with open(conf_fpath) as f:
            config_dic = json.load(f)
        if self.argp.update_config:
            update_dic(dic=config_dic, update=self.argp.update_config)
            print("Config updated by:", self.argp.update_config)
        return config_dic

    def set_ressources(self, num_threads=None, memory=None, num_jobs=1):
        if memory is None:
            memory = psutil.virtual_memory().available / 1024**3
        if num_threads is None:
            num_threads = os.cpu_count()
        self.memory = 2**floor(log2(memory / num_jobs))
        self.num_threads = num_threads / num_jobs

    @staticmethod
    def get_param_tag_fpath(path, fname, params):
        assert not isinstance(params, str)
        suffix = '_'.join([join_list(param_pair, sep='') for param_pair in params])
        (basename, ext) = os.path.splitext(fname)
        new_basename = '{}_{}{}'.format(basename, suffix, ext)
        return os.path.join(path, new_basename)

    def _build_param_tag_paths(self, job_idx=None):

        idx_name = 'num'

        model_params = [(self.algo,),
                        ('cnt', self.min_count),
                        ('win', self.win_size),
                        ('neg', self.negative),
                        ('dim', self.embeds_dim),
                        ('itr', self.iter),
                        ('spl', self.sample)]

        if job_idx is not None:
            model_params += [(idx_name, job_idx)]

        self.embeds_fpath = self.get_param_tag_fpath(path=self.model_path,
                                                     fname=self.embeds_basename + self.emb_ext,
                                                     params=model_params)
        self.result_fpath = self.get_param_tag_fpath(path=self.result_path,
                                                     fname=self.eval_fname,
                                                     params=model_params)


class Word2vec:
    def __init__(self, opts):
        self.opts = opts

    @staticmethod
    def _run_command(command, name=None, stdin=None, stdout=None):
        print('Run:', join_list(command))
        subprocess.run(lst2str_lst(command), stdin=stdin, stdout=stdout)
        if name is None:
            name = command[0]
        print("'{}' done.\n".format(name))

    def train(self):
        opts = self.opts
        command = [opts.w2v,    # TODO: Convert to dic?
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
                   '-min-count', opts.min_count]
        self._run_command(command)

    def sim_eval(self):
        # TODO: apply same tests in GloVe and W2V
        opts = self.opts
        # Should get to almost 78% accuracy on 99.7% of questions
        command_wrd = [opts.eval,   # TODO: Convert to dic?
                       opts.embeds_fpath,
                       int(4e5)]  # Voc threshold: reduce model voc for fast approximate evaluation
        # about 78% accuracy with 77% coverage
        command_phr = [opts.eval,   # TODO: Convert to dic?
                       opts.embeds_fpath,
                       int(10e5)]  # Voc threshold

        with open(opts.result_fpath, 'w') as f_out:
            # Print parameters as header to evaluation output file
            f_out.write(str(opts.config) + '\n')
            f_out.write('====================\n\n')
            f_out.write("Start evaluation: questions-words\n\n")
        with open(opts.quest_words_fpath) as f_wrd_in, \
                open(opts.result_fpath, 'a') as f_out:
            self._run_command(command_wrd, stdin=f_wrd_in, stdout=f_out)
            f_out.write('====================\n\n')
            f_out.write("Start evaluation: questions-phrases\n\n")
        with open(opts.quest_phrases_fpath) as f_phr_in, \
                open(opts.result_fpath, 'a') as f_out:
            self._run_command(command_phr, stdin=f_phr_in, stdout=f_out)

        # with open(opts.eval_fpath, 'w') as f_out:
        #     f_out.write(params + '\n')
        #     f_out.write('==========\n')
        # with open(opts.eval_fpath, 'a') as f_out:
        #     self._run_command(lst2str_lst(command), name=opts.eval, stdout=f_out)


class Export:
    def __init__(self, opts):
        self.opts = opts

        self.id2word = []
        self.embeds = []

    def import_embeds(self):
        embeds, id2word = [], []
        with open(self.opts.embeds_fpath) as f:
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
        with open(self.opts.embeds_fpath[:-4] + '_voc.txt', 'w') as f:
            for word in self.id2word:
                f.write(word + '\n')
        np.save(self.opts.embeds_fpath[:-4], self.embeds)
        print("Mean | STD:", np.mean(self.embeds), '|', np.std(self.embeds))


def lst2str_lst(lst):
    return [str(el) for el in lst]


def join_list(lst, sep=' '):
    return sep.join(lst2str_lst(lst))


def update_dic(dic, update):
    for key in update:
        try:
            dic[key]
        except KeyError as err:
            print(err, "Update dict holds wrong keys!")
            raise
    return dic.update(update)


def get_args(args=None):     # Add possibility to manually insert args at runtime (e.g. for ipynb)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train the models.')
    parser.add_argument('-e', '--eval', action='store_true',
                        help='Eval the models.')
    parser.add_argument('-c', '--cbow', action='store_true', default=False,
                        help='Use CBOW algorithm for training.')
    parser.add_argument('--corpus-fpath',
                        help='Training dataset filepath.')
    parser.add_argument('-x', '--export-embeds', action='store_true',
                        help='Export embeddings and vocabulary to file.')
    parser.add_argument('-u', '--update-config', type=json.loads,
                        help="Add configuration setting(s) to local json config file.")

    return parser.parse_args(args)


def main(argp):

    options = Option(argp)
    word2vec = Word2vec(options)
    export = Export(options)

    if argp.train:
        print("\n** TRAIN MODEL **\n")
        word2vec.train()

    if argp.eval:
        print("\n** EVALUATE MODEL **\n")
        word2vec.sim_eval()

    if argp.export_embeds:
        export.import_embeds()
        export.export_embeds()


if __name__ == '__main__':
    try:
        main(get_args())
    except KeyboardInterrupt:
        sys.exit("\nProgram interrupted by user.\n")
