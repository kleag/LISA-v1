#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Timothy Dozat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.





import argparse
import numpy as np
import os
import sys
import tensorflow as tf

from configparser import ConfigParser

#***************************************************************
class Configurable(object):
  """"""

  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""

    self._name = kwargs.pop('name', type(self).__name__)
    if args and kwargs:
      raise TypeError('Configurables must take either a config parser or keyword args')
    if args:
      if len(args) > 1:
        raise TypeError('Configurables take at most one argument')

    if args:
      self._config = args[0]
    else:
      self._config = self._configure(**kwargs)
    return

  #=============================================================
  def _configure(self, **kwargs):
    """"""

    config = ConfigParser()
    config_files = [os.path.join('config', 'defaults.cfg'),
                    os.path.join('config', self.name.lower() + '.cfg')]
    if 'config_file' in kwargs:
      config_files.append(kwargs.pop('config_file'))
    elif 'save_dir' in kwargs:
      config_files.append(f"{kwargs['save_dir']}/config.cfg")
    print(f"Configurable.configure {config_files}", file=sys.stderr)
    files_read = config.read(config_files)
    #print(f"Configurable.configure loaded: {files_read}")
    for option, value in list(kwargs.items()):
      if option == "files":
        continue
      assigned = False
      for section in config.sections():
        if option in config.options(section):
          config.set(section, option, str(value))
          assigned = True
          break
      if not assigned:
        raise ValueError('%s is not a valid option.' % option)

    return config

  #=============================================================
  argparser = argparse.ArgumentParser()
  argparser.add_argument('files', nargs='*')
  argparser.add_argument('--config_file')
  argparser.add_argument('--data_dir')
  argparser.add_argument('--embed_dir')

  @property
  def name(self):
    return self._name
  argparser.add_argument('--name')

  #=============================================================
  # [OS]
  @property
  def word_file(self):
    return self._config.get('OS', 'word_file')
  argparser.add_argument('--word_file')
  @property
  def tag_file(self):
    return self._config.get('OS', 'tag_file')
  argparser.add_argument('--tag_file')
  @property
  def rel_file(self):
    return self._config.get('OS', 'rel_file')
  argparser.add_argument('--rel_file')
  @property
  def srl_file(self):
    return self._config.get('OS', 'srl_file')
  argparser.add_argument('--srl_file')
  @property
  def predicates_file(self):
    return self._config.get('OS', 'predicate_file')
  argparser.add_argument('--predicate_file')
  @property
  def domain_file(self):
    return self._config.get('OS', 'domain_file')
  argparser.add_argument('--domain_file')
  @property
  def embed_file(self):
    return self._config.get('OS', 'embed_file')
  argparser.add_argument('--embed_file')
  @property
  def embed_aux_file(self):
    return self._config.get('OS', 'embed_aux_file')
  argparser.add_argument('--embed_aux_file')
  @property
  def train_file(self):
    return self._config.get('OS', 'train_file')
  argparser.add_argument('--train_file')
  @property
  def valid_file(self):
    return self._config.get('OS', 'valid_file')
  argparser.add_argument('--valid_file')
  @property
  def test_file(self):
    return self._config.get('OS', 'test_file')
  argparser.add_argument('--test_file')
  @property
  def save_dir(self):
    return self._config.get('OS', 'save_dir')
  argparser.add_argument('--save_dir')
  @property
  def load_dir(self):
    return self._config.get('OS', 'load_dir')
  argparser.add_argument('--load_dir')
  @property
  def save(self):
    return self._config.getboolean('OS', 'save')
  argparser.add_argument('--save')

  @property
  def gold_dev_props_file(self):
    return self._config.get('OS', 'gold_dev_props_file')
  argparser.add_argument('--gold_dev_props_file')

  @property
  def gold_test_props_file(self):
    return self._config.get('OS', 'gold_test_props_file')
  argparser.add_argument('--gold_test_props_file')

  @property
  def gold_dev_parse_file(self):
    return self._config.get('OS', 'gold_dev_parse_file')
  argparser.add_argument('--gold_dev_parse_file')

  @property
  def gold_test_parse_file(self):
    return self._config.get('OS', 'gold_test_parse_file')
  argparser.add_argument('--gold_test_parse_file')

  @property
  def transition_statistics(self):
    return self._config.get('OS', 'transition_statistics')
  argparser.add_argument('--transition_statistics')

  #=============================================================
  # [Dataset]
  @property
  def cased(self):
    return self._config.getboolean('Dataset', 'cased')
  argparser.add_argument('--cased')
  @property
  def ensure_tree(self):
    return self._config.getboolean('Dataset', 'ensure_tree')
  argparser.add_argument('--ensure_tree')
  @property
  def root_label(self):
    return self._config.get('Dataset', 'root_label')
  argparser.add_argument('--root_label')
  @property
  def add_to_pretrained(self):
    return self._config.getboolean('Dataset', 'add_to_pretrained')
  argparser.add_argument('--add_to_pretrained')
  @property
  def min_occur_count(self):
    return self._config.getint('Dataset', 'min_occur_count')
  argparser.add_argument('--min_occur_count')
  @property
  def minimize_pads(self):
    return self._config.getboolean('Dataset', 'minimize_pads')
  argparser.add_argument('--minimize_pads')
  @property
  def n_bkts(self):
    return self._config.getint('Dataset', 'n_bkts')
  argparser.add_argument('--n_bkts')
  @property
  def n_valid_bkts(self):
    return self._config.getint('Dataset', 'n_valid_bkts')
  argparser.add_argument('--n_valid_bkts')
  @property
  def lines_per_buffer(self):
    return self._config.getint('Dataset', 'lines_per_buffer')
  argparser.add_argument('--lines_per_buffer')

  @property
  def conll(self):
    return self._config.getboolean('Dataset', 'conll')
  argparser.add_argument('--conll')

  @property
  def conll2012(self):
    return self._config.getboolean('Dataset', 'conll2012')
  argparser.add_argument('--conll2012')

  @property
  def train_on_nested(self):
    return self._config.getboolean('Dataset', 'train_on_nested')
  argparser.add_argument('--train_on_nested')

  @property
  def joint_pos_predicates(self):
    return self._config.getboolean('Dataset', 'joint_pos_predicates')
  argparser.add_argument('--joint_pos_predicates')

  @property
  def train_domains(self):
    return self._config.get('Dataset', 'train_domains')
  argparser.add_argument('--train_domains')

  #=============================================================
  # [Layers]
  @property
  def n_recur(self):
    return self._config.getint('Layers', 'n_recur')
  argparser.add_argument('--n_recur')
  @property
  def recur_cell(self):
    from lib import rnn_cells
    return getattr(rnn_cells, self._config.get('Layers', 'recur_cell'))
  argparser.add_argument('--recur_cell')
  @property
  def recur_bidir(self):
    return self._config.getboolean('Layers', 'recur_bidir')
  argparser.add_argument('--recur_bidir')
  @property
  def forget_bias(self):
    if self._config.get('Layers', 'forget_bias') == 'None':
      from lib.linalg import sig_const
      return sig_const
    else:
      return self._config.getfloat('Layers', 'forget_bias')
  argparser.add_argument('--forget_bias')

  #=============================================================
  # [Sizes]
  @property
  def embed_size(self):
    return self._config.getint('Sizes', 'embed_size')
  argparser.add_argument('--embed_size')
  @property
  def predicate_embed_size(self):
    return self._config.getint('Sizes', 'predicate_embed_size')
  argparser.add_argument('--predicate_embed_size')
  @property
  def recur_size(self):
    return self._config.getint('Sizes', 'recur_size')
  argparser.add_argument('--recur_size')
  @property
  def attn_mlp_size(self):
    return self._config.getint('Sizes', 'attn_mlp_size')
  argparser.add_argument('--attn_mlp_size')
  @property
  def class_mlp_size(self):
    return self._config.getint('Sizes', 'class_mlp_size')
  argparser.add_argument('--class_mlp_size')
  @property
  def info_mlp_size(self):
    return self._config.getint('Sizes', 'info_mlp_size')
  argparser.add_argument('--info_mlp_size')

  @property
  def predicate_mlp_size(self):
    return self._config.getint('Sizes', 'predicate_mlp_size')
  argparser.add_argument('--predicate_mlp_size')

  @property
  def predicate_pred_mlp_size(self):
    return self._config.getint('Sizes', 'predicate_pred_mlp_size')
  argparser.add_argument('--predicate_pred_mlp_size')

  @property
  def role_mlp_size(self):
    return self._config.getint('Sizes', 'role_mlp_size')
  argparser.add_argument('--role_mlp_size')

  #=============================================================
  # [Functions]
  @property
  def recur_func(self):
    func = self._config.get('Functions', 'recur_func')
    if func == 'identity':
      return tf.identity
    elif func == 'leaky_relu':
      return lambda x: tf.maximum(.1*x, x)
    else:
      return getattr(tf.nn, func)
  argparser.add_argument('--recur_func')
  @property
  def info_func(self):
    func = self._config.get('Functions', 'info_func')
    if func == 'identity':
      return tf.identity
    elif func == 'leaky_relu':
      return lambda x: tf.maximum(.1*x, x)
    else:
      return getattr(tf.nn, func)
  argparser.add_argument('--info_func')
  @property
  def mlp_func(self):
    func = self._config.get('Functions', 'mlp_func')
    if func == 'identity':
      return tf.identity
    elif func == 'leaky_relu':
      return lambda x: tf.maximum(.1*x, x)
    else:
      return getattr(tf.nn, func)
  argparser.add_argument('--mlp_func')

  #=============================================================
  # [Regularization]
  @property
  def word_l2_reg(self):
    return self._config.getfloat('Regularization', 'word_l2_reg')
  argparser.add_argument('--word_l2_reg')

  #=============================================================
  # [Dropout]
  @property
  def word_keep_prob(self):
    return self._config.getfloat('Dropout', 'word_keep_prob')
  argparser.add_argument('--word_keep_prob')
  @property
  def tag_keep_prob(self):
    return self._config.getfloat('Dropout', 'tag_keep_prob')
  argparser.add_argument('--tag_keep_prob')
  @property
  def rel_keep_prob(self):
    return self._config.getfloat('Dropout', 'rel_keep_prob')
  argparser.add_argument('--rel_keep_prob')
  @property
  def recur_keep_prob(self):
    return self._config.getfloat('Dropout', 'recur_keep_prob')
  argparser.add_argument('--recur_keep_prob')
  @property
  def cell_include_prob(self):
    return self._config.getfloat('Dropout', 'cell_include_prob')
  argparser.add_argument('--cell_include_prob')
  @property
  def hidden_include_prob(self):
    return self._config.getfloat('Dropout', 'hidden_include_prob')
  argparser.add_argument('--hidden_include_prob')
  @property
  def ff_keep_prob(self):
    return self._config.getfloat('Dropout', 'ff_keep_prob')
  argparser.add_argument('--ff_keep_prob')
  @property
  def mlp_keep_prob(self):
    return self._config.getfloat('Dropout', 'mlp_keep_prob')
  argparser.add_argument('--mlp_keep_prob')
  @property
  def info_keep_prob(self):
    return self._config.getfloat('Dropout', 'info_keep_prob')
  argparser.add_argument('--info_keep_prob')

  @property
  def attn_dropout(self):
    return self._config.getfloat('Dropout', 'attn_dropout')
  argparser.add_argument('--attn_dropout')
  @property
  def prepost_dropout(self):
    return self._config.getfloat('Dropout', 'prepost_dropout')
  argparser.add_argument('--prepost_dropout')
  @property
  def relu_dropout(self):
    return self._config.getfloat('Dropout', 'relu_dropout')
  argparser.add_argument('--relu_dropout')
  @property
  def input_dropout(self):
    return self._config.getfloat('Dropout', 'input_dropout')
  argparser.add_argument('--input_dropout')

  #=============================================================
  # [Learning rate]
  @property
  def learning_rate(self):
    return self._config.getfloat('Learning rate', 'learning_rate')
  argparser.add_argument('--learning_rate')
  @property
  def decay(self):
    return self._config.getfloat('Learning rate', 'decay')
  argparser.add_argument('--decay')
  @property
  def decay_steps(self):
    return self._config.getfloat('Learning rate', 'decay_steps')
  argparser.add_argument('--decay_steps')
  @property
  def clip(self):
    return self._config.getfloat('Learning rate', 'clip')
  argparser.add_argument('--clip')
  @property
  def warmup_steps(self):
    return self._config.getint('Learning rate', 'warmup_steps')
  argparser.add_argument('--warmup_steps')

  #=============================================================
  # [Radam]
  @property
  def mu(self):
    return self._config.getfloat('Radam', 'mu')
  argparser.add_argument('--mu')
  @property
  def nu(self):
    return self._config.getfloat('Radam', 'nu')
  argparser.add_argument('--nu')
  @property
  def gamma(self):
    return self._config.getfloat('Radam', 'gamma')
  argparser.add_argument('--gamma')
  @property
  def epsilon(self):
    return self._config.getfloat('Radam', 'epsilon')
  argparser.add_argument('--epsilon')
  @property
  def chi(self):
    return self._config.getfloat('Radam', 'chi')
  argparser.add_argument('--chi')

  #=============================================================
  # [Training]
  @property
  def pretrain_iters(self):
    return self._config.getint('Training', 'pretrain_iters')
  argparser.add_argument('--pretrain_iters')
  @property
  def train_iters(self):
    return self._config.getint('Training', 'train_iters')
  argparser.add_argument('--train_iters')
  @property
  def train_batch_size(self):
    return self._config.getint('Training', 'train_batch_size')
  argparser.add_argument('--train_batch_size')
  @property
  def test_batch_size(self):
    return self._config.getint('Training', 'test_batch_size')
  argparser.add_argument('--test_batch_size')
  @property
  def validate_every(self):
    return self._config.getint('Training', 'validate_every')
  argparser.add_argument('--validate_every')
  @property
  def print_every(self):
    return self._config.getint('Training', 'print_every')
  argparser.add_argument('--print_every')
  @property
  def save_every(self):
    return self._config.getint('Training', 'save_every')
  argparser.add_argument('--save_every')
  @property
  def per_process_gpu_memory_fraction(self):
    return self._config.getfloat('Training', 'per_process_gpu_memory_fraction')
  argparser.add_argument('--per_process_gpu_memory_fraction')
  @property
  def eval_criterion(self):
    return self._config.get('Training', 'eval_criterion')
  argparser.add_argument('--eval_criterion')

  @property
  def roots_penalty(self):
    return self._config.getfloat('Training', 'roots_penalty')
  argparser.add_argument('--roots_penalty')
  @property
  def pairs_penalty(self):
    return self._config.getfloat('Training', 'pairs_penalty')
  argparser.add_argument('--pairs_penalty')
  @property
  def svd_penalty(self):
    return self._config.getfloat('Training', 'svd_penalty')
  argparser.add_argument('--svd_penalty')
  @property
  def mask_roots(self):
    return self._config.getboolean('Training', 'mask_roots')
  argparser.add_argument('--mask_roots')
  @property
  def mask_pairs(self):
    return self._config.getboolean('Training', 'mask_pairs')
  argparser.add_argument('--mask_pairs')

  @property
  def viterbi_train(self):
    return self._config.getboolean('Training', 'viterbi_train')
  argparser.add_argument('--viterbi_train')

  @property
  def viterbi_decode(self):
    return self._config.getboolean('Training', 'viterbi_decode')
  argparser.add_argument('--viterbi_decode')

  @property
  def predicate_loss_penalty(self):
    return self._config.getfloat('Training', 'predicate_loss_penalty')
  argparser.add_argument('--predicate_loss_penalty')

  @property
  def role_loss_penalty(self):
    return self._config.getfloat('Training', 'role_loss_penalty')
  argparser.add_argument('--role_loss_penalty')

  @property
  def rel_loss_penalty(self):
    return self._config.getfloat('Training', 'rel_loss_penalty')
  argparser.add_argument('--rel_loss_penalty')

  @property
  def arc_loss_penalty(self):
    return self._config.getfloat('Training', 'arc_loss_penalty')
  argparser.add_argument('--arc_loss_penalty')

  @property
  def add_pos_to_input(self):
    return self._config.getboolean('Training', 'add_pos_to_input')
  argparser.add_argument('--add_pos_to_input')

  @property
  def add_predicates_to_input(self):
    return self._config.getboolean('Training', 'add_predicates_to_input')
  argparser.add_argument('--add_predicates_to_input')

  @property
  def save_attn_weights(self):
    return self._config.getboolean('Training', 'save_attn_weights')
  argparser.add_argument('--save_attn_weights')

  #=============================================================
  # [Transformer]
  @property
  def num_heads(self):
    return self._config.getint('Training', 'num_heads')
  argparser.add_argument('--num_heads')
  @property
  def head_size(self):
    return self._config.getint('Training', 'head_size')
  argparser.add_argument('--head_size')
  @property
  def cnn_dim(self):
    return self._config.getint('Training', 'cnn_dim')
  argparser.add_argument('--cnn_dim')
  @property
  def cnn_layers(self):
    return self._config.getint('Training', 'cnn_layers')
  argparser.add_argument('--cnn_layers')
  @property
  def relu_hidden_size(self):
    return self._config.getint('Training', 'relu_hidden_size')
  argparser.add_argument('--relu_hidden_size')

  @property
  def svd_tree(self):
    return self._config.getboolean('Training', 'svd_tree')
  argparser.add_argument('--svd_tree')

  @property
  def cnn2d_layers(self):
    return self._config.getint('Training', 'cnn2d_layers')
  argparser.add_argument('--cnn2d_layers')
  @property
  def cnn_dim_2d(self):
    return self._config.getint('Training', 'cnn_dim_2d')
  argparser.add_argument('--cnn_dim_2d')

  @property
  def num_blocks(self):
    return self._config.getint('Training', 'num_blocks')
  argparser.add_argument('--num_blocks')

  @property
  def dist_model(self):
    return self._config.get('Training', 'dist_model')
  argparser.add_argument('--dist_model')

  @property
  def lstm_residual(self):
    return self._config.getboolean('Training', 'lstm_residual')
  argparser.add_argument('--lstm_residual')

  @property
  def cnn_residual(self):
    return self._config.getboolean('Training', 'cnn_residual')
  argparser.add_argument('--cnn_residual')

  @property
  def parse_update_proportion(self):
    return self._config.getfloat('Training', 'parse_update_proportion')
  argparser.add_argument('--parse_update_proportion')

  @property
  def multitask_penalties(self):
    return self._config.get('Training', 'multitask_penalties')
  argparser.add_argument('--multitask_penalties')

  @property
  def multitask_layers(self):
    return self._config.get('Training', 'multitask_layers')
  argparser.add_argument('--multitask_layers')

  @property
  def predicate_str(self):
    return self._config.get('Training', 'predicate_str')
  argparser.add_argument('--predicate_str')

  @property
  def inject_manual_attn(self):
    return self._config.getboolean('Training', 'inject_manual_attn')
  argparser.add_argument('--inject_manual_attn')

  @property
  def train_pos(self):
    return self._config.getboolean('Training', 'train_pos')
  argparser.add_argument('--train_pos')
  @property
  def pos_layer(self):
    return self._config.getint('Training', 'pos_layer')
  argparser.add_argument('--pos_layer')

  @property
  def predicate_layer(self):
    return self._config.getint('Training', 'predicate_layer')
  argparser.add_argument('--predicate_layer')

  @property
  def pos_penalty(self):
    return self._config.getfloat('Training', 'pos_penalty')
  argparser.add_argument('--pos_penalty')

  @property
  def parse_layer(self):
    return self._config.getint('Training', 'parse_layer')
  argparser.add_argument('--parse_layer')

  @property
  def eval_parse(self):
    return self._config.getboolean('Training', 'eval_parse')
  argparser.add_argument('--eval_parse')

  @property
  def eval_srl(self):
    return self._config.getboolean('Training', 'eval_srl')
  argparser.add_argument('--eval_srl')

  @property
  def eval_by_domain(self):
    return self._config.getboolean('Training', 'eval_by_domain')
  argparser.add_argument('--eval_by_domain')

  @property
  def num_capsule_heads(self):
    return self._config.getint('Training', 'num_capsule_heads')
  argparser.add_argument('--num_capsule_heads')

  @property
  def gold_attn_at_train(self):
    return self._config.getboolean('Training', 'gold_attn_at_train')
  argparser.add_argument('--gold_attn_at_train')

  @property
  def eval_single_token_sents(self):
    return self._config.getboolean('Training', 'eval_single_token_sents')
  argparser.add_argument('--eval_single_token_sents')

  @property
  def hard_attn(self):
    return self._config.getboolean('Training', 'hard_attn')
  argparser.add_argument('--hard_attn')

  @property
  def full_parse(self):
    return self._config.getboolean('Training', 'full_parse')
  argparser.add_argument('--full_parse')

  @property
  def use_elmo(self):
    return self._config.getboolean('Training', 'use_elmo')
  argparser.add_argument('--use_elmo')

  @property
  def sampling_schedule(self):
    return self._config.get('Training', 'sampling_schedule')
  argparser.add_argument('--sampling_schedule')

  @property
  def sample_prob(self):
    return self._config.getfloat('Training', 'sample_prob')
  argparser.add_argument('--sample_prob')

  @property
  def max_test_batch_size(self):
    return self._config.getint('Training', 'max_test_batch_size')
  argparser.add_argument('--max_test_batch_size')

  @property
  def max_dev_batch_size(self):
    return self._config.getint('Training', 'max_dev_batch_size')
  argparser.add_argument('--max_dev_batch_size')

  @property
  def ff_kernel(self):
    return self._config.getint('Training', 'ff_kernel')
  argparser.add_argument('--ff_kernel')

  @property
  def one_example_per_predicate(self):
    return self._config.getboolean('Training', 'one_example_per_predicate')
  argparser.add_argument('--one_example_per_predicate')

  @property
  def srl_simple_tagging(self):
    return self._config.getboolean('Training', 'srl_simple_tagging')
  argparser.add_argument('--srl_simple_tagging')

  @property
  def label_smoothing(self):
    return self._config.getfloat('Training', 'label_smoothing')
  argparser.add_argument('--label_smoothing')
