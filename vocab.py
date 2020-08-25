#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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





import os
import sys
from collections import Counter

import numpy as np
import tensorflow as tf

from configurable import Configurable

#***************************************************************
class Vocab(Configurable):
  """"""
  
  # SPECIAL_TOKENS = ('<PAD>', '<ROOT>', '<UNK>')
  # START_IDX = len(SPECIAL_TOKENS)
  # PAD, ROOT, UNK = list(range(START_IDX))
  PAD = 0
  ROOT = 1
  UNK = 2
  
  #=============================================================
  def __init__(self, vocab_file, conll_idx, embed_size, *args, **kwargs):
    """"""

    self._vocab_file = vocab_file
    self._conll_idx = conll_idx
    # global_step = kwargs.pop('global_step', None)
    cased = kwargs.pop('cased', None)
    self._use_pretrained = kwargs.pop('use_pretrained', False)
    super(Vocab, self).__init__(*args, **kwargs)

    self.train_domains_set = set(self.train_domains.split(',')) if self.train_domains != '-' and self.name != "Domains" else set()

    self._embed_size = embed_size

    self.SPECIAL_TOKENS = ('<PAD>', '<UNK>') #, '<ROOT>', '<UNK>')

    if cased is None:
      self._cased = super(Vocab, self).cased
    else:
      self._cased = cased
    if self.name == 'Tags':
      self.SPECIAL_TOKENS = ('PAD',)
      # self.SPECIAL_TOKENS = ('PAD', 'ROOT', 'UNK')
    elif self.name == 'Rels':
      self.SPECIAL_TOKENS = ('PAD',) #, self.root_label)
      # self.SPECIAL_TOKENS = ('pad', self.root_label, 'unk')
    elif self.name == 'Predicates':
      self.SPECIAL_TOKENS = ('PAD',)
    elif self.name == 'SRLs':
      self.SPECIAL_TOKENS = ('PAD',)

    self.START_IDX = len(self.SPECIAL_TOKENS)

    self.predicate_true_start_idx = self.START_IDX
    
    self._counts = Counter()
    self._str2idx = {}
    self._idx2str = {}
    self.trainable_embeddings = None
    if self.use_pretrained:
      self._str2embed = {}
      self._embed2str = {}
      self.pretrained_embeddings = None
    
    if os.path.isfile(self.vocab_file):
      self.load_vocab_file()
    else:
      self.add_train_file()
      self.save_vocab_file()
    if self.use_pretrained:
      self.load_embed_file()
    self._finalize()
    
    # if global_step is not None:
    #   self._global_sigmoid = 1-tf.nn.sigmoid(3*(2*global_step/(self.train_iters-1)-1))
    # else:
    #   self._global_sigmoid = 1
    return
  
  #=============================================================
  def add(self, counts, word, count=1):
    """"""
    
    if not self.cased:
      word = word.lower()
    
    counts[word] += int(count)
    return
  
  #=============================================================
  def init_str2idx(self):
    return dict(list(zip(self.SPECIAL_TOKENS, list(range(self.START_IDX)))))
  def init_idx2str(self):
    return dict(list(zip(list(range(self.START_IDX)), self.SPECIAL_TOKENS)))
  
  #=============================================================
  def index_vocab(self, counts):
    """"""
    
    str2idx = self.init_str2idx()
    idx2str = self.init_idx2str()
    cur_idx = self.START_IDX
    self.predicate_true_start_idx = cur_idx
    for word, count in self.sorted_vocab(counts):
      if (count >= self.min_occur_count or self.name == 'SRLs') and word not in str2idx:
        str2idx[word] = cur_idx
        idx2str[cur_idx] = word
        cur_idx += 1
    return str2idx, idx2str

  # =============================================================
  def index_vocab_joint(self, counts):
    """"""

    str2idx = self.init_str2idx()
    idx2str = self.init_idx2str()
    cur_idx = self.START_IDX
    add_to_end = []
    for word, count in self.sorted_vocab(counts):
      if (count >= self.min_occur_count or self.name == 'SRLs') and word not in str2idx:
        if word.split('/')[0] == "True":
          add_to_end.append(word)
        else:
          str2idx[word] = cur_idx
          idx2str[cur_idx] = word
          cur_idx += 1
    self.predicate_true_start_idx = cur_idx - 1
    for word in add_to_end:
      str2idx[word] = cur_idx
      idx2str[cur_idx] = word
      cur_idx += 1
    return str2idx, idx2str
  
  #=============================================================
  @staticmethod
  def sorted_vocab(counts):
    """"""
    
    buff = []
    partial = []
    words_and_counts = counts.most_common()
    words_and_counts.append( (None, None) )
    for word_and_count in words_and_counts:
      if (not buff) or buff[-1][1] == word_and_count[1]:
        buff.append(word_and_count)
      else:
        buff.sort()
        partial.extend(buff)
        buff = [word_and_count]
    return partial
  
  #=============================================================
  def add_train_file(self):
    """"""
    counts = Counter()

    with open(self.train_file, 'r') as f:
      buff = []
      for line_num, line in enumerate(f):
        line = line.strip().split()
        # print(line)
        if line and (not self.train_domains_set or line[0].split('/')[0] in self.train_domains_set or self.name == "Tags"):
          if self.conll and len(line) == 10:
            if hasattr(self.conll_idx, '__iter__'):
              for idx in self.conll_idx:
                self.add(counts, line[idx])
            else:
              self.add(counts, line[self.conll_idx])
          elif self.conll2012: #and len(line) > 1:
            if hasattr(self.conll_idx, '__iter__'):
              if self.name == "Predicates":
                actual = "False" if line[self.conll_idx[0]] == '-' else "True"
                actual = actual + "/" + line[self.conll_idx[1]]
                self.add(counts, actual)
              else:
                for idx in self.conll_idx:
                  if idx < len(line) and (self.name != 'SRLs' or (idx != len(line)-1 and (self.train_on_nested or '/' not in line[idx]))):
                    # print("adding ", line[idx])
                    self.add(counts, line[idx])
            else:
              if self.name == "Predicates":
                actual = "False" if line[self.conll_idx] == '-' else "True"
                self.add(counts, actual)
              elif self.name == "Domains":
                actual = line[self.conll_idx].split('/')[0]
                self.add(counts, actual)
              else:
                self.add(counts, line[self.conll_idx])
          else:
            print('The training file is misformatted at line %d (had %d columns, expected %d)' % (line_num+1, len(line), 13))

      # # add all the labels
      # if self.name == "SRLs":
      #   with open(self.valid_file, 'r') as f:
      #     for line_num, line in enumerate(f):
      #       line = line.strip().split()
      #       if line:
      #         if hasattr(self.conll_idx, '__iter__'):
      #           for idx in self.conll_idx:
      #             if idx < len(line)-1 and (self.train_on_nested or '/' not in line[idx]):
      #               # print("adding ", line[idx])
      #               self.add(counts, line[idx])
      #   with open(self.test_file, 'r') as f:
      #     for line_num, line in enumerate(f):
      #       line = line.strip().split()
      #       if line:
      #         if hasattr(self.conll_idx, '__iter__'):
      #           for idx in self.conll_idx:
      #             if idx < len(line)-1 and (self.train_on_nested or '/' not in line[idx]):
      #               # print("adding ", line[idx])
      #               self.add(counts, line[idx])


    self._counts = counts
    self._str2idx, self._idx2str = self.index_vocab_joint(counts) if self.joint_pos_predicates and self.name == "Predicates" else self.index_vocab(counts)
    return

  #=============================================================
  def load_embed_file(self):
    """"""
    
    self._str2embed = self.init_str2idx()
    self._embed2str = self.init_idx2str()
    embeds = []
    with open(self.embed_file, 'r') as f:
      cur_idx = self.START_IDX
      for line_num, line in enumerate(f):
        line = line.strip().split()
        if line:
          try:
            self._str2embed[line[0]] = cur_idx
            self._embed2str[cur_idx] = line[0]
            embeds.append(line[1:])
            cur_idx += 1
          except:
            raise ValueError('The embedding file is misformatted at line %d' % (line_num+1))
    self.pretrained_embeddings = np.array(embeds, dtype=np.float32)
    self.pretrained_embeddings = np.pad(self.pretrained_embeddings, ((self.START_IDX, 0), (0, 0)), 'constant')
    if os.path.isfile(self.embed_aux_file):
      with open(self.embed_aux_file, 'r') as f:
        for line in f:
          line = line.strip().split()
          if self.START_IDX > 0 and line[0] == self.SPECIAL_TOKENS[0]:
            self.pretrained_embeddings[0] = np.array(line[1:], dtype=np.float32)
          elif self.START_IDX > 1 and line[0] == self.SPECIAL_TOKENS[1]:
            self.pretrained_embeddings[1] = np.array(line[1:], dtype=np.float32)
          elif self.START_IDX > 2 and line[0] == self.SPECIAL_TOKENS[2]:
            self.pretrained_embeddings[2] = np.array(line[1:], dtype=np.float32)
    return
  
  #=============================================================
  def save_vocab_file(self):
    """"""
    
    with open(self.vocab_file, 'w') as f:
      for word, count in self.sorted_vocab(self._counts):
        f.write('%s\t%d\n' % (word, count))
    return
  
  #=============================================================
  def load_vocab_file(self):
    """"""
    
    counts = Counter()
    with open(self.vocab_file, 'r') as f:
      for line_num, line in enumerate(f):
        line = line.strip().split('\t')
        if line:
          if len(line) == 1:
            line.insert(0, '')
          if len(line) == 2:
            self.add(counts, line[0], line[1])
          else:
            raise ValueError('The vocab file is misformatted at line %d' % (line_num+1))
    self._counts = counts
    self._str2idx, self._idx2str = self.index_vocab_joint(counts) if self.joint_pos_predicates and self.name == "Predicates" else self.index_vocab(counts)
    return
  
  #=============================================================
  def get_embed(self, key):
    """"""
    
    return self._embed2str[key]
  
  #=============================================================
  def _finalize(self):
    """"""
    
    if self.use_pretrained and self.name == "Words":
      initializer = tf.zeros_initializer()
      embed_size = self.pretrained_embeddings.shape[1]
    else:
      initializer = tf.random_normal_initializer()
      embed_size = self._embed_size
    
    with tf.device('/cpu:0'):
      with tf.variable_scope(self.name):
        if self.name == "Words" and self._embed_size > 0 and self.use_pretrained and not self.add_to_pretrained:
          self.pretrained_embeddings /= np.std(self.pretrained_embeddings)
          self.trainable_embeddings = tf.Variable(self.pretrained_embeddings, trainable=True, name='Trainable')
          print("Loaded pre-trained embeddings. Trainable: True")
        else:
          if self._embed_size > 0:
            self.trainable_embeddings = tf.get_variable('Trainable', shape=(len(self._str2idx), embed_size), initializer=initializer)
          if self.use_pretrained:
            self.pretrained_embeddings /= np.std(self.pretrained_embeddings)
            self.pretrained_embeddings = tf.Variable(self.pretrained_embeddings, trainable=False, name='Pretrained')
            print("Loaded pre-trained embeddings. Trainable: False")
    return
  
  #=============================================================
  def embedding_lookup(self, inputs, pret_inputs=None, moving_params=None):
    """"""
    
    if moving_params is not None:
      trainable_embeddings = moving_params.average(self.trainable_embeddings)
    else:
      trainable_embeddings = self.trainable_embeddings
    
    embed_input = tf.nn.embedding_lookup(trainable_embeddings, inputs)
    if moving_params is None:
      tf.add_to_collection('Weights', embed_input)
    if self.use_pretrained and pret_inputs is not None:
      return embed_input, tf.nn.embedding_lookup(self.pretrained_embeddings, pret_inputs)
    else:
      return embed_input
  
  #=============================================================
  def weighted_average(self, inputs, moving_params=None):
    """"""
    
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = len(self)
    
    if moving_params is not None:
      trainable_embeddings = moving_params.average(self.trainable_embeddings)
    else:
      trainable_embeddings = self.trainable_embeddings
    
    embed_input = tf.matmul(tf.reshape(inputs, [-1, input_size]),
                            trainable_embeddings)
    embed_input = tf.reshape(embed_input, tf.stack([batch_size, bucket_size, self._embed_size]))
    embed_input.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(self._embed_size)])
    if moving_params is None:
      tf.add_to_collection('Weights', embed_input)
    return embed_input
  
  #=============================================================
  @property
  def vocab_file(self):
    return self._vocab_file
  @property
  def use_pretrained(self):
    return self._use_pretrained
  @property
  def cased(self):
    return self._cased
  @property
  def conll_idx(self):
    return self._conll_idx
  # @property
  # def global_sigmoid(self):
  #   return self._global_sigmoid
  
  #=============================================================
  def keys(self):
    return list(self._str2idx.keys())
  def values(self):
    return list(self._str2idx.values())
  def items(self):
    return list(self._str2idx.items())
  
  #=============================================================
  def __getitem__(self, key):
    if isinstance(key, str):
      if not self.cased:
        key = key.lower()
      if self.use_pretrained:
        return (self._str2idx.get(key, self.UNK), self._str2embed.get(key, self.UNK))
      else:
        return (self._str2idx.get(key, self.UNK),)
    elif isinstance(key, (int, np.int32, np.int64)):
      if key not in self._idx2str:
        if self.name != "Words":
          raise ValueError("idx %d not in vocab %s" % (key, self.name))
        else:
          print("idx %d not in vocab %s" % (key, self.name))
      return self._idx2str.get(key, self.UNK)
    elif hasattr(key, '__iter__'):
      return tuple(self[k] for k in key)
    else:
      raise ValueError('key to Vocab.__getitem__ must be (iterable of) string or integer')
    return
  
  def __contains__(self, key):
    if isinstance(key, str):
      if not self.cased:
        key = key.lower()
      return key in self._str2idx
    elif isinstance(key, int):
      return key in self._idx2str
    else:
      raise ValueError('key to Vocab.__contains__ must be string or integer')
    return
  
  def __len__(self):
    return len(self._str2idx)
  
  def __iter__(self):
    return (key for key in self._str2idx)
  
