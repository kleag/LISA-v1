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





import numpy as np
import tensorflow as tf
from collections import Counter

from lib.etc.k_means import KMeans
from configurable import Configurable
from vocab import Vocab
from metabucket import Metabucket

#***************************************************************
class Dataset(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, filename, vocabs, builder, *args, **kwargs):
    """"""
    
    super(Dataset, self).__init__(*args, **kwargs)
    self.vocabs = vocabs

    self.train_domains_set = set(self.train_domains.split(',')) if self.train_domains != '-' and self.name == "Trainset" else set()
    print("Loading training data from domains:", self.train_domains_set if self.train_domains_set else "all")

    self._file_iterator = self.file_iterator(filename)
    self._train = (filename == self.train_file)
    self._metabucket = Metabucket(self._config, n_bkts=self.n_bkts)
    self._data = None
    self.rebucket()

    if self.use_elmo:
      from lib.models import ElmoLSTMEncoder
      with tf.variable_scope(tf.get_variable_scope(), reuse=(self.name != "Trainset")):
        self.elmo_encoder = ElmoLSTMEncoder(self)

    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='inputs')
    self.targets = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='targets')
    self.step = tf.placeholder_with_default(0., shape=None, name='step')
    self.builder = builder()
  
  #=============================================================
  def file_iterator(self, filename):
    """"""
    
    #print(f"Dataset.file_iterator {filename}")
    with open(filename) as f:
      if self.lines_per_buffer > 0:
        buff = [[]]
        while True:
          line = f.readline()
          while line:
            #print(f"Dataset.file_iterator line: {line}")
            line = line.strip().split()
            if line and (not self.train_domains_set or line[0].split('/')[0] in self.train_domains):
              buff[-1].append(line)
            else:
              if len(buff) < self.lines_per_buffer:
                if len(buff[-1]) > 0:
                  buff.append([])
                else:
                  buff[-1] = []
              else:
                break
            line = f.readline()
          if not line:
            f.seek(0)
          else:
            buff = self._process_buff(buff)
            yield buff
            line = line.strip().split()
            if line:
              buff = [[line]]
            else:
              buff = [[]]
      else:
        buff = [[]]
        for line in f:
          line = line.strip().split()
          #print(f"Dataset.file_iterator line: {line}")
          if line and (not self.train_domains_set or line[0].split('/')[0] in self.train_domains):
            buff[-1].append(line)
          else:
            if len(buff[-1]) > 0:
              buff.append([])
            else:
              buff[-1] = []
        if buff[-1] == []:
          buff.pop()
        buff = self._process_buff(buff)
        while True:
          yield buff
  
  #=============================================================
  def _process_buff(self, buff):
    """"""

    # tmp_f = open("debug_data_%s" % self.name, 'w')
    
    words, tags, rels, srls, predicates, domains = self.vocabs
    srl_start_field = srls.conll_idx[0]
    sents = 0
    toks = 0
    examples = 0
    total_predicates = 0
    buff2 = []
    for i, sent in enumerate(buff):
      # if not self.conll2012 or (self.conll2012 and len(list(sent)) > 1):
      #print(f"Dataset._process_buff sent {i} of length {len(sent)}: {sent}")
      sents += 1
      sent_len = len(sent)
      num_fields = len(sent[0])
      srl_take_indices = [idx for idx in list(range(srl_start_field, srl_start_field + sent_len)) if idx < num_fields - 1 and (self.train_on_nested or np.all(['/' not in sent[j][idx] for j in list(range(sent_len))]))]
      predicate_indices = []
      for j, token in enumerate(sent):
        toks += 1
        if self.conll:
          word, tag1, tag2, head, rel = token[words.conll_idx], token[tags.conll_idx[0]], token[tags.conll_idx[1]], token[6], token[rels.conll_idx]
          if rel == 'root':
            head = j
          else:
            head = int(head) - 1
          buff[i][j] = (word,) + words[word] + tags[tag1] + tags[tag2] + (head,) + rels[rel]
        elif self.conll2012:
          #print(f"Dataset._process_buff token {j}:{token}")
          word, auto_tag, gold_tag, head, rel = token[words.conll_idx], token[tags.conll_idx[0]], token[tags.conll_idx[1]], token[6], token[rels.conll_idx]
          #print(f"Dataset token read {word}, {auto_tag}, {gold_tag}, {head}, {rel}")
          domain = token[0].split('/')[0]
          # print(word, tag1, tag2, head, rel)
          if rel == 'root':
            head = j
          else:
            head = int(head) - 1

          # srl_fields = [token[idx] if idx < len(token)-1 else 'O' for idx in list(range(srl_start_field, srl_start_field + sent_len))
          srl_fields = [token[idx] for idx in srl_take_indices] # todo can we use fancy indexing here?
          srl_fields += ['O'] * (sent_len - len(srl_take_indices))
          srl_tags = [srls[s][0] for s in srl_fields]

          if self.joint_pos_predicates:
            is_predicate = token[predicates.conll_idx[0]] != '-' and (self.train_on_nested or self.predicate_str in srl_fields)
            tok_predicate_str = str(is_predicate) + '/' + gold_tag
          else:
            is_predicate = token[predicates.conll_idx] != '-' and (self.train_on_nested or self.predicate_str in srl_fields)
            tok_predicate_str = str(is_predicate)

          if is_predicate:
            predicate_indices.append(j)

          buff[i][j] = (word,) + words[word] + tags[auto_tag] + predicates[tok_predicate_str] + domains[domain] + (sents,) + tags[gold_tag] + (head,) + rels[rel] + tuple(srl_tags)
          #print(f"Dataset buff[{i}][{j}] = {buff[i][j]}")
          #print(f"Dataset buff {word}, {word}, {auto_tag}, {tok_predicate_str}, {domain}, {sents}, {gold_tag}, {head}, {rel}, {srl_tags}")

      # Expand sentences into one example per predicate
      if self.one_example_per_predicate:
        # grab the sent
        # should be sent_len x sent_elements
        sent = np.array(buff[i])
        # print(sent)
        is_predicate_idx = 4
        srl_start_idx = 10
        word_part = sent[:, 0].astype('O')
        srl_part = sent[:, srl_start_idx:].astype(np.int32)
        rest_part = sent[:, 1:srl_start_idx].astype(np.int32)
        # print("orig sent (%d):" % len(predicate_indices), sent[:, :8+len(predicate_indices)])
        # print("orig preds:", [map(lambda x: srls[int(x)], t) for t in sent[:, srl_start_idx:srl_start_idx+len(predicate_indices)]])
        if predicate_indices:
          for k, p_idx in enumerate(predicate_indices):
            # should be sent_len x sent_elements
            rest_part[:, is_predicate_idx-1] = predicates["False"][0]
            rest_part[p_idx, is_predicate_idx-1] = predicates["True"][0]
            correct_srls = srl_part[:, k]
            new_sent = np.concatenate([np.expand_dims(word_part, -1), rest_part, np.expand_dims(correct_srls, -1)], axis=1)
            buff2.append(new_sent)
            # print("new sent:", new_sent)
            # print("new preds:", map(lambda x: srls[int(x)], new_sent[:, -1]))
            # tokens_str = ' '.join(word_part)
            # labels_str = ' '.join(map(lambda x: srls[x], correct_srls))
            ## idx, tokens, labels
            # print("%d %s ||| %s" % (p_idx, tokens_str, labels_str), file=tmp_f)
            total_predicates += 1
            examples += 1
        else:
           new_sent = np.concatenate([np.expand_dims(word_part, -1), rest_part], axis=1)
           buff2.append(new_sent)
           examples += 1
      # else:
      #   buff2.append(np.concatenate[np.expand_dims(word_part, -1), rest_part, srl_part], axis=1) #(sent[0],) + map(int, sent[1:]))
      #   examples += 1
    # tmp_f.close()
    if self.one_example_per_predicate:
      print("Loaded %d sentences with %d tokens, %d examples (%d predicates) (%s)" % (sents, toks, examples, total_predicates, self.name))
      return buff2
    else:
      print(f"Loaded {sents} sentences with {toks} tokens {self.name}")
      return buff
  
  #=============================================================
  def reset(self, sizes):
    """"""
    
    self._data = []
    self._targets = []
    self._metabucket.reset(sizes)
    return
  
  #=============================================================
  def rebucket(self):
    """"""

    buff = next(self._file_iterator)
    len_cntr = Counter()
    
    for sent in buff:
      len_cntr[len(sent)] += 1
    n_bkts = self.n_bkts if len(len_cntr) >= self.n_bkts else len(len_cntr)
    self.reset(KMeans(n_bkts, len_cntr).splits)
    
    for sent in buff:
      self._metabucket.add(sent)
    self._finalize()
    return
  
  #=============================================================
  def _finalize(self):
    """"""
    
    self._metabucket._finalize()
    return
  
  #=============================================================
  def get_minibatches(self, batch_size, input_idxs, target_idxs, shuffle=True):
    """"""
    
    minibatches = []
    for bkt_idx, bucket in enumerate(self._metabucket):
      if batch_size == 0:
        n_splits = 1
      else:
        n_tokens = len(bucket) * bucket.size
        n_splits = max(n_tokens // batch_size, 1)
      if shuffle:
        range_func = np.random.permutation
      else:
        range_func = np.arange
      arr_sp = np.array_split(range_func(len(bucket)), n_splits)
      for bkt_mb in arr_sp:
        minibatches.append( (bkt_idx, bkt_mb) )
    if shuffle:
      np.random.shuffle(minibatches)
    for bkt_idx, bkt_mb in minibatches:
      feed_dict = {}
      data = self[bkt_idx].data[bkt_mb]
      sents = self[bkt_idx].sents[bkt_mb]
      maxlen = np.max(np.sum(np.greater(data[:,:,0], 0), axis=1))

      # np.set_printoptions(threshold=np.nan)
      # print("maxlen", maxlen)
      # print("maxlen+max(target_idxs)", maxlen+max(target_idxs))
      # print("data.shape[2]", data.shape[2])
      # targets = data[:,:maxlen,min(target_idxs):maxlen+max(target_idxs)+1]
      # print("data shape", targets.shape)
      # print("data[:,:,3:] shape", targets[:,:,3:].shape)

      feed_dict.update({
        self.inputs: data[:,:maxlen,input_idxs],
        self.targets: data[:,:maxlen,min(target_idxs):maxlen+max(target_idxs)+1]
      })
      if self.use_elmo:
        feed_dict = self.elmo_encoder.get_feed_dict(feed_dict, sents)
      #print(f"Dataset.get_minibatches yields {feed_dict}")
      yield feed_dict, sents
  
  #=============================================================
  @property
  def n_bkts(self):
    if self._train:
      return super(Dataset, self).n_bkts
    else:
      return super(Dataset, self).n_valid_bkts

  # =============================================================
  def max_batch_size(self):
    print([b._data.shape[0] for b in self._metabucket._buckets])
    max_batch_size = np.max([b._data.shape[0] for b in self._metabucket._buckets])
    print("max batch size: ", max_batch_size)
    if self.name == "Testset":
      return self.max_test_batch_size
    elif self.name == "Validset":
      return self.max_dev_batch_size
    return max_batch_size

  #=============================================================
  def __getitem__(self, key):
    return self._metabucket[key]
  def __len__(self):
    return len(self._metabucket)
