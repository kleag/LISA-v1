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





import os
import sys
import time
import pickle as pkl

import numpy as np
import tensorflow as tf
import spacy
import tempfile

from lib import models
from lib.models.parsers.parser import Parser
from lib import optimizers
from lib.optimizers.radam_optimizer import RadamOptimizer
from lib import rnn_cells

from configurable import Configurable
from vocab import Vocab
from dataset import Dataset
import contextlib
from subprocess import check_output, CalledProcessError
import operator

xrange = range

dep_transcat_table = {
    "compound": "nn",
    "acl": "vmod",
    "oprd": "dep",
    "attr": "xcomp"
}

nlp = spacy.load('en')
reverse_index = nlp.tagger.vocab.morphology.tag_map
nlp.tagger.vocab.morphology.reverse_index.pop(next(iter(nlp.tagger.vocab.morphology.tag_map['_SP'])))
nlp.tagger.vocab.morphology.tag_map.pop('_SP')

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

#***************************************************************
class Network(Configurable):
  """"""

  #=============================================================
  def __init__(self, model, *args, **kwargs):
    """"""

    if args:
      if len(args) > 1:
        raise TypeError('Parser takes at most one argument')
    self._optimizer = None
    kwargs['name'] = kwargs.pop('name', model.__name__)
    super(Network, self).__init__(*args, **kwargs)
    if not os.path.isdir(self.save_dir):
      os.mkdir(self.save_dir)
    with open(os.path.join(self.save_dir, 'config.cfg'), 'w') as f:
      self._config.write(f)

    self._global_step = tf.Variable(0., trainable=False, name="global_step")
    self._global_epoch = tf.Variable(0., trainable=False, name="global_epoch")

    # todo what is this??
    # self._model = model(self._config, global_step=self.global_step)
    self._model = model(self._config)

    self._vocabs = []

    if self.conll:
      vocab_files = [(self.word_file, 1, 'Words', self.embed_size),
                     (self.tag_file, [3, 4], 'Tags', self.embed_size if self.add_pos_to_input else 0),
                     (self.rel_file, 7, 'Rels', 0)]
    elif self.conll2012:
      vocab_files = [(self.word_file, 3, 'Words', self.embed_size),
                     (self.tag_file, [5, 4], 'Tags', self.embed_size if self.add_pos_to_input else 0), # auto, gold
                     (self.rel_file, 7, 'Rels', 0),
                     (self.srl_file, list(range(14, 50)), 'SRLs', 0),
                     (self.predicates_file, [10, 4] if self.joint_pos_predicates else 10,
                        'Predicates', self.predicate_embed_size if self.add_predicates_to_input else 0),
                     (self.domain_file, 0, 'Domains', 0)]

    print("Loading vocabs")
    sys.stdout.flush()
    for i, (vocab_file, index, name, embed_size) in enumerate(vocab_files):
      vocab = Vocab(vocab_file, index, embed_size, self._config,
                    name=name,
                    cased=self.cased if not i else True,
                    use_pretrained=(not i))
      self._vocabs.append(vocab)

    print("Relations vocab: ")
    for l, i in sorted(list(self._vocabs[2].items()), key=operator.itemgetter(1)):
      print(f"{l}: {i}")
    print("Predicates vocab: ")
    for l, i in sorted(list(self._vocabs[4].items()), key=operator.itemgetter(1)):
      print(f"{l}: {i}")
    print("predicate_true_start_idx", self._vocabs[4].predicate_true_start_idx)

    print("Loading data")
    sys.stdout.flush()
    self._trainset = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset')
    self._validset = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset')
    self._testset = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset')

    self._ops = self._gen_ops()
    self._save_vars = [x for x in tf.global_variables() if 'Pretrained' not in x.name]
    self.history = {
      'train_loss': [],
      'train_accuracy': [],
      'valid_loss': [],
      'valid_accuracy': [],
      'test_acuracy': 0
    }
    return

  #=============================================================
  def train_minibatches(self):
    """Return a set of minibatches from the train set (shuffled) """

    return self._trainset.get_minibatches(self.train_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs)

  #=============================================================
  def valid_minibatches(self):
    """"""

    return self._validset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle=False)

  #=============================================================
  def test_minibatches(self):
    """"""

    return self._testset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle=False)

  def convert_bilou(self, indices):
    strings = [self._vocabs[3][i] for i in indices]
    converted = []
    started_types = []
    # print(strings)
    for i, s in enumerate(strings):
      label_parts = s.split('/')
      curr_len = len(label_parts)
      combined_str = ''
      Itypes = []
      Btypes = []
      for idx, label in enumerate(label_parts):
        bilou = label[0]
        label_type = label[2:]
        props_str = ''
        if bilou == 'I':
          Itypes.append(label_type)
          props_str = ''
        elif bilou == 'O':
          curr_len = 0
          props_str = ''
        elif bilou == 'U':
          # need to check whether last one was ended
          props_str = '(' + label_type + ('*)' if idx == len(label_parts) - 1 else "")
        elif bilou == 'B':
          # need to check whether last one was ended
          props_str = '(' + label_type
          started_types.append(label_type)
          Btypes.append(label_type)
        elif bilou == 'L':
          props_str = ')'
          started_types.pop()
          curr_len -= 1
        combined_str += props_str
      while len(started_types) > curr_len:
        converted[-1] += ')'
        started_types.pop()
      while len(started_types) < len(Itypes) + len(Btypes):
        combined_str = '(' + Itypes[-1] + combined_str
        started_types.append(Itypes[-1])
        Itypes.pop()
      if not combined_str:
        combined_str = '*'
      elif combined_str[0] == "(" and combined_str[-1] != ")":
        combined_str += '*'
      elif combined_str[-1] == ")" and combined_str[0] != "(":
        combined_str = '*' + combined_str
      converted.append(combined_str)
    while len(started_types) > 0:
      converted[-1] += ')'
      started_types.pop()
    return converted

  def parens_check(self, srl_preds_str):
    for srl_preds in srl_preds_str:
      parens_count = 0
      for pred in srl_preds:
        for c in pred:
          if c == '(':
            parens_count += 1
          if c == ')':
            parens_count -= 1
            if parens_count < 0:
              return False
      if parens_count != 0:
        return False
    return True

  def analyze(self, sess, text_file, viterbi=False, validate=False):
    """
    """

    tf.logging.log(tf.logging.INFO, f"analyze_text: parser initialized")  
    tokenized_file = ""
    tokenized_filename = ""
    tf.logging.log(tf.logging.INFO, f"analyze_text analyzing: {text_file}")
    temp = tempfile.NamedTemporaryFile(mode='w+t')
    print(f"analyze_text writing to temp file {temp.name}:")
    with open(text_file, 'r') as f:
      text = f.read()
      #print(text)
      doc = nlp(text)
      for sentence_id, sentence in enumerate(doc.sents):
        tokens = {}
        for token_id, token in enumerate(sentence):
          tokens[token] = token_id
        for token_id, token in enumerate(sentence):
          if len(token.text) > 0 and token.text != '\n' and not token.is_space:
            # 0:domain  1:sent_id 2:id  3:word+word_type
            # 4:gold_pos 5:auto_pos    6:parse_head  7:parse_label
            if token.dep_ == "ROOT":
              depid = 0
              dep = "root"
            else:
              depid = int(tokens[token.head]) + 1
              dep = dep_transcat_table[token.dep_] if token.dep_ in dep_transcat_table else token.dep_
            line = (f'conll05\t{sentence_id}\t{token_id}'
                    f'\t{token.text}\t{token.tag_}\t{token.tag_}'
                    f'\t{depid}\t{dep}'
                    f'\t_\t-\t-\t-\t-\tO')
            #print(f">>> {line}")
            temp.write(f"{line}\n")
        #print(f">>>")
        temp.write(f"\n")
      temp.flush()
               
    tokenized_file = temp
    tokenized_filename = temp.name

    # TODO Convert text_file to conll_file
    analyze_set = Dataset(tokenized_filename, self._vocabs, model, self._config, name='Analyzeset')
    def analyze_minibatches():
      """"""
               
      return analyze_set.get_minibatches(self.test_batch_size,
          self.model.input_idxs,
          self.model.target_idxs,
          shuffle=False)

    analyze_output = self._model(analyze_set, moving_params=self._optimizer)
    #analyze_output = self._model(analyze_set)

    ops = [analyze_output['probabilities'],
           analyze_output['n_cycles'],
           analyze_output['len_2_cycles'],
           analyze_output['srl_probs'],
           analyze_output['srl_preds'],
           analyze_output['srl_logits'],
           analyze_output['srl_correct'],
           analyze_output['srl_count'],
           analyze_output['srl_predicates'],
           analyze_output['srl_predicate_targets'],
           analyze_output['transition_params'],
           analyze_output['attn_weights'],
           analyze_output['attn_correct'],
           analyze_output['pos_correct'],
           analyze_output['pos_preds']]

    all_predictions = [[]]
    all_sents = [[]]
    bkt_idx = 0
    total_time = 0.
    roots_lt_total = 0.
    roots_gt_total = 0.
    cycles_2_total = 0.
    cycles_n_total = 0.
    not_tree_total = 0.
    srl_correct_total = 0.
    srl_count_total = 0.
    forward_total_time = 0.
    non_tree_preds_total = []
    attention_weights = {}
    attn_correct_counts = {}
    pos_correct_total = 0.
    n_tokens = 0.
    for batch_num, (feed_dict, sents) in enumerate(analyze_minibatches()):
      mb_inputs = feed_dict[analyze_set.inputs]
      mb_targets = feed_dict[analyze_set.targets]
      forward_start = time.time()
      probs, n_cycles, len_2_cycles, srl_probs, srl_preds, srl_logits, srl_correct, srl_count, srl_predicates, srl_predicate_targets, transition_params, attn_weights, attn_correct, pos_correct, pos_preds = sess.run(ops, feed_dict=feed_dict)
      forward_total_time += time.time() - forward_start
      preds, parse_time, roots_lt, roots_gt, cycles_2, cycles_n, non_trees, non_tree_preds, n_tokens_batch = self.model.validate(mb_inputs, mb_targets, probs, n_cycles, len_2_cycles, srl_preds, srl_logits, srl_predicates, srl_predicate_targets, pos_preds, transition_params if viterbi else None)
      n_tokens += n_tokens_batch
      for k, v in list(attn_weights.items()):
        attention_weights["b%d:layer%d" % (batch_num, k)] = v
      for k, v in list(attn_correct.items()):
        if k not in attn_correct_counts:
          attn_correct_counts[k] = 0.
        attn_correct_counts[k] += v
      total_time += parse_time
      roots_lt_total += roots_lt
      roots_gt_total += roots_gt
      cycles_2_total += cycles_2
      cycles_n_total += cycles_n
      not_tree_total += non_trees
      srl_correct_total += srl_correct
      srl_count_total += srl_count
      pos_correct_total += pos_correct
      non_tree_preds_total.extend(non_tree_preds)
      all_predictions[-1].extend(preds)
      all_sents[-1].extend(sents)
      if len(all_predictions[-1]) == len(analyze_set[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(analyze_set._metabucket):
          all_predictions.append([])
          all_sents.append([])

    data_indices = analyze_set._metabucket.data
    # all_predictions = [p for s in all_predictions for p in s]

    correct = {'UAS': 0., 'LAS': 0., 'parse_eval': '', 'F1': 0.}
    srl_acc = 0.0
    if self.eval_parse:
      # ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
      # FORM: Word form or punctuation symbol.
      # LEMMA: Lemma or stem of word form.
      # UPOSTAG: Universal part-of-speech tag.
      # XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
      # FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
      # HEAD: Head of the current word, which is either a value of ID or zero (0).
      # DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
      # DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
      # MISC: Any other annotation.

      # write predicted parse
      parse_pred_fname = os.path.join(self.save_dir, "parse_preds.tsv")
      with open(parse_pred_fname, 'w') as f:
        for p_idx, (bkt_idx, idx) in enumerate(data_indices):
          preds = all_predictions[p_idx] if self.one_example_per_predicate else all_predictions[bkt_idx][idx]
          words = all_sents[bkt_idx][idx]
          # sent[:, 6] = targets[tokens, 0] # 5 targets[0] = gold_tag
          # sent[:, 7] = parse_preds[tokens]  # 6 = pred parse head
          # sent[:, 8] = rel_preds[tokens]  # 7 = pred parse label
          # sent[:, 9] = targets[tokens, 1]  # 8 = gold parse head
          # sent[:, 10] = targets[tokens, 2]  # 9 = gold parse label
          sent_len = len(words)
          if self.eval_single_token_sents or sent_len > 1:
            for i, (word, pred) in enumerate(zip(words, preds)):
              #print(f"pred n°{i}: {word}, {pred}")
              head = pred[8] + 1
              tok_id = i + 1
              # assert self.tags[datum[6]] == self.tags[pred[7]]
              tup = (
                tok_id,  # id
                word,  # form
                self.tags[pred[7]],  # gold tag
                # self.tags[pred[11]] if self.joint_pos_predicates or self.train_pos else self.tags[pred[4]], # pred tag or auto tag
                str(head if head != tok_id else 0),  # pred head
                self.rels[pred[9]] # pred label
              )
              f.write('%s\t%s\t_\t%s\t_\t_\t%s\t%s\n' % tup)
            f.write('\n')

    if self.eval_srl:
      # load the real gold preds file
      srl_gold_fname = self.gold_dev_props_file if validate else self.gold_test_props_file

      # save SRL output
      srl_preds_fname = os.path.join(self.save_dir, 'srl_preds.tsv')
      # print("writing srl preds file: %s" % srl_preds_fname)
      with open(srl_preds_fname, 'w') as f:
        for p_idx, (bkt_idx, idx) in enumerate(data_indices):
          # for each word, if predicate print word, otherwise -
          # then all the SRL labels
          preds = all_predictions[p_idx] if self.one_example_per_predicate else all_predictions[bkt_idx][idx]
          words = all_sents[bkt_idx][idx]
          # if len(preds.shape) < 2:
          #   preds = np.reshape(preds, [1, preds.shape[0]])
          # print("preds", preds)
          num_gold_srls = preds[0, 13]
          num_pred_srls = preds[0, 14]
          srl_preds = preds[:, 15 + num_gold_srls + num_pred_srls:]
          if self.one_example_per_predicate:
            # srl_preds = preds[:, 14 + num_gold_srls + num_pred_srls:]
            predicate_indices = np.where(preds[:, 4] == 1)[0]
            # print("predicate indices", predicate_indices)
          else:
            predicate_indices = preds[0, 15:15+num_pred_srls]
          # print("predicate indices", predicate_indices)
          srl_preds_str = list(map(list, list(zip(*[self.convert_bilou(j) for j in np.transpose(srl_preds)]))))
          for i, word in enumerate(words):
            pred = srl_preds_str[i] if srl_preds_str else []
            word_str = word if i in predicate_indices else '-'
            fields = (word_str,) + tuple(pred)
            owpl_str = '\t'.join(fields)
            f.write(owpl_str + "\n")
          if not self.parens_check(np.transpose(srl_preds_str)):
            print(np.transpose(srl_preds_str))
            print([self._vocabs[3][i] for i in np.transpose(srl_preds)])
          f.write('\n')

    return correct


  #=============================================================
  def _gen_ops(self):
    """Generate a graph for each set (train, valid and test) and retrurn a dict 
    with all tensors and the output values necessary to compute the result
    """

    self._optimizer = RadamOptimizer(self._config, global_step=self._global_step)
    train_output = self._model(self._trainset)

    lr = self._optimizer.learning_rate

    train_op = self._optimizer.minimize(train_output['loss'])

    # These have to happen after optimizer.minimize is called
    valid_output = self._model(self._validset, moving_params=self._optimizer)
    test_output = self._model(self._testset, moving_params=self._optimizer)



    ops = {}
    ops['train_op'] = [train_op] + [train_output['loss'],
                       train_output['n_correct'],
                       train_output['n_tokens']]
    ops['train_op_svd'] = [train_op] + [train_output['loss'],
                           train_output['n_correct'],
                           train_output['n_tokens'],
                           train_output['roots_loss'],
                           train_output['2cycle_loss'],
                           train_output['svd_loss'],
                           train_output['log_loss'],
                           train_output['rel_loss']]
    ops['train_op_srl'] = [train_op] + [train_output['loss'],
                           train_output['n_correct'],
                           train_output['n_tokens'],
                           train_output['roots_loss'],
                           train_output['2cycle_loss'],
                           train_output['svd_loss'],
                           train_output['log_loss'],
                           train_output['rel_loss'],
                           train_output['srl_loss'],
                           train_output['srl_correct'],
                           train_output['srl_count'],
                           train_output['predicate_loss'],
                           train_output['predicate_count'],
                           train_output['predicate_correct'],
                           train_output['pos_loss'],
                           train_output['pos_correct'],
                           train_output['multitask_losses'],
                           lr,
                           train_output['sample_prob']]
    ops['valid_op'] = [valid_output['loss'],
                       valid_output['n_correct'],
                       valid_output['n_tokens'],
                       valid_output['predictions']]
    ops['valid_test_op'] = [valid_output['probabilities'],
                      valid_output['n_cycles'],
                      valid_output['len_2_cycles'],
                      valid_output['srl_probs'],
                      valid_output['srl_preds'],
                      valid_output['srl_logits'],
                      valid_output['srl_correct'],
                      valid_output['srl_count'],
                      valid_output['srl_predicates'],
                      valid_output['srl_predicate_targets'],
                      valid_output['transition_params'],
                      valid_output['attn_weights'],
                      valid_output['attn_correct'],
                      valid_output['pos_correct'],
                      valid_output['pos_preds']]
    ops['test_op'] = [
                      test_output['probabilities'],
                      test_output['n_cycles'],
                      test_output['len_2_cycles'],
                      test_output['srl_probs'],
                      test_output['srl_preds'],
                      test_output['srl_logits'],
                      test_output['srl_correct'],
                      test_output['srl_count'],
                      test_output['srl_predicates'],
                      test_output['srl_predicate_targets'],
                      test_output['transition_params'],
                      test_output['attn_weights'],
                      test_output['attn_correct'],
                      test_output['pos_correct'],
                      test_output['pos_preds'],
                      ]
    # ops['optimizer'] = optimizer

    return ops

  #=============================================================
  # @property
  # def global_step(self):
  #   return self._global_step
  @property
  def global_epoch(self):
    return self._global_epoch
  @property
  def model(self):
    return self._model
  @property
  def words(self):
    return self._vocabs[0]
  @property
  def tags(self):
    return self._vocabs[1]
  @property
  def rels(self):
    return self._vocabs[2]
  @property
  def ops(self):
    return self._ops
  @property
  def save_vars(self):
    return self._save_vars

#***************************************************************
if __name__ == '__main__':
  """"""

  import argparse

  argparser = argparse.ArgumentParser()
  argparser.add_argument('--analyze')
  argparser.add_argument('--test', action='store_true')
  argparser.add_argument('--load', action='store_true')
  argparser.add_argument('--model', default='Parser')
  argparser.add_argument('--matrix', action='store_true')
  argparser.add_argument('--profile', action='store_true')
  argparser.add_argument('--test_eval', action='store_true')

  args, extra_args = argparser.parse_known_args()
  cargs = {k: v for (k, v) in list(vars(Configurable.argparser.parse_args(extra_args)).items()) if v is not None}

  print('*** '+args.model+' ***')
  #model = getattr(models, args.model)
  model = Parser

  profile = args.profile

  # if 'save_dir' in cargs and os.path.isdir(cargs['save_dir']) and not (args.test or args.matrix or args.load):
  #   raw_input('Save directory already exists. Press <Enter> to overwrite or <Ctrl-C> to exit.')
  # if (args.test or args.load or args.matrix) and 'save_dir' in cargs:
  #   cargs['config_file'] = os.path.join(cargs['save_dir'], 'config.cfg')
  network = Network(model, **cargs)
      
  os.system('echo Model: %s > %s/MODEL' % (network.model.__class__.__name__, network.save_dir))

  # print variable names (but not the optimizer ones)
  print([v.name for v in network.save_vars if 'Optimizer' not in v.name and 'layer_norm' not in v.name])

  config_proto = tf.ConfigProto()
  config_proto.gpu_options.per_process_gpu_memory_fraction = network.per_process_gpu_memory_fraction

  # Create options to profile the time and memory information.
  if profile:
    builder = tf.profiler.ProfileOptionBuilder
    opts = builder(builder.time_and_memory()).order_by('micros').build()
  # Create a profiling context, set constructor argument `trace_steps`,
  # `dump_steps` to empty for explicit control.
  with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                        trace_steps=[],
                                        dump_steps=[]) if profile else dummy_context_mgr() as pctx:
    with tf.Session(config=config_proto) as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver(var_list=network.save_vars, save_relative_paths=True)
      print("Loading model: ", network.load_dir)
      print(network.name.lower())
      saver.restore(sess, tf.train.latest_checkpoint(network.load_dir, latest_filename=network.name.lower()))

      text_file = args.analyze
      print(f"Analyzing text file: {text_file}")
      network.analyze(sess, text_file, False, validate=True)

