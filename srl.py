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

import numpy as np
import tensorflow as tf
import spacy
import tempfile

from lib.models.parsers.parser import Parser
from lib.optimizers.radam_optimizer import RadamOptimizer

from configurable import Configurable
from vocab import Vocab
from dataset import Dataset

dep_transcat_table = {
    "compound": "nn",
    "acl": "vmod",
    "oprd": "dep",
    "attr": "xcomp"
}

nlp = spacy.load('en')

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

    #print("Loading vocabs")
    for i, (vocab_file, index, name, embed_size) in enumerate(vocab_files):
      vocab = Vocab(vocab_file, index, embed_size, self._config,
                    name=name,
                    cased=self.cased if not i else True,
                    use_pretrained=(not i))
      self._vocabs.append(vocab)

    #print("Loading data")
    self._trainset = Dataset(None, self._vocabs, model, self._config, name='Trainset')

    self._ops = self._gen_ops()
    self._save_vars = [x for x in tf.compat.v1.global_variables() if 'Pretrained' not in x.name]
    return

  def analyze(self, sess, text):
    """
    """

    tf.compat.v1.logging.log(tf.compat.v1.logging.INFO,
                             f"analyze analyzing: {text}")
    result = ""
    temp = tempfile.NamedTemporaryFile(mode='w+t')
    #print(f"analyze writing to temp file {temp.name}:")
    #print(text)
    doc = nlp(text)
    #sentence_id = 0
    tokens_localization = {}
    for sentence_id, sentence in enumerate(doc.sents):
      tokens = {}
      for token_id, token in enumerate(sentence):
        tokens[token] = token_id
      real_token_id = 0
      for token_id, token in enumerate(sentence):
        if len(token.text) > 0 and token.text != '\n' and not token.is_space:
          # associate token pos and len to the identification of the token
          tokens_localization[(sentence_id, real_token_id)] = (token.idx, token.__len__())
          real_token_id += 1
          # 0:domain  1:sent_id 2:id  3:word+word_type
          # 4:gold_pos 5:auto_pos    6:parse_head  7:parse_label
          if token.dep_ == "ROOT":
            depid = 0
            dep = "root"
          else:
            depid = int(tokens[token.head]) + 1
            dep = dep_transcat_table[token.dep_] if token.dep_ in dep_transcat_table else token.dep_
          line = (f'conll05\t{sentence_id}\t{real_token_id}'
                  f'\t{token.text}\t{token.tag_}\t{token.tag_}'
                  f'\t{depid}\t{dep}'
                  f'\t_\t-\t-\t-\t-\tO')
          #print(f">>> {line}")
          temp.write(f"{line}\n")
      #print(f">>>")
      temp.write(f"\n")
    temp.flush()
    #print(f"tokens_localization: {tokens_localization}")
    tokenized_file = temp
    tokenized_filename = temp.name

    # TODO Convert text_file to conll_file
    analyze_set = Dataset(tokenized_filename, self._vocabs, Parser,
                        self._config, name='Analyzeset')
    def analyze_minibatches():
      """"""

      return analyze_set.get_minibatches(self.test_batch_size,
        self.model.input_idxs,
        self.model.target_idxs,
        shuffle=False)

    analyze_output = self._model(analyze_set, moving_params=self._optimizer)

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
    for batch_num, (feed_dict, sents) in enumerate(analyze_minibatches()):
      mb_inputs = feed_dict[analyze_set.inputs]
      mb_targets = feed_dict[analyze_set.targets]
      (probs, n_cycles, len_2_cycles, srl_probs, srl_preds, srl_logits,
       srl_correct, srl_count, srl_predicates, srl_predicate_targets,
       transition_params, attn_weights, attn_correct, pos_correct,
       pos_preds) = sess.run(ops, feed_dict=feed_dict)
      preds, _, _, _, _, _, _, _, _ = self.model.validate(mb_inputs,
          mb_targets, probs, n_cycles, len_2_cycles, srl_preds, srl_logits,
          srl_predicates, srl_predicate_targets, pos_preds, None)
      all_predictions[-1].extend(preds)
      all_sents[-1].extend(sents)
      if len(all_predictions[-1]) == len(analyze_set[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(analyze_set._metabucket):
          all_predictions.append([])
          all_sents.append([])

    data_indices = analyze_set._metabucket.data

    # ID: Word index, integer starting at 1 for each new sentence; may be a
    #     range for multiword tokens; may be a decimal number for empty nodes.
    # FORM: Word form or punctuation symbol.
    # LEMMA: Lemma or stem of word form.
    # UPOSTAG: Universal part-of-speech tag.
    # XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
    # FEATS: List of morphological features from the universal feature inventory
    #        or from a defined language-specific extension; underscore if not
    #        available.
    # HEAD: Head of the current word, which is either a value of ID or zero (0).
    # DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a
    #         defined language-specific subtype of one.
    # DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
    # MISC: Any other annotation.
    sentence_id = 0
    # write predicted parse
    for p_idx, (bkt_idx, idx) in enumerate(data_indices):
      preds = all_predictions[p_idx] if self.one_example_per_predicate else all_predictions[bkt_idx][idx]
      sentence_id += 1
      words = all_sents[bkt_idx][idx]
      num_gold_srls = preds[0, 13]
      num_pred_srls = preds[0, 14]
      srl_preds = preds[:, 15 + num_gold_srls + num_pred_srls:]
      if self.one_example_per_predicate:
        predicate_indices = np.where(preds[:, 4] == 1)[0]
      else:
        predicate_indices = preds[0, 15:15+num_pred_srls]
      # print("predicate indices", predicate_indices)
      srl_preds_str = list(map(list,
                             list(zip(*[[self._vocabs[3][i]
                                         for i in j]
                             for j in np.transpose(srl_preds)]))))
      # sent[:, 6] = targets[tokens, 0] # 5 targets[0] = gold_tag
      # sent[:, 7] = parse_preds[tokens]  # 6 = pred parse head
      # sent[:, 8] = rel_preds[tokens]  # 7 = pred parse label
      # sent[:, 9] = targets[tokens, 1]  # 8 = gold parse head
      # sent[:, 10] = targets[tokens, 2]  # 9 = gold parse label
      sent_len = len(words)
      for i, (word, pred) in enumerate(zip(words, preds)):
        #print(f"pred n°{i}: {word}, {pred}")
        head = pred[8] + 1
        tok_id = i + 1
        # assert self.tags[datum[6]] == self.tags[pred[7]]
        position, length = tokens_localization[(sentence_id-1, tok_id-1)]
        tup = (
              str(sentence_id), # sent id
              str(tok_id),  # tok id
              str(position),
              str(length),
              word,  # form
              "_",
              self.tags[pred[7]],  # gold tag
              # self.tags[pred[11]] if self.joint_pos_predicates or self.train_pos else self.tags[pred[4]], # pred tag or auto tag
              "_",
              "_",
              str(head if head != tok_id else 0),  # pred head
              self.rels[pred[9]] # pred label
        )
        srl_pred = srl_preds_str[i] if srl_preds_str else []
        word_str = word if i in predicate_indices else '-'
        fields = (word_str,) + tuple(srl_pred)
        tup += fields
        #print(f"Network.analyze tup: {tup}")
        result += "\t".join(tup) + "\n"
      result += "\n"
    return result

  #=============================================================
  def _gen_ops(self):
    """Generate a graph for train set and return a dict with all tensors and
    the output values necessary to compute the result
    """

    self._optimizer = RadamOptimizer(self._config, global_step=self._global_step)
    train_output = self._model(self._trainset)

    train_op = self._optimizer.minimize(train_output['loss'])

    ops = {}
    ops['train_op'] = [train_op] + [train_output['loss'],
                       train_output['n_correct'],
                       train_output['n_tokens']]
    return ops

  ##=============================================================
  @property
  def model(self):
    return self._model
  @property
  def tags(self):
    return self._vocabs[1]
  @property
  def rels(self):
    return self._vocabs[2]
  @property
  def save_vars(self):
    return self._save_vars

class Analyzer(Configurable):
  """"""

  #=============================================================
  def __init__(self, cargs):
    """"""

    self._network = Network(Parser, **cargs)

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.per_process_gpu_memory_fraction = self._network.per_process_gpu_memory_fraction

    self._sess = tf.compat.v1.Session(config=config_proto)
    self._sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(var_list=self._network.save_vars,
                           save_relative_paths=True)
    print("Loading model: ", self._network.load_dir)
    saver.restore(self._sess,
                  tf.train.latest_checkpoint(
                      self._network.load_dir,
                      latest_filename=self._network.name.lower()))


  def analyze(self, text):
      return self._network.analyze(self._sess, text)


#***************************************************************
if __name__ == '__main__':
  """"""

  import argparse

  argparser = argparse.ArgumentParser()

  _, extra_args = argparser.parse_known_args()
  cargs = {k: v for (k, v) in
           list(vars(Configurable.argparser.parse_args(extra_args)).items())
           if v is not None}
  analyzer = Analyzer(cargs)
  for text_file in cargs['files']:
    #print(f"Analyzing text file: {text_file}")
    with open(text_file, 'r') as f:
      text = f.read()
      result = analyzer.analyze(text)
      with open(text_file+'.srl.conll', 'w') as output_file:
        print(result, file=output_file)
