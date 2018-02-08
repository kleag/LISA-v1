from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import operator
import os


save_dir = "errors_by_pos"
fnames = ["parents0_wsj22-dev.sdep.spos.conllu", "parents1_wsj22-dev.sdep.spos.conllu", "parents2_wsj22-dev.sdep.spos.conllu", "parents3_wsj22-dev.sdep.spos.conllu"]
pos_fname = "tags.txt"
rels_fname = "rels.txt"

PUNCT = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])

with open(rels_fname, 'r') as f:
  rels_str_int_map = {s: i for i, s in enumerate(map(lambda l: l.strip().split()[0], f.readlines()))}
  rels_int_str_map = {i: s for s, i in rels_str_int_map.iteritems()}

with open(pos_fname, 'r') as f:
  tags_str_int_map = {s: i+1 for i, s in enumerate(map(lambda l: l.strip().split()[0], f.readlines()))}
  tags_str_int_map['ROOT'] = 0
  tags_int_str_map = {i: s for s, i in tags_str_int_map.iteritems()}

num_tags = len(tags_str_int_map)
num_rels = len(rels_str_int_map)

last_pos_errors = None
last_pos_corrects = None
for fname in fnames:
  pos_errors = np.zeros((num_tags, num_tags), dtype=np.float32)
  rel_errors = np.zeros(num_rels, dtype=np.float32)
  pos_corrects = np.zeros((num_tags, num_tags), dtype=np.float32)
  rel_corrects = np.zeros(num_rels, dtype=np.float32)
  rel_counts = np.zeros(num_rels, dtype=np.float32)
  pos_counts = np.ones((num_tags, num_tags), dtype=np.float32)
  name = fname[:8]
  num_tokens = 0
  num_non_punct_tokens = 0
  pos_correct = 0
  rel_correct = 0
  edge_correct = 0
  with open(fname, 'r') as f:
    buf = []
    for line in f:
      split_line = line.strip().split()
      # 1       Influential     _       JJ      JJ      _       2       amod    2       amod
      if split_line:
        buf.append(split_line)
      else:
        for token in buf:
          tok_idx, word1, word2, auto_pos, gold_pos, _, gold_head, gold_label, pred_head, pred_label = token
          num_tokens += 1
          if auto_pos == gold_pos:
            pos_correct += 1
          if gold_pos not in PUNCT:
            num_non_punct_tokens += 1
            head_idx = int(gold_head) - 1
            head_tag = tags_str_int_map[buf[head_idx][4]] if head_idx > -1 else 0
            rel_counts[rels_str_int_map[gold_label]] += 1
            pos_counts[tags_str_int_map[gold_pos], head_tag] += 1
            if gold_head == pred_head:
              edge_correct += 1
              pos_corrects[tags_str_int_map[gold_pos], head_tag] += 1
              rel_corrects[rels_str_int_map[gold_label]] += 1
            else:
              head_idx = int(gold_head) - 1
              head_tag = tags_str_int_map[buf[head_idx][4]] if head_idx > -1 else 0
              pos_errors[tags_str_int_map[gold_pos], head_tag] += 1
              rel_errors[rels_str_int_map[gold_label]] += 1
        buf = []

  print("Deprel counts:")
  for i, count in enumerate(rel_counts):
    print("%s\t%d" % (rels_int_str_map[i], count))
  print()

  print("Errors by deprel:")
  # for i, errors in reversed(sorted(enumerate(rel_errors), key=operator.itemgetter(1))):
  for i, errors in enumerate(rel_errors):
    print("%s\t%2.2f" % (rels_int_str_map[i], 100*errors/rel_counts[i] if rel_counts[i] > 0 else 0))
  print()

  print("Corrects by deprel:")
  for i, corrects in enumerate(rel_corrects):
    print("%s\t%2.2f" % (rels_int_str_map[i], 100*corrects/rel_counts[i] if rel_counts[i] > 0 else 0))

  fig1, ax1 = plt.subplots()
  ax1.imshow(np.divide(pos_errors, pos_counts), cmap=plt.cm.viridis, interpolation=None)
  ax1.set_title("Errors")
  ax1.set_xticks(range(num_tags))
  ax1.set_yticks(range(num_tags))
  ax1.set_xticklabels([tags_int_str_map[i] for i in range(num_tags)], fontsize=8, rotation=90)
  ax1.set_yticklabels([tags_int_str_map[i] for i in range(num_tags)], fontsize=8)
  ax1.set_xlabel('Head gold POS tag')
  ax1.set_ylabel('Dep gold POS tag')
  fig1.tight_layout()
  fig1.savefig(os.path.join(save_dir, name + "_errors_by_pos" + ".pdf"))

  fig2, ax2 = plt.subplots()
  ax2.imshow(np.divide(pos_corrects, pos_counts), cmap=plt.cm.viridis, interpolation=None)
  ax2.set_title("Corrects")
  ax2.set_xticks(range(num_tags))
  ax2.set_yticks(range(num_tags))
  ax2.set_xticklabels([tags_int_str_map[i] for i in range(num_tags)], fontsize=8, rotation=90)
  ax2.set_yticklabels([tags_int_str_map[i] for i in range(num_tags)], fontsize=8)
  ax2.set_xlabel('Head gold POS tag')
  ax2.set_ylabel('Dep gold POS tag')
  fig2.tight_layout()
  fig2.savefig(os.path.join(save_dir, name + "_corrects_by_pos" + ".pdf"))

  if last_pos_errors is not None:
    fig3, ax3 = plt.subplots()
    diff = (pos_errors - last_pos_errors) / (last_pos_errors + 1)
    print(diff)
    im = ax3.imshow(np.divide(diff, pos_counts), cmap=plt.cm.Spectral, interpolation=None)
    ax3.set_title("Errors diff")
    ax3.set_xticks(range(num_tags))
    ax3.set_yticks(range(num_tags))
    ax3.set_xticklabels([tags_int_str_map[i] for i in range(num_tags)], fontsize=8, rotation=90)
    ax3.set_yticklabels([tags_int_str_map[i] for i in range(num_tags)], fontsize=8)
    ax3.set_xlabel('Head gold POS tag')
    ax3.set_ylabel('Dep gold POS tag')
    # handles, labels = ax3.get_legend_handles_labels()
    fig3.colorbar(im)
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, name + "_errors_by_pos_diff" + ".pdf"))

  if last_pos_corrects is not None:
    fig4, ax4 = plt.subplots()
    diff = (pos_corrects - last_pos_corrects) / (last_pos_corrects + 1)
    im = ax4.imshow(np.divide(diff, pos_counts), cmap=plt.cm.Spectral, interpolation=None)
    ax4.set_title("Corrects diff")
    ax4.set_xticks(range(num_tags))
    ax4.set_yticks(range(num_tags))
    ax4.set_xticklabels([tags_int_str_map[i] for i in range(num_tags)], fontsize=8, rotation=90)
    ax4.set_yticklabels([tags_int_str_map[i] for i in range(num_tags)], fontsize=8)
    ax4.set_xlabel('Head gold POS tag')
    ax4.set_ylabel('Dep gold POS tag')
    fig4.colorbar(im)
    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, name + "_corrects_by_pos_diff" + ".pdf"))

  last_pos_corrects = pos_corrects
  last_pos_errors = pos_errors

# plt.show()

