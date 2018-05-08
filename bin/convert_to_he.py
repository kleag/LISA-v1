
import numpy as np

fname = "srl_sanity.tsv"
with open(fname, 'r') as f:
  buff = []
  num_predicates = 0
  for line in f:
    line = line.strip()
    if line:
      split_line = line.split()
      srl_preds = split_line[7:]
      if srl_preds:
        # bio_preds = srl_preds[:(len(srl_preds)/2) + 1]
        buff.append(srl_preds)
      if split_line[6] != '-':
        num_predicates += 1
    else:
      buff = np.transpose(np.array(buff))
      for pred in buff[:num_predicates]:
        print(' '.join(pred))
      buff = []
      num_predicates = 0

  # last one
  buff = np.transpose(np.array(buff))
  for pred in buff[:num_predicates]:
    print(' '.join(pred))