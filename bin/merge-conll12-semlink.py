from __future__ import print_function


conll12_fname = "/home/strubell/research/data/conll-2012-new/conll2012-train.txt"
semlink_fname = "/home/strubell/research/data/semlink-1.2.2c/semlink-wsj.txt"

remove_list = ['rel', 'LINK-SLC', 'LINK-PSV', 'LINK-PRO']

semlink_map = {}
arg_mapping_counts = {}
arg_mappings = {}
proposition_count = 0
with open(semlink_fname, 'r') as semlink_file:
  for line in semlink_file:
    line = line.strip()
    if line:
      proposition_count += 1
      split_line = line.split()

      # key is doc name without ending + sentence number
      key = (split_line[0].split('.')[0], split_line[1])

      # value is predicate + args
      args = split_line[10:]
      # take just the verbnet senses
      stripped_args = map(lambda a: '-'.join(a.split('*')[-1].split('-')[1:]).split(';')[0].replace('-DSP', ''), args)
      stripped_removed_args = [a for a in stripped_args if a not in remove_list]

      # update mapping counts
      for arg in stripped_removed_args:
        if arg not in arg_mapping_counts:
          arg_mapping_counts[arg] = 0
        arg_mapping_counts[arg] += 1
        if '=' in arg:
          pb_arg, vn_arg = arg.split('=')
          if pb_arg not in arg_mappings:
            arg_mappings[pb_arg] = []
          if vn_arg not in arg_mappings[pb_arg]:
            arg_mappings[pb_arg].append(vn_arg)

      value = (split_line[7].split('.')[0], ' '.join(stripped_removed_args))
      if key not in semlink_map:
        semlink_map[key] = []
      semlink_map[key].append(value)

print("Loaded %d semlink propositions" % proposition_count)
print(arg_mapping_counts)
print(arg_mappings)

with open(conll12_fname, 'r') as conll12_file:
  # want to scan conll12 file until we find a sentence that is in semlink,
  # then process that sentence
  for line in conll12_file:
    line = line.strip()
    if line:
      split_line = line.split()
