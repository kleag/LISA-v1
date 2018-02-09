#!/usr/bin/env bash

orig_data_file=$1
pred_pos_file=$2

# pred parse file format:
# nw      semiconductors  NNS     IN      NNS     -       I-ARG1  I-ARG1  B-C-ARG1        *       *)      (C-ARG1*)
#
# orig file format:
# nw/wsj/24/wsj_2437      0       0       Avions  NNP     NNS     7       compound        _       -       -       -       -       B-ORG   B-ARG0  O       O       (1

paste <(awk '{print $1"\t"$2"\t"$3"\t"$4"}'  $orig_data_file) \
      <(awk '{print $5}' $pred_pos_file ) \
      <(awk '{ s = ""; for (i = 6; i <= NF; i++) s = s $i "\t"; print s }' $orig_data_file)

