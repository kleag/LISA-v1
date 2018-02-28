#!/usr/bin/env bash

orig_data_file=$1
pred_parse_file=$2

# pred parse file format:
# 1       Avions  _       NNP     _       _       7       compound
#
# orig file format:
# nw/wsj/24/wsj_2437      0       0       Avions  NNP     NNS     7       compound        _       -       -       -       -       B-ORG   B-ARG0  O       O       (1

paste <(awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6}'  $orig_data_file) \
      <(awk '{print $7"\t"$8}' $pred_parse_file ) \
      <(awk '{ s = ""; for (i = 9; i <= NF; i++) s = s $i "\t"; print s }' $orig_data_file)

