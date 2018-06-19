#!/usr/bin/env bash
#
# Usage: ./eval-conll12-parse.sh [model_dir]

model_dir=$1

data_dir=$DATA_DIR/conll-2012-sdeps-filt

valid_file=$data_dir/conll2012-dev.txt.bio
valid_props_file=$data_dir/conll2012-dev-gold-props.txt
valid_parse_file=$data_dir/conll2012-dev-gold-parse.txt

# Test
test_file=$data_dir/conll2012-test.txt.bio
test_props_file=$data_dir/conll2012-test-wsj-gold-props.txt
test_parse_file=$data_dir/conll2012-test-wsj-gold-parse.txt

# Dev
python $DOZAT_ROOT/network.py \
    --load \
    --test \
    --eval_srl False \
    --load_dir $model_dir \
    --config_file $model_dir/config.cfg \
    --valid_file $valid_file \
    --test_file $wsj_test_file \
    --gold_test_props_file $wsj_test_props_file \
    --gold_dev_parse_file $valid_parse_file \
    --gold_test_parse_file $wsj_test_parse_file
cp $model_dir/parse_preds.tsv $model_dir/parse_preds-conll2012-lstm-dev.tsv

# Test
python $DOZAT_ROOT/network.py \
    --load \
    --test \
    --test_eval \
    --eval_srl False \
    --load_dir $model_dir \
    --config_file $model_dir/config.cfg \
    --valid_file $valid_file \
    --test_file $wsj_test_file \
    --gold_test_props_file $wsj_test_props_file \
    --gold_dev_parse_file $valid_parse_file \
    --gold_test_parse_file $wsj_test_parse_file
cp $model_dir/parse_preds.tsv $model_dir/parse_preds-conll2012-lstm-test.tsv
