#!/usr/bin/env bash
#
# Usage: ./eval-conll05.sh [model_dir] (regular|dm|gold)

model_dir=$1
parse_config=$2

data_dir=$DATA_DIR/conll05st-release

valid_file=$data_dir/dev-set.gz.parse.sdeps.combined.bio
valid_props_file=$data_dir/conll2005-dev-gold-props.txt
valid_parse_file=$data_dir/dev-set.conll

# WSJ test
wsj_test_file=$data_dir/test.wsj.gz.parse.sdeps.combined.bio
wsj_test_props_file=$data_dir/conll2005-test-wsj-gold-props.txt
wsj_test_parse_file=$data_dir/test.wsj.conll

# Brown test
brown_test_file=$data_dir/test.brown.gz.parse.sdeps.combined.bio
brown_test_props_file=$data_dir/conll2005-test-brown-gold-props.txt
brown_test_parse_file=$data_dir/test.brown.conll

# D&M injected
dm_valid_file=lstm_predicted_parses_elmo/parse_preds-conll2005-lstm-dev24.tsv.replaced
dm_wsj_test_file=lstm_predicted_parses_elmo/parse_preds-conll2005-lstm-test.tsv.replaced
dm_brown_test_file=lstm_predicted_parses_elmo/parse_preds-conll2005-lstm-test-brown.tsv.replaced

if [[ "$parse_config" == "dm" || "$parse_config" == "gold" ]]; then
    gold_attn_at_train="False"
    inject_manual_attn="True"
    if [[ "$parse_config" == "dm" ]]; then
        echo "Doing D&M eval"
        valid_file=$dm_valid_file
        wsj_test_file=$dm_wsj_test_file
        brown_test_file=$dm_brown_test_file
    else
        echo "Doing Gold eval"
    fi
else
    echo "Doing regular eval"
    gold_attn_at_train="True"
    inject_manual_attn="True"
fi

# Dev only
#python $DOZAT_ROOT/network.py \
#    --load \
#    --test \
#    --load_dir $model_dir \
#    --save_dir tmp \
#    --config_file $model_dir/config.cfg \
#    --gold_attn_at_train $gold_attn_at_train \
#    --inject_manual_attn $inject_manual_attn \
#    --valid_file $valid_file \
#    --test_file $wsj_test_file \
#    --gold_test_props_file $wsj_test_props_file \
#    --gold_dev_parse_file $valid_parse_file \
#    --gold_test_parse_file $wsj_test_parse_file

# WSJ test
python $DOZAT_ROOT/network.py \
    --load \
    --test \
    --test_eval \
    --load_dir $model_dir \
    --config_file $model_dir/config.cfg \
    --gold_attn_at_train $gold_attn_at_train \
    --inject_manual_attn $inject_manual_attn \
    --valid_file $valid_file \
    --test_file $wsj_test_file \
    --gold_test_props_file $wsj_test_props_file \
    --gold_dev_parse_file $valid_parse_file \
    --gold_test_parse_file $wsj_test_parse_file

# Brown test
python $DOZAT_ROOT/network.py \
    --load \
    --test \
    --test_eval \
    --load_dir $model_dir \
    --config_file $model_dir/config.cfg \
    --gold_attn_at_train $gold_attn_at_train \
    --inject_manual_attn $inject_manual_attn \
    --valid_file $valid_file \
    --test_file $brown_test_file \
    --gold_test_props_file $brown_test_props_file \
    --gold_dev_parse_file $valid_parse_file \
    --gold_test_parse_file $brown_test_parse_file
