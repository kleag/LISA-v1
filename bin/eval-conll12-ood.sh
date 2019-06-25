#!/usr/bin/env bash
#
# Usage: ./eval-conll12.sh [model_dir] (regular|dm|gold)

model_dir=$1
parse_config=$2

domains="bc bn mz nw pt tc wb"

model_name=${model_dir##*/}
eval_save_dir="eval-$model_name"
#mkdir $eval_save_dir

data_dir=$DATA_DIR/conll-2012-sdeps-filt-new

valid_file_base=$data_dir/conll2012-dev.txt.bio
valid_props_file_base=$data_dir/conll2012-dev-gold-props.txt
valid_parse_file_base=$data_dir/conll2012-dev-gold-parse.txt

# Test
test_file_base=$data_dir/conll2012-test.txt.bio
test_props_file_base=$data_dir/conll2012-test-gold-props.txt
test_parse_file_base=$data_dir/conll2012-test-gold-parse.txt

# D&M injected
#dm_valid_file_base=lstm_predicted_parses_elmo_new/parse_preds-conll2012-lstm-dev.tsv.replaced
#dm_test_file_base=lstm_predicted_parses_elmo_new/parse_preds-conll2012-lstm-test.tsv.replaced

if [[ "$parse_config" == "dm" || "$parse_config" == "gold" ]]; then
    gold_attn_at_train="False"
    inject_manual_attn="True"
    if [[ "$parse_config" == "dm" ]]; then
        echo "Doing D&M eval"
        valid_file=$dm_valid_file
        test_file=$dm_test_file
    else
        echo "Doing Gold eval"
    fi
else
    echo "Doing regular eval"
    gold_attn_at_train="True"
    inject_manual_attn="True"
fi



# overall eval
this_save_dir="$eval_save_dir/overall"
mkdir -p $this_save_dir
python $DOZAT_ROOT/network.py \
    --load \
    --test \
    --test_eval \
    --load_dir $model_dir \
    --save_dir $this_save_dir \
    --config_file $model_dir/config.cfg \
    --gold_attn_at_train $gold_attn_at_train \
    --inject_manual_attn $inject_manual_attn \
    --valid_file $valid_file \
    --test_file $test_file \
    --gold_test_props_file $test_props_file_base \
    --gold_dev_parse_file $valid_parse_file_base \
    --gold_test_parse_file $test_parse_file_base \
    --eval_by_domain True

for domain in domains; do
    this_save_dir="$eval_save_dir/$domain"
    mkdir -p $this_save_dir

    valid_file="$domain\_$valid_file_base"
    test_file="$domain\_$test_file_base"
    test_props_file="$domain\_$test_props_file_base"
    valid_parse_file="$domain\_$valid_parse_file_base"
    test_parse_file="$domain\_$test_parse_file_base"

    python $DOZAT_ROOT/network.py \
        --load \
        --test \
        --test_eval \
        --load_dir $model_dir \
        --save_dir $this_save_dir \
        --config_file $model_dir/config.cfg \
        --gold_attn_at_train $gold_attn_at_train \
        --inject_manual_attn $inject_manual_attn \
        --valid_file $valid_file \
        --test_file $test_file \
        --gold_test_props_file $test_props_file \
        --gold_dev_parse_file $valid_parse_file \
        --gold_test_parse_file $test_parse_file \
        --eval_by_domain True
done