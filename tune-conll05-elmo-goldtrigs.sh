#!/bin/bash

debug="false"

timestamp=`date +%Y-%m-%d-%H-%M-%S`
ROOT_DIR=$DOZAT_ROOT

OUT_LOG=$ROOT_DIR/hyperparams/tune-$timestamp
if [[ "$debug" == "false" ]]; then
    mkdir -p $OUT_LOG
fi

echo "Writing to $OUT_LOG"

#num_gpus=100
num_gpus=4

lrs="0.04" # 0.06"
mus="0.9"
nus="0.98"
epsilons="1e-12"
warmup_steps="4000"
batch_sizes="5000"

trans_layers="4" # 3
cnn_dims="1024" # 768
num_heads="8" #4 8"
head_sizes="64"
relu_hidden_sizes="256"

parents_penalties="0.1"
#grandparents_penalties="0.0 0.1 1.0 0.01 10.0 0.0001"
parents_layers="parents:2 no"
#grandparents_layers="grandparents:2 grandparents:3 no"
children_layers="no" #children:1 children:2 no"
pos_layers="1"
embedding_sizes="100 512"
#embedding_files="data/glove/glove.6B.100d.txt data/elmo/elmo-conll05-embeddings.txt"

reps="1"

# 2*2 = 4

# array to hold all the commands we'll distribute
declare -a commands

i=1
for lr in ${lrs[@]}; do
    for mu in ${mus[@]}; do
        for nu in ${nus[@]}; do
            for epsilon in ${epsilons[@]}; do
                for warmup_steps in ${warmup_steps[@]}; do
                    for cnn_dim in ${cnn_dims[@]}; do
                        for trans_layer in ${trans_layers[@]}; do
                            for num_head in ${num_heads[@]}; do
                                for head_size in ${head_sizes[@]}; do
                                    for relu_hidden_size in ${relu_hidden_sizes[@]}; do
                                        for batch_size in ${batch_sizes[@]}; do
                                            for parents_penalty in ${parents_penalties[@]}; do
                                                for parents_layer in ${parents_layers[@]}; do
                                                    for children_layer in ${children_layers[@]}; do
                                                        for pos_layer in ${pos_layers[@]}; do
                                                            for embedding_size in ${embedding_sizes[@]}; do
                                                                for rep in `seq $reps`; do
                                                                    fname_append="$rep-$lr-$mu-$nu-$epsilon-$warmup_steps-$batch_size-$cnn_dim-$trans_layer-$num_head-$head_size-$relu_hidden_size-$parents_penalty-$parents_layer-$children_layer-$pos_layer-$embedding_size"
                                                                    multitask_layer=""
                                                                    orig_parents_layer=$parents_layer
                                                                    if [[ "$parents_layer" == "no" ]]; then
                                                                        parents_layer=""
                                                                    else
                                                                        multitask_layer=$parents_layer
                                                                    fi
                                                                    orig_children_layer=$children_layer
                                                                    if [[ "$children_layer" == "no" ]]; then
                                                                        children_layer=""
                                                                    else
                                                                        if [[ "$multitask_layer" != "" ]]; then
                                                                            multitask_layer="$multitask_layer;"
                                                                        fi
                                                                        multitask_layer="$multitask_layer$children_layer"
                                                                    fi

                                                                    embed_file="data/glove/glove.6B.100d.txt"
                                                                    if [[ "$embedding_size" == "512" ]]; then
                                                                        embed_file="data/elmo/elmo-conll05-embeddings.txt"
                                                                    fi

#                                                                    partition="titanx-long"
#                                                                    if [[ $i -le 10 ]]; then
#                                                                        partition="m40-long"
#                                                                    fi

#                                                                    train_pos="False"
#                                                                    if [[ "$orig_parents_layer" == "no" ]]; then
#                                                                        train_pos="True"
#                                                                    fi

                                                                    commands+=("srun --gres=gpu:1 --partition=titanx-long --mem=24G python network.py  \
                                                                    --config_file config/trans-conll05-bio-manualattn-goldtrigs-sdeps.cfg \
                                                                    --save_dir $OUT_LOG/scores-$fname_append \
                                                                    --save_every 500 \
                                                                    --train_iters 5000000 \
                                                                    --train_batch_size $batch_size \
                                                                    --test_batch_size $batch_size \
                                                                    --warmup_steps $warmup_steps \
                                                                    --learning_rate $lr \
                                                                    --cnn_dim $cnn_dim \
                                                                    --n_recur $trans_layer \
                                                                    --num_heads $num_head \
                                                                    --head_size $head_size \
                                                                    --relu_hidden_size $relu_hidden_size \
                                                                    --embed_size $embedding_size \
                                                                    --trig_embed_size 100 \
                                                                    --embed_file $embed_file \
                                                                    --mu $mu \
                                                                    --nu $nu \
                                                                    --epsilon $epsilon \
                                                                    --pos_layer $pos_layer \
                                                                    --train_pos True \
                                                                    --multitask_layers \"$multitask_layer\" \
                                                                    --multitask_penalties \"parents:$parents_penalty;children:$parents_penalty\"
                                                                    --eval_by_domain False \
                                                                    --eval_srl True \
                                                                    --save True \
                                                                    &> $OUT_LOG/train-$fname_append.log")
                                                                    i=$((i + 1))
                                                                    parents_layer=$orig_parents_layer
                                                                    children_layer=$orig_children_layer
                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                         done
                    done
                done
            done
        done
    done
done

# now distribute them to the gpus
num_jobs=${#commands[@]}
jobs_per_gpu=$((num_jobs / num_gpus))
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for (( gpuid=0; gpuid<num_gpus; gpuid++)); do
    for (( i=0; i<jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]}"
        comm=${comm/XX/$gpuid}
#        echo "Starting job $jobid on gpu $gpuid"
        echo ${comm}
        if [[ "$debug" == "false" ]]; then
            eval ${comm}
        fi
    done &
    j=$((j + 1))
done