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
num_gpus=2

lrs="0.04" # 0.06"
mus="0.9"
nus="0.98"
epsilons="1e-12"
warmup_steps="8000"
batch_sizes="5000"

trans_layers="10" # "10 8 6" # 3
num_heads="8" #4 8"
head_sizes="25"
relu_hidden_sizes="800"

parents_penalties="1.0"
rels_penalties="0.1"
#grandparents_penalties="0.0 0.1 1.0 0.01 10.0 0.0001"
parents_layers="parents:4"
#grandparents_layers="grandparents:2 grandparents:3 no"
predicate_layers="1"
scheduled_sampling="constant=1.0" # constant=0.0 sigmoid=64000 sigmoid=32000"
use_full_parse="True"
one_example_per_predicates="True"


reps="2"

# 2*2*2 = 8

# array to hold all the commands we'll distribute
declare -a commands

i=1
for lr in ${lrs[@]}; do
    for mu in ${mus[@]}; do
        for nu in ${nus[@]}; do
            for epsilon in ${epsilons[@]}; do
                for warmup_steps in ${warmup_steps[@]}; do
                    for trans_layer in ${trans_layers[@]}; do
                        for num_head in ${num_heads[@]}; do
                            for head_size in ${head_sizes[@]}; do
                                for relu_hidden_size in ${relu_hidden_sizes[@]}; do
                                    for batch_size in ${batch_sizes[@]}; do
                                        for parents_penalty in ${parents_penalties[@]}; do
                                            for rel_penalty in ${rels_penalties[@]}; do
                                                for parents_layer in ${parents_layers[@]}; do
                                                    for predicate_layer in ${predicate_layers[@]}; do
                                                        for full_parse in ${use_full_parse[@]}; do
                                                            for ss in ${scheduled_sampling[@]}; do
                                                                for one_example_per_predicate in ${one_example_per_predicates[@]}; do
                                                                    for rep in `seq $reps`; do
    #                                                                    if [[ "$cnn_layer" != "2" || "$trans_layer" != "10" ]]; then
                                                                        fname_append="$rep-$lr-$mu-$nu-$epsilon-$warmup_steps-$batch_size-$trans_layer-$num_head-$head_size-$relu_hidden_size-$parents_penalty-$rel_penalty-$parents_layer-$predicate_layer-$ss-$full_parse-$one_example_per_predicate"
                                                                        orig_parents_layer=$parents_layer
                                                                        eval_parse="True"
                                                                        rel_loss_penalty=$rel_penalty
                                                                        arc_loss_penalty=$parents_penalty
                                                                        if [[ "$parents_layer" == "no" ]]; then
                                                                            parents_layer=""
                                                                            eval_parse="False"
                                                                            rel_loss_penalty=0.0
                                                                            arc_loss_penalty=0.0
                                                                        fi

                                                                        partition="m40-long"

                                                                        ss_arr=(${ss//=/ })
                                                                        sampling_sched=${ss_arr[0]}
                                                                        sample_prob=${ss_arr[1]}


                                                                        commands+=("srun --gres=gpu:1 --partition=$partition --mem=24G python network.py  \
                                                                        --config_file config/trans-conll05-bio-manualattn-goldtrigs-sdeps-tanconf.cfg \
                                                                        --save_dir $OUT_LOG/scores-$fname_append \
                                                                        --save_every 500 \
                                                                        --train_iters 5000000 \
                                                                        --train_batch_size $batch_size \
                                                                        --test_batch_size $batch_size \
                                                                        --warmup_steps $warmup_steps \
                                                                        --learning_rate $lr \
                                                                        --n_recur $trans_layer \
                                                                        --num_heads $num_head \
                                                                        --head_size $head_size \
                                                                        --relu_hidden_size $relu_hidden_size \
                                                                        --mu $mu \
                                                                        --nu $nu \
                                                                        --epsilon $epsilon \
                                                                        --predicate_layer $predicate_layer \
                                                                        --multitask_layers \"$parents_layer\" \
                                                                        --multitask_penalties \"parents:$parents_penalty\" \
                                                                        --one_example_per_predicate $one_example_per_predicate \
                                                                        --eval_by_domain False \
                                                                        --eval_srl True \
                                                                        --ensure_tree True \
                                                                        --eval_parse $eval_parse \
                                                                        --full_parse $full_parse \
                                                                        --arc_loss_penalty $arc_loss_penalty \
                                                                        --rel_loss_penalty $rel_loss_penalty \
                                                                        --sampling_schedule $sampling_sched \
                                                                        --sample_prob $sample_prob \
                                                                        --save True \
                                                                        &> $OUT_LOG/train-$fname_append.log")
                                                                        i=$((i + 1))
                                                                        parents_layer=$orig_parents_layer
                                                                        eval_parse="True"
                                                                        arc_loss_penalty=$parents_penalty
                                                                        rel_loss_penalty=$rel_penalty
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