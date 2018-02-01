#!/bin/bash

debug="false"

timestamp=`date +%Y-%m-%d-%H-%M-%S`
ROOT_DIR=$DOZAT_ROOT

OUT_LOG=$ROOT_DIR/hyperparams/tune-$timestamp
if [[ "$debug" == "false" ]]; then
    mkdir -p $OUT_LOG
fi

echo "Writing to $OUT_LOG"

num_gpus=108
#num_gpus=50

lrs="0.04" # 0.06"
mus="0.9"
nus="0.98"
epsilons="1e-12"
warmup_steps="4000" # 2000 1000"
batch_sizes="5000"

#learn_rates = [0.04, 0.08, 0.1, 0.02]
#warmup_steps = [2000, 4000, 8000, 16000]
#decays = [1.5, 1.25, 0.75, 1.0, 1.75]

trans_layers="4" # 3
cnn_dims="1024" # 512 768 1024"
num_heads="8" # 4 8"
head_sizes="64" # 128"
relu_hidden_sizes="256"
trigger_mlp_sizes="64 128 256"
trigger_pred_mlp_sizes="64 128 256"
role_mlp_sizes="64 128 256"
add_pos_tags="False"
#pos_layers="-1 0 1 2 3"
trigger_layers="3"
#aux_trigger_layers="-1 0 1 2"
# 5*5*4*2 = 200
mlp_keep_probs="0.67 0.34"
attn_dropouts="0.67 0.34"
prepost_dropouts="0.67 0.34"
relu_dropouts="0.67 0.34"

#add_pos_tags="False"
#pos_layers="0"
#trigger_layers="-1 0 1 2 3"
#aux_trigger_layers="-1 0 1 2 no"
# 5*5*2*2 = 100

# 2*2*2*2*2*3*3*3 = 864

reps="2"




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
                                            for role_mlp_size in ${role_mlp_sizes[@]}; do
                                                for trigger_mlp_size in ${trigger_mlp_sizes[@]}; do
                                                    for trigger_pred_mlp_size in ${trigger_pred_mlp_sizes[@]}; do
                                                        for add_pos in ${add_pos_tags[@]}; do
                                                            for trigger_layer in ${trigger_layers[@]}; do
                                                                for mlp_keep_prob in ${mlp_keep_probs[@]}; do
                                                                    for attn_dropout in ${attn_dropouts[@]}; do
                                                                        for prepost_dropout in ${prepost_dropouts[@]}; do
                                                                            for relu_dropout in ${relu_dropouts[@]}; do
                                                                                for rep in `seq $reps`; do
                                                                                    fname_append="$rep-$lr-$mu-$nu-$epsilon-$warmup_steps-$batch_size-$cnn_dim-$trans_layer-$num_head-$head_size-$relu_hidden_size-$role_mlp_size-$trigger_mlp_size-$trigger_pred_mlp_size-$add_pos-t$trigger_layer-$mlp_keep_prob-$attn_dropout-$prepost_dropout-$relu_dropout"

                                                                                    commands+=("srun --gres=gpu:1 --partition=titanx-short,m40-short --mem=12000 --time=04:00:00 python network.py \
                                                                                    --config_file config/trans-conll05-bio.cfg \
                                                                                    --save_dir $OUT_LOG/scores-$fname_append \
                                                                                    --save_every 500 \
                                                                                    --train_iters 500000 \
                                                                                    --train_batch_size $batch_size \
                                                                                    --warmup_steps $warmup_steps \
                                                                                    --learning_rate $lr \
                                                                                    --cnn_dim $cnn_dim \
                                                                                    --n_recur $trans_layer \
                                                                                    --num_heads $num_head \
                                                                                    --head_size $head_size \
                                                                                    --relu_hidden_size $relu_hidden_size \
                                                                                    --mu $mu \
                                                                                    --nu $nu \
                                                                                    --epsilon $epsilon \
                                                                                    --decay 1.5 \
                                                                                    --trigger_mlp_size $trigger_mlp_size \
                                                                                    --trigger_pred_mlp_size $trigger_pred_mlp_size \
                                                                                    --role_mlp_size $role_mlp_size \
                                                                                    --add_pos_to_input $add_pos \
                                                                                    --train_pos False \
                                                                                    --trigger_layer $trigger_layer \
                                                                                    --train_aux_trigger_layer False \
                                                                                    --save False \
                                                                                    &> $OUT_LOG/train-$fname_append.log")
                                                                                    i=$((i + 1))
        #                                                                            aux_trigger_layer=$old_aux_trigger_layer
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
