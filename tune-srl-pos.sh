#!/bin/bash

debug="false"

timestamp=`date +%Y-%m-%d-%H-%M-%S`
ROOT_DIR=$DOZAT_ROOT

OUT_LOG=$ROOT_DIR/hyperparams/tune-$timestamp
if [[ "$debug" == "false" ]]; then
    mkdir -p $OUT_LOG
fi

echo "Writing to $OUT_LOG"

#num_gpus=96
num_gpus=50

lrs="0.04" # 0.06"
mus="0.9"
nus="0.98"
epsilons="1e-12"
warmup_steps="4000" # 2000 1000"
batch_sizes="3500"

#learn_rates = [0.04, 0.08, 0.1, 0.02]
#warmup_steps = [2000, 4000, 8000, 16000]
#decays = [1.5, 1.25, 0.75, 1.0, 1.75]

trans_layers="4" # 3
cnn_dims="1024" # 512 768 1024"
num_heads="8" # 4 8"
head_sizes="64" # 128"
relu_hidden_sizes="256"
trigger_mlp_sizes="256"
trigger_pred_mlp_sizes="256"
role_mlp_sizes="256"
#add_pos_tags="False"
#pos_layers="-1 0 1 2 3"
#trigger_layers="-1 0 1 2 3"
#aux_trigger_layers="-1 0 1 2"
# 5*5*4*2 = 200

add_pos_tags="True False"
pos_layers="0"
trigger_layers="-1 0 1 2 3"
aux_trigger_layers="-1 0 1 2 no"
# 5*5*2*2 = 100

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
                                                                for aux_trigger_layer in ${aux_trigger_layers[@]}; do
                                                                    for pos_layer in ${pos_layers[@]}; do
                                                                        for rep in `seq $reps`; do
                                                                            fname_append="$rep-$lr-$mu-$nu-$epsilon-$warmup_steps-$batch_size-$cnn_dim-$trans_layer-$num_head-$head_size-$relu_hidden_size-$role_mlp_size-$trigger_mlp_size-$trigger_pred_mlp_size-$add_pos-p$pos_layer-t$trigger_layer-at$aux_trigger_layer"
                                                                            partition="titanx-long"
                                                                            if [[ $((i % 4)) == 0 ]]; then
                                                                                partition="m40-long"
                                                                            fi

                                                                            train_pos="True"
                                                                            if [[ "$add_pos_tags" == "True" ]]; then
                                                                                train_pos="False"
                                                                            fi

                                                                            train_aux_trigger_layer="True"
                                                                            if [[ "$aux_trigger_layer" == "no" ]]; then
                                                                                train_aux_trigger_layer="False"
                                                                                old_aux_trigger_layer=$aux_trigger_layer
                                                                                aux_trigger_layer=0
                                                                            fi

                                                                            commands+=("srun --gres=gpu:1 --partition=$partition --mem=16000 --time=24:00:00 python network.py \
                                                                            --config_file config/trans-conll12-bio.cfg \
                                                                            --save_dir $OUT_LOG/scores-$fname_append \
                                                                            --save_every 500 \
                                                                            --train_iters 500000 \
                                                                            --train_batch_size $batch_size \
                                                                            --test_batch_size $batch_size \
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
                                                                            --pos_layer $pos_layer \
                                                                            --train_pos $train_pos \
                                                                            --trigger_layer $trigger_layer \
                                                                            --aux_trigger_layer $aux_trigger_layer \
                                                                            --train_aux_trigger_layer $train_aux_trigger_layer \
                                                                            --svd_tree False \
                                                                            --mask_pairs True \
                                                                            --mask_roots True \
                                                                            --ensure_tree True \
                                                                            --save False \
                                                                            &> $OUT_LOG/train-$fname_append.log")
                                                                            i=$((i + 1))
                                                                            aux_trigger_layer=$old_aux_trigger_layer
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
