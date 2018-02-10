#!/bin/bash

debug="false"

timestamp=`date +%Y-%m-%d-%H-%M-%S`
ROOT_DIR=$DOZAT_ROOT

OUT_LOG=$ROOT_DIR/hyperparams/tune-$timestamp
if [[ "$debug" == "false" ]]; then
    mkdir -p $OUT_LOG
fi

echo "Writing to $OUT_LOG"

num_splits=10

num_gpus=$num_splits

# array to hold all the commands we'll distribute
declare -a commands
i=1

data_dir="/home/strubell/research/data/conll-2012/conll2012-train-jackknife"
for split in `seq $num_splits`; do
    fname_append="split$split"
    commands+=("srun --gres=gpu:1 --partition=titanx-long --mem=16000 --time=12:00:00 python network.py \
    --config_file config/trans-conll12-bio-justpos.cfg \
    --train_file $data_dir/train_$split \
    --dev_file $data_dir/test_$split \
    --save_dir $OUT_LOG/scores-$fname_append \
    --save_every 500 \
    --train_iters 500000 \
    &> $OUT_LOG/train-$fname_append.log")
    i=$((i + 1))
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
