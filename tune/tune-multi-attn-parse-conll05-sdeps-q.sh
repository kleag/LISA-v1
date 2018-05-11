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
num_gpus=100

lrs="0.04 0.06 0.08 0.1 0.02" # 0.06"
mus="0.9 0.99"
nus="0.98 0.999"
epsilons="1e-12 1e-4 1e-8"
warmup_steps="0 4000 8000 10000"
batch_sizes="5000"

trans_layers="10" # "10 8 6" # 3
cnn_layers="0"
cnn_dims="1024" # 768
num_heads="8" #4 8"
head_sizes="25"
relu_hidden_sizes="800"

predicate_mlp_sizes="200 100"

reps="5"

# 5*2*2*3*4*5 = 1200

# Settings and Regularization
#The settings of our models
#are described as follows. The dimension of word embeddings
#and predicate mask embeddings is set to 100 and the
#number of hidden layers is set to 10. We set the number
#of hidden units d to 200. The number of heads h is set to
#8. We apply dropout (Srivastava et al. 2014) to prevent the
#networks from over-fitting. Dropout layers are added before
#residual connections with a keep probability of 0.8. Dropout
#is also applied before the attention softmax layer and the
#feed-froward ReLU hidden layer, and the keep probabilities
#are set to 0.9. We also employ label smoothing technique
#(Szegedy et al. 2016) with a smoothing value of 0.1
#during training.

# Learning
#Parameter optimization is performed using
#stochastic gradient descent. We adopt Adadelta (Zeiler
#2012) (eps=10e6 and p=0.95) as the optimizer. To avoid
#exploding gradients problem, we clip the norm of gradients
#with a predefined threshold 1.0 (Pascanu et al. 2013).
#Each SGD contains a mini-batch of approximately 4096 tokens
#for the CoNLL-2005 dataset and 8192 tokens for the
#CoNLL-2012 dataset. The learning rate is initialized to 1.0.
#After training 400k steps, we halve the learning rate every
#100K steps. We train all models for 600K steps. For DEEPATT
#with FFN sub-layers, the whole training stage takes
#about two days to finish on a single Titan X GPU, which is
#2.5 times faster than the previous approach (He et al. 2017).

# 2*4 = 8

# array to hold all the commands we'll distribute
declare -a commands

i=1
for lr in ${lrs[@]}; do
    for mu in ${mus[@]}; do
        for nu in ${nus[@]}; do
            for epsilon in ${epsilons[@]}; do
                for warmup_steps in ${warmup_steps[@]}; do
                    for cnn_dim in ${cnn_dims[@]}; do
                        for cnn_layer in ${cnn_layers[@]}; do
                            for trans_layer in ${trans_layers[@]}; do
                                for num_head in ${num_heads[@]}; do
                                    for head_size in ${head_sizes[@]}; do
                                        for relu_hidden_size in ${relu_hidden_sizes[@]}; do
                                            for batch_size in ${batch_sizes[@]}; do
                                                for predicate_mlp_size in ${predicate_mlp_sizes[@]}; do
                                                    for rep in `seq $reps`; do
                                                        fname_append="$rep-$lr-$mu-$nu-$epsilon-$warmup_steps-$batch_size-$cnn_dim-$cnn_layer-$trans_layer-$num_head-$head_size-$relu_hidden_size-$predicate_mlp_size"

                                                        partition="qnd"

                                                        commands+=("srun --gres=gpu:1 --partition=$partition --mem=24G python network.py  \
                                                        --config_file config/trans-conll05-bio-goldtrigs-tan.cfg \
                                                        --save_dir $OUT_LOG/scores-$fname_append \
                                                        --save_every 500 \
                                                        --train_iters 5000000 \
                                                        --train_batch_size $batch_size \
                                                        --test_batch_size $batch_size \
                                                        --warmup_steps $warmup_steps \
                                                        --learning_rate $lr \
                                                        --cnn_dim $cnn_dim \
                                                        --cnn_layers $cnn_layer \
                                                        --n_recur $trans_layer \
                                                        --num_heads $num_head \
                                                        --head_size $head_size \
                                                        --relu_hidden_size $relu_hidden_size \
                                                        --predicate_mlp_size $predicate_mlp_size \
                                                        --predicate_pred_mlp_size $predicate_mlp_size \
                                                        --role_mlp_size $predicate_mlp_size \
                                                        --mu $mu \
                                                        --nu $nu \
                                                        --epsilon $epsilon \
                                                        --eval_by_domain False \
                                                        --eval_srl True \
                                                        --save False \
                                                        &> $OUT_LOG/train-$fname_append.log")
                                                        i=$((i + 1))
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