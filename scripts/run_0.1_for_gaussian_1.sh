#!/bin/bash

#### This script will
#### 1. create the $output_dir; if it exists, exit.
#### 2. set log file path which loated in $output_dir/$exp_name.log.
#### 3. automatically calculate gradient_accumulation_steps.
#### 4. running the tasks.

export TRANSFORMERS_CACHE=cache

gpu_id=1
per_device_train_batch_size=170
taskname=sst-2
modelname=roberta-base
noise_type=Gaussian
# target_epsilon=2.5
# target_epsilon=1.9
# target_epsilon=1.7
# target_epsilon=1.4
# target_epsilon=1.1
# target_epsilon=0.8
target_epsilon=0.7
# target_epsilon=0.2
noise_multiplier=1.5701171875
per_example_max_grad_norm=0.1
batch_size=1024
current_path="$PWD"
output_dir=$current_path/results/classification/sst2/C_0.1

echo " "
echo Running on GPU: $gpu_id
echo Running $per_device_train_batch_size samples on each GPU
echo Runing $taskname on $modelname with $noise_type noise: C=$per_example_max_grad_norm, bz=$batch_size
echo " "

#### 1 check if the $output_dir exists: if not, create the path, if exists, exit
if [ "$target_epsilon" == "0" ]; then
    exp_name=$noise_type.$modelname.clip_$per_example_max_grad_norm
else
    exp_name=$noise_type.$modelname.clip_$per_example_max_grad_norm.eps_$target_epsilon
fi
output_dir=$output_dir/$exp_name

if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    echo "Directory $output_dir created."
else
    echo "Directory already exists. Exiting..."
    echo $output_dir
    exit 0
fi
echo "The output directory is $output_dir"

#### 2 create .log file which loated in $output_dir/$exp_name.log
logfile=$output_dir/$exp_name.log
echo "logfile is in $logfile"

#### 3 automatically calculate gradient_accumulation_steps
gradient_accumulation_steps=$(($batch_size/$per_device_train_batch_size))
echo "gradient_accumulation_steps is $gradient_accumulation_steps"

#### 4 running the tasks
cd ../examples/
python -m classification.run_wrapper \
    --gpu_id $gpu_id \
    --per_device_train_batch_size $per_device_train_batch_size \
    --task_name $taskname \
    --model_name_or_path $modelname \
    --noise_multiplier $noise_multiplier  \
    --per_example_max_grad_norm $per_example_max_grad_norm \
    --output_dir $output_dir \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 5e-4 2>&1 > $logfile
