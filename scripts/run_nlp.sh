#!/bin/bash

#### This script will
#### 1. create the $output_dir; if it exists, exit.
#### 2. set log file path which loated in $output_dir/$exp_name.log.
#### 3. automatically calculate gradient_accumulation_steps.
#### 4. running the tasks.


export TRANSFORMERS_CACHE=cache

gpu_id=$1
per_device_train_batch_size=$2
taskname=$3
modelname=$4
noise_type=$5
target_epsilon=$6
per_example_max_grad_norm=$7
batch_size=$8
output_dir=$9

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
    echo "Directory $output_dir already exists. Exiting..."
    exit 0
fi
echo "The output directory is $output_dir"

#### 2 create .log file which loated in $output_dir/$exp_name.log
logfile=$output_dir/$exp_name.log
echo "logfile is in $logfile"

#### 3 automatically calculate gradient_accumulation_steps
gradient_accumulation_steps=$(($batch_size/$per_device_train_batch_size))
echo "gradient_accumulation_steps is $gradient_accumulation_steps"

cd ../examples/
#### 4 running the tasks: classification tasks
case $taskname in
    "sst-2"|"qnli"|"mnli"|"qqp")
        running_module=classification.run_wrapper_final
        ;;
    "e2e"|"dart")
        running_module=table2text.run_wrapper_final
        ;;
    *)
        task_type="cv"
        ;;
esac


#### 4 running the tasks: generation tasks
echo running $taskname
python -m $running_module \
    --gpu_id $gpu_id \
    --per_device_train_batch_size $per_device_train_batch_size \
    --task_name $taskname \
    --model_name_or_path $modelname \
    --noise_type $noise_type \
    --target_epsilon $target_epsilon  \
    --per_example_max_grad_norm $per_example_max_grad_norm \
    --output_dir $output_dir \
    --gradient_accumulation_steps $gradient_accumulation_steps 2>&1 > $logfile