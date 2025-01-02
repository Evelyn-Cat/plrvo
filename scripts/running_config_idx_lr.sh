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
config_idx=$6
batch_size=$7
output_dir=$8
lr=$9

echo " "
echo Running on GPU: $gpu_id
echo Running $per_device_train_batch_size samples on each GPU
echo Runing $taskname on $modelname with $noise_type noise: bz=$batch_size
echo " "

#### 1 check if the $output_dir exists: if not, create the path, if exists, exit
if [ "$config_idx" == "0" ]; then
    exp_name=$noise_type.$modelname
else
    exp_name=$noise_type.$modelname.config_$config_idx
fi
output_dir=$output_dir/$exp_name

if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    echo "Directory $output_dir created."
else
    date=$(date +%Y%m%d%H%M%S)
    output_dir=${output_dir}_${date}
    # echo "Directory already exists. Exiting..."
    echo $output_dir
    mkdir -p "$output_dir"
    echo "Directory $output_dir created."
    # exit 0
fi
echo "The output directory is $output_dir"

#### 2 create .log file which loated in $output_dir/$exp_name.log
logfile=$output_dir/$exp_name.log
echo "logfile is in $logfile"

#### 3 automatically calculate gradient_accumulation_steps
gradient_accumulation_steps=$(awk -v a="$batch_size" -v b="$per_device_train_batch_size" 'BEGIN {result = a / b; print int(result) + (result > int(result))}')
echo "gradient_accumulation_steps is $gradient_accumulation_steps"

cd ../examples/
#### 4 running the tasks: classification tasks
case $taskname in
    "sst2"|"qnli"|"mnli"|"qqp"|"sst-2")
        task_type=nlp
        running_module=classification.run_wrapper_final
        ;;
    "e2e"|"dart")
        task_type=nlp
        running_module=table2text.run_wrapper_final
        
        export MASTER_ADDR='localhost'
        export MASTER_PORT='12355'
        export WORLD_SIZE=1
        export RANK=0  # For the master process

        ;;
    *)
        task_type=cv
        running_module=image_classification.main
        ;;
esac


#### 4 running the tasks: generation tasks
echo running $taskname ...
if [ "$task_type" == "nlp" ]; then
    echo $running_module
    
    # cmd="$running_module \
    #     --gpu_id $gpu_id \
    #     --per_device_train_batch_size $per_device_train_batch_size \
    #     --task_name $taskname \
    #     --model_name_or_path $modelname \
    #     --noise_type $noise_type \
    #     --config_idx $config_idx  \
    #     --output_dir $output_dir \
    #     --learning_rate $lr \
    #     --gradient_accumulation_steps $gradient_accumulation_steps 2>&1 > $logfile"
    # echo $cmd
    
    python3 -m $running_module \
        --gpu_id $gpu_id \
        --per_device_train_batch_size $per_device_train_batch_size \
        --task_name $taskname \
        --model_name_or_path $modelname \
        --noise_type $noise_type \
        --config_idx $config_idx  \
        --output_dir $output_dir \
        --learning_rate $lr \
        --gradient_accumulation_steps $gradient_accumulation_steps 2>&1 > $logfile
else
    echo $running_module
    
    CUDA_VISIBLE_DEVICES=$gpu_id python3 -m $running_module \
        --per_device_train_batch_size $per_device_train_batch_size \
        --task_name $taskname \
        --model_name_or_path $modelname \
        --noise_type $noise_type \
        --config_idx $config_idx \
        --output_dir $output_dir \
        --learning_rate $lr \
        --batch_size $batch_size 2>&1 > $logfile
fi

# git add -f results/*/*/*/final_results.json
# git add -f results/*/*/*/*log
# git add -f results/*/*/*/log_history.json