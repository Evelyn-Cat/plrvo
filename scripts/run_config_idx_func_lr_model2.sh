#!/bin/bash

#### Setting the parameters and then this script will
#### 1. automatically set output_dir: $pwd_folder/results/${task_type}/${taskname}.
#### 2. automatically call scripts/run_${task_type}.sh for different $taskname.

#### Following parameters need to be set
# noise_type
# gpu_id
## per_device_train_batch_size
## batch_size
# config_idx
## array_modelname
## taskname
# output folder name (if needed): $output_dir


## How to use this file:
## bash run_config_idx_func.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
## bash run_config_idx_func.sh Gaussian 0 14 sst2 150

#### 1 choose noise type
# noise_type=non
# noise_type=Gaussian
# noise_type=Laplace
# noise_type=PLRVO
noise_type=$1

#### 2 setting hyperparamters (applicable for each noise type)
# gpu_id=0
gpu_id=$2
config_idx=$3
taskname=$4
per_device_train_batch_size=$5
lr=$6
# per_device_train_batch_size=170 # classification task: 170 on datasec; 340 on datasec2;
# per_device_train_batch_size=200 # generation task: 200 [gpt2]
batch_size=1024
# echo $noise_type $gpu_id $config_idx $taskname
# exit 0

#### 3 setting privacy parameters
case $noise_type in
    "Gaussian"|"Laplace"|"PLRVO")
        # use config_idx to choose .json file from ./plrvo/config/${idx}.json
        config_idx=$config_idx  # Example index, you can change this value as needed
        echo "Setting privacy for $noise_type noise where config idx: $config_idx"
        ;;
    "non")
        config_idx=0
        echo "Runing non-private Mode."
        ;;
    *)
        echo "Unknown noise type: $noise_type"
        exit 1
        ;;
esac

#### 4 choose modelname and taskname to run
# array_modelname=(roberta-large bert-large-uncased)
# array_modelname=(roberta-base bert-base-uncased)
# array_modelname=(roberta-base)
# array_modelname=(roberta-base bert-base-uncased roberta-large bert-large-uncased)
array_modelname=(bert-base-uncased roberta-base)
# array_modelname=(roberta-large bert-large-uncased roberta-base bert-base-uncased)
# array_modelname=(distilgpt2) # distilgpt2, gpt2, gpt2-medium, gpt2-large
# array_modelname=(vit)
## [fine-tuning] running nlp classification task
taskname=$taskname
# taskname=qnli
# taskname=mnli
# taskname=qqp
## [training] running nlp generation task
# taskname=e2e # TODO
# taskname=dart
## [training] running cv task
# taskname=cifar100
# taskname=cifar10
# taskname=mnist # kmnist fmnist
# taskname=svhn
# taskname=cinic10 # cinic # https://github.com/BayesWatch/cinic-10#data-loading


### automatically choose the script to run
case $taskname in
    "sst2"|"qnli"|"mnli"|"qqp"|"sst-2")
        task_type="classification"
        ;;
    "e2e"|"dart")
        task_type="generation"
        ;;
    *)
        task_type="cv"
        ;;
esac

### automatically set the output directory
pwd_folder=`pwd`
echo "Curent Directory is $pwd_folder"
output_dir=$pwd_folder/results/${task_type}/${taskname}

#### 5 running the scripts
for modelname in "${array_modelname[@]}"; do
    # bash run_${task_type}.sh $gpu_id $per_device_train_batch_size $taskname $modelname $noise_type $config_idx $batch_size $output_dir
    bash running_config_idx_lr.sh $gpu_id $per_device_train_batch_size $taskname $modelname $noise_type $config_idx $batch_size $output_dir $lr
    wait
done

