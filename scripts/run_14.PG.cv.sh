#!/bin/bash

#### Setting the parameters and then this script will
#### 1. automatically set output_dir: $pwd_folder/results/${task_type}/${taskname}.
#### 2. automatically call scripts/run_${task_type}.sh for different $taskname.

#### Following parameters need to be set
# noise_type
# gpu_id
## per_device_train_batch_size
## per_example_max_grad_norm
## batch_size
# target_epsilon | config_idx
## array_modelname
## taskname
# output folder name (if needed): $output_dir

#### 1 choose noise type
# noise_type=non
# noise_type=Gaussian
# noise_type=Laplace
# noise_type=PLRVO
noise_type=$1
config=$2

#### 2 setting hyperparamters (applicable for each noise type)
gpu_id=0
per_device_train_batch_size=170 # classification task: 170 on datasec; 340 on datasec2;
# per_device_train_batch_size=200 # generation task: 200 [gpt2]
per_example_max_grad_norm=3  # clipping threshold C
batch_size=1024

#### 3 setting privacy parameters
case $noise_type in
    "Gaussian"|"Laplace")
        # use target_epsilon for only Gaussian and Laplace noises
        target_epsilon=0.6776754134398772  # Example value, you can change this value as needed
        echo "Setting privacy for $noise_type noise where target_epsilon: $target_epsilon"
        ;;
    "PLRVO")
        # use config_idx for only PLRVO noise to choose .json file from ./plrvo/config/${idx}.json
        config_idx=$config  # Example index, you can change this value as needed
        echo "Setting privacy for PLRVO noise where config idx: $config_idx"
        ;;
    "non")
        target_epsilon=0
        per_example_max_grad_norm=0
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
# array_modelname=(roberta-large bert-large-uncased roberta-base bert-base-uncased)
# array_modelname=(distilgpt2) # distilgpt2, gpt2, gpt2-medium, gpt2-large
<<<<<<< HEAD
# array_modelname=(vit)
=======
array_modelname=(vit)
>>>>>>> main
## [fine-tuning] running nlp classification task
# taskname=sst2
# taskname=qnli
# taskname=mnli
# taskname=qqp
## [training] running nlp generation task
# taskname=e2e # TODO
# taskname=dart
## [training] running cv task
# taskname=cifar100
taskname=cifar10
# taskname=mnist # kmnist fmnist
# taskname=svhn
# taskname=cinic10 # cinic # https://github.com/BayesWatch/cinic-10#data-loading


#### 5 running the scripts
array_taskname=($taskname)
for taskname in "${array_taskname[@]}"; do

    # ### automatically choose the script to run
    case $taskname in
        "sst2"|"qnli"|"mnli"|"qqp")
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

    for modelname in "${array_modelname[@]}"; do
        if [ "$noise_type" == "PLRVO" ]; then
            target_epsilon=$config_idx
        fi
        
        # bash run_${task_type}.sh $gpu_id $per_device_train_batch_size $taskname $modelname $noise_type $target_epsilon $per_example_max_grad_norm $batch_size $output_dir
        bash running.sh $gpu_id $per_device_train_batch_size $taskname $modelname $noise_type $target_epsilon $per_example_max_grad_norm $batch_size $output_dir
        wait
        
    done
done
