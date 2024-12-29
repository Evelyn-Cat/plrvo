#!/bin/bash

batch_size=1024

gpu_id=$1
per_device_train_batch_size=$2
taskname=$3
modelname=$4
noise_type=$5
target_epsilon=$6
per_example_max_grad_norm=$7
output_dir=$8

