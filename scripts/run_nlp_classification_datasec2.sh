export TRANSFORMERS_CACHE=cache

cd ../examples/

gpu_id=$1
task_name=$2
batch_size=$3
output_dir=$4
noise_type=$5
target_epsilon=$6
per_example_max_grad_norm=$7
modelname=$8
per_device_train_batch_size=$9

echo noise type is $noise_type and related parameters is $target_epsilon
echo " "

if [ "$target_epsilon" == "None" ]; then
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


logfile=$output_dir/$exp_name.log
echo $logfile

if [ "$noise_type" == "Non_private" ]; then
    echo "Noise type is Non_private. Running Non_private process..."
    
    python -m classification.run_wrapper \
    --gpu_id $gpu_id \
    --task_name $task_name \
    --model_name_or_path $modelname \
    --batch_size $batch_size \
    --noise_multiplier 0 \
    --target_delta $target_delta \
    --per_example_max_grad_norm 0 \
    --output_dir $output_dir \
    --learning_rate 1e-3 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --few_shot_type finetune 2>&1 > $logfile

elif [ "$noise_type" == "Gaussian" ]; then
    echo "Noise type is Gaussian. Running process with Gaussian noise..."

    python -m classification.run_wrapper \
    --gpu_id $gpu_id \
    --task_name $task_name \
    --model_name_or_path $modelname \
    --batch_size $batch_size \
    --target_epsilon $target_epsilon \
    --target_delta $target_delta \
    --per_example_max_grad_norm $per_example_max_grad_norm \
    --output_dir $output_dir \
    --learning_rate 1e-3 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --few_shot_type finetune 2>&1 > $logfile
else
    echo "Noise type is PLRVO. Running process with PLRVO noise..."
    echo "Notice that the varname 'target_epsilon' is the index of PLRVO config files..."
    echo "the config files are located in 'plrvo/configs/*.json' ..."

    python -m classification.run_wrapper2 \
    --gpu_id $gpu_id \
    --task_name $task_name \
    --model_name_or_path $modelname \
    --batch_size $batch_size \
    --config_idx $target_epsilon \
    --target_delta $target_delta \
    --per_example_max_grad_norm $per_example_max_grad_norm \
    --output_dir $output_dir \
    --learning_rate 1e-3 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --few_shot_type finetune 2>&1 > $logfile
fi

# 170