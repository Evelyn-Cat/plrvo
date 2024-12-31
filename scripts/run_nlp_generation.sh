export TRANSFORMERS_CACHE=cache

cd ../examples/
pwd_folder=`pwd`
echo $pwd_folder
data_dir=$pwd_folder/table2text/data


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

gradient_accumulation_steps=$(($batch_size/$per_device_train_batch_size))
cache_dir=cache_dir_check
clipping_mode=ghost

if [[ ${task_name} == "e2e" ]]; then
  data_dir="${data_dir}/data/e2e_data"
  echo $data_dir
  target_delta=1.18874967309384e-05
  num_train_epochs=10
  learning_rate=2e-3
  max_seq_len=100
else
  if [[ ${task_name} == "dart" ]]; then
    target_delta=1e-5 # 1e-5
    data_dir="${data_dir}/data/dart"
    num_train_epochs=15 # Approximately same number of updates.
    learning_rate=5e-4  # Lower learning rate for stability in large models.
    max_seq_len=120
  else
    echo "Unknown task: ${task_mode}"
    exit 1
  fi
fi

non_private=no
if [ "$noise_type" == "Non_private" ]; then
    echo "Noise type is Non_private. Running Non_private process..."
    
    # TODO
    non_private=yes

elif [ "$noise_type" == "Gaussian" ]; then
    echo "Noise type is Gaussian. Running process with Gaussian noise..."

    python -m table2text.run_language_modeling \
  --output_dir ${output_dir} --overwrite_output_dir \
  --task_mode ${task_name} \
  --model_name_or_path ${modelname} \
  --tokenizer_name ${modelname} \
  --do_train --do_eval \
  --line_by_line \
  --save_steps 100 --save_total_limit 1 --save_at_last no \
  --logging_dir ${output_dir} --logging_steps -1 \
  --seed 0 \
  --eval_steps 100 --eval_epochs 2 --max_eval_batches 100 --evaluation_strategy epoch --evaluate_before_training "no" --evaluate_during_training "yes" --per_device_eval_batch_size 100 \
  --max_generations 9223372036854775807 --max_generations_train 10 --max_generations_valid 9223372036854775807 \
  --max_train_examples 9223372036854775807 --max_valid_examples 9223372036854775807 --max_eval_examples 9223372036854775807 \
  --data_folder ${data_dir} --max_seq_len ${max_seq_len} --format_mode cat \
  --per_example_max_grad_norm ${per_example_max_grad_norm} --target_delta ${target_delta} --target_epsilon ${target_epsilon} \
  --learning_rate ${learning_rate} --lr_decay "no" --num_train_epochs ${num_train_epochs} --per_device_train_batch_size ${per_device_train_batch_size} --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --non_private ${non_private} \
  --clipping_mode "${clipping_mode}" \
  --cache_dir ${cache_dir} 2>&1 > $logfile
else
    echo "Noise type is PLRVO. Running process with PLRVO noise..."
    echo "Notice that the varname 'target_epsilon' is the index of PLRVO config files..."
    echo "the config files are located in 'plrvo/configs/*.json' ..."

    # python -m classification.run_wrapper2 \
    # --gpu_id $gpu_id \
    # --task_name $task_name \
    # --model_name_or_path $modelname \
    # --batch_size $batch_size \
    # --config_idx $target_epsilon \
    # --target_delta $target_delta \
    # --per_example_max_grad_norm $per_example_max_grad_norm \
    # --output_dir $output_dir \
    # --learning_rate 1e-3 \
    # --few_shot_type finetune
    
    python -m table2text.run_language_modeling2 \
  --output_dir ${output_dir} --overwrite_output_dir \
  --task_mode ${task_name} \
  --model_name_or_path ${model_name_or_path} \
  --tokenizer_name ${model_name_or_path} \
  --do_train --do_eval \
  --line_by_line \
  --save_steps 100 --save_total_limit 1 --save_at_last no \
  --logging_dir ${output_dir} --logging_steps -1 \
  --seed 0 \
  --eval_steps 100 --eval_epochs 2 --max_eval_batches 100 --evaluation_strategy epoch --evaluate_before_training "no" --evaluate_during_training "yes" --per_device_eval_batch_size 100 \
  --max_generations 9223372036854775807 --max_generations_train 10 --max_generations_valid 9223372036854775807 \
  --max_train_examples 9223372036854775807 --max_valid_examples 9223372036854775807 --max_eval_examples 9223372036854775807 \
  --data_folder ${data_dir} --max_seq_len ${max_seq_len} --format_mode cat \
  --per_example_max_grad_norm ${per_example_max_grad_norm} --target_delta ${target_delta} --target_epsilon ${target_epsilon} \
  --learning_rate ${learning_rate} --lr_decay "no" --num_train_epochs ${num_train_epochs} --per_device_train_batch_size ${per_device_train_batch_size} --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --non_private ${non_private} \
  --clipping_mode "${clipping_mode}" \
  --cache_dir ${cache_dir} 2>&1 > $logfile
fi
