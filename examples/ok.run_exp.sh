export MASTER_ADDR='localhost'
export MASTER_PORT='12355'
export WORLD_SIZE=1
export RANK=0  # For the master process

export TRANSFORMERS_CACHE=cache

# output_dir=results/new.non_private/dart/
# data_dir=table2text/data/prefix-tuning
# task_mode=dart
# model_name_or_path=gpt2
# target_epsilon=3
# cache_dir=table2text/cache
# clipping_mode=ghost
# non_private=yes
# gpu_id=0
# per_example_max_grad_norm=0

# bash run.sh $output_dir $data_dir $task_mode $model_name_or_path $target_epsilon $cache_dir $clipping_mode $non_private $gpu_id $per_example_max_grad_norm
# wait

#  decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
output_dir=results/non_private/e2e
data_dir=table2text/prefix-tuning  # table2text/data/prefix-tuning
task_mode=e2e
model_name_or_path=gpt2
target_epsilon=3
cache_dir=table2text/cache
clipping_mode=ghost
non_private=yes
gpu_id=0
per_example_max_grad_norm=0

bash ok.run.sh $output_dir $data_dir $task_mode $model_name_or_path $target_epsilon $cache_dir $clipping_mode $non_private $gpu_id $per_example_max_grad_norm
wait
