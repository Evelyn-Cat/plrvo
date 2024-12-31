# CUDA_VISIBLE_DEVICES={gpu_id} python -m table2text.run_language_modeling_final \
#   --noise_type {noise_type} \
#   --config_idx {config_idx} \

export MASTER_ADDR='localhost'
export MASTER_PORT='12355'
export WORLD_SIZE=1
export RANK=0  # For the master process

export TRANSFORMERS_CACHE=cache


cd ../examples/

python -m table2text.run_wrapper_final

