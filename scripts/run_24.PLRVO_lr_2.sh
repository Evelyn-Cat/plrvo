
config_idx=24

noise_type=PLRVO
gpu_id=2
taskname=e2e
per_device_train_batch_size=120
lr=5e-6

bash run_config_idx_func_tmp_for_gen.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr

