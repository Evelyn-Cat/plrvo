config_idx=14

noise_type=PLRVO
gpu_id=0
taskname=sst2
per_device_train_batch_size=150

bash run_config_idx_func_2models.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
wait
