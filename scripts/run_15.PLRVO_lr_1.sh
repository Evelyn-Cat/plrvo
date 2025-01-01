# config_idx=14

# noise_type=Gaussian
# gpu_id=1
# taskname=sst2
# per_device_train_batch_size=150

# bash run_config_idx_func_2models.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
# wait


config_idx=15

noise_type=PLRVO
gpu_id=1
per_device_train_batch_size=170
lr=5e-5

taskname=sst2
bash run_config_idx_func_lr_model2.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr

