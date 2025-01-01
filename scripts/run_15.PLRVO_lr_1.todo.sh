# config_idx=14

# noise_type=Gaussian
# gpu_id=1
# taskname=sst2
# per_device_train_batch_size=150

# bash run_config_idx_func_2models.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
# wait


noise_type=PLRVO
per_device_train_batch_size=170
lr=5e-5
taskname=sst2


config_idx=15
gpu_id=0
bash run_config_idx_func_lr_model2_large1.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr


# config_idx=15
# gpu_id=1
# bash run_config_idx_func_lr_model2_large2.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr


# config_idx=18
# gpu_id=2
# bash run_config_idx_func_lr_model2_large1.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr


# config_idx=18
# gpu_id=3
# bash run_config_idx_func_lr_model2_large2.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr



# config_idx=14
# gpu_id=4
# bash run_config_idx_func_lr_model2_large1.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr


# config_idx=14
# gpu_id=5
# bash run_config_idx_func_lr_model2_large2.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr



# noise_type=Gaussian

# config_idx=15
# gpu_id=6
# bash run_config_idx_func_lr_model2_large1.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr


# config_idx=15
# gpu_id=7
# bash run_config_idx_func_lr_model2_large2.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr

