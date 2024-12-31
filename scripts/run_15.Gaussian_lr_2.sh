# config_idx=14

# noise_type=Gaussian
# gpu_id=1
# taskname=sst2
# per_device_train_batch_size=150

# bash run_config_idx_func_2models.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
# wait

config_idx=15

noise_type=Gaussian
gpu_id=2
taskname=sst2
per_device_train_batch_size=170
lr=5e-4

for taskname in sst2 qnli mnli qqp
do
    bash run_config_idx_func_lr_model2.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr
    wait
done