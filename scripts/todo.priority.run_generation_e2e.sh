# priority
config_idx=506
config_idx=505
config_idx=501

# lr=5e-5
# lr=5e-6


for config_idx in 506 505 501
do
noise_type=Gaussian
gpu_id=0
taskname=e2e
per_device_train_batch_size=200
lr=5e-5

bash run_config_idx_func_tmp_for_gen.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr

done