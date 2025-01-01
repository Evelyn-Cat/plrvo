# e2e: 501-600;
# dart: 601-700;
config_idx=501
gpu_id=0
per_device_train_batch_size=340

noise_type=PLRVO
taskname=e2e
lr=5e-4 # 5e-5 5e-6

bash run_config_idx_func_tmp_for_gen.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr


