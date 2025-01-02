# priority
config_idx=607
config_idx=606
config_idx=605

# lr=5e-5
# lr=5e-6


for config_idx in 506 505 501
do
    for noise_type in Gaussian PLRVO
    do
        gpu_id=1
        taskname=dart
        per_device_train_batch_size=200
        # lr=5e-5

        bash run_config_idx_func_tmp_for_gen_dart.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr
    done
done