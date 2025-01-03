# priority
config_idx=706
config_idx=705
config_idx=704


# lr=5e-5
# lr=5e-6


for config_idx in 706 705 704
do
    for noise_type in PLRVO Gaussian
    do
        # noise_type=Gaussian
        gpu_id=0
        taskname=cifar10
        per_device_train_batch_size=100
        # lr=5e-5

        bash run_config_idx_func_lr_vit.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr
    done
done

