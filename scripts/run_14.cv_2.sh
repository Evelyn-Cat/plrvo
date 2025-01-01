# gpu_id=2 need to be runned
gpu_id=2
per_device_train_batch_size=150

for taskname in sst2
do
    for config_idx in 13 14 15 16 17 18
    do
        noise_type=PLRVO
        bash run_config_idx_func_4models.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
        wait

        noise_type=Gaussian
        bash run_config_idx_func_4models.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
        wait
        
    done
done