gpu_id=1
per_device_train_batch_size=150

for taskname in sst2 qnli
do
    for config_idx in 7 8 9 10 11 12
    do
        noise_type=PLRVO
        bash run_config_idx_func_4models.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
        wait

        noise_type=Gaussian
        bash run_config_idx_func_4models.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
        wait
        
    done
done