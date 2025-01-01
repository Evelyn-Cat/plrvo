
gpu_id=2
per_device_train_batch_size=150

for taskname in e2e
do
    for config_idx in 24 # 18 17 16 15 13 12 11 10 9 8 7 6 5 4 3 2 1 
    do
        noise_type=PLRVO
        bash run_config_idx_func.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
        wait
        
        # exit 0
        # noise_type=Gaussian
        # bash run_config_idx_func.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size
        # wait
        
    done
done
