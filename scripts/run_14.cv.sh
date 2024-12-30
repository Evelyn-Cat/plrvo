config_idx=14
noise_type=PLRVO
gpu_id=0
taskname=sst2
per_device_train_batch_size=150

bash run_config_idx_func.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size

for idx in 2 3 4 5 6 7
do
    bash run_7.PG.cv.sh PLRVO $idx
    wait
done
