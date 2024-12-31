for config_idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
do
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
# bash run_14.PG.cv.sh Gaussian
# wait

# for idx in 2 3 4 5 6 7
# do
#     bash run_7.PG.cv.sh PLRVO $idx
#     wait
# done