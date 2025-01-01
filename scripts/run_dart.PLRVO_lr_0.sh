
for config_idx in 601 607
do
noise_type=PLRVO
gpu_id=0
taskname=dart
per_device_train_batch_size=200
lr=5e-4

bash run_config_idx_func_tmp_for_gen.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr
wait
done
