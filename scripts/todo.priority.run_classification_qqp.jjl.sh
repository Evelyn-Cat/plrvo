# priority
noise_type=PLRVO
taskname=qqp
config_idx=423



per_device_train_batch_size=96 # suggestion: memory 80G and running large model

gpu_id=0
lr=5e-5
bash run_config_idx_func_lr_roberta_large.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr &

gpu_id=1
lr=5e-6
bash run_config_idx_func_lr_roberta_large.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr &
    
gpu_id=2
lr=5e-5
bash run_config_idx_func_lr_bert_large.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr &

gpu_id=3
lr=5e-6
bash run_config_idx_func_lr_bert_large.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr &



per_device_train_batch_size=170 # suggestion: memory 40G and running base model

gpu_id=4
lr=5e-5
bash run_config_idx_func_lr_roberta_base.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr &

gpu_id=5
lr=5e-6
bash run_config_idx_func_lr_roberta_base.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr &
    
gpu_id=6
lr=5e-5
bash run_config_idx_func_lr_roberta_base.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr &

gpu_id=7
lr=5e-6
bash run_config_idx_func_lr_roberta_base.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr &

wait

while true; do
    gpu_id=7
    noise_type=PLRVO
    taskname=mnli
    config_idx=324

    lr=5e-6
    per_device_train_batch_size=96 # suggestion: memory 80G and running large model
    bash run_config_idx_func_lr_roberta_large.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr &
done