# priority
noise_type=PLRVO
gpu_id=1
taskname=qnli
per_device_train_batch_size=170 # suggestion: memory 40G and running base model
per_device_train_batch_size=96 # suggestion: memory 80G and running large model


for config_idx in 220 219
do
    for lr in 5e-5 5e-6
    do
    # add & at the end of the line for parallel running.
    bash run_config_idx_func_lr_bert_base.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr
    # bash run_config_idx_func_lr_bert_large.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr
    # bash run_config_idx_func_lr_roberta_base.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr
    # bash run_config_idx_func_lr_roberta_large.sh $noise_type $gpu_id $config_idx $taskname $per_device_train_batch_size $lr
    wait
    done
done
