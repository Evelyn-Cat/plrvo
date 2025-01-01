"""Wrapper launcher script."""

import os

import fire


def _get_command(
    gpu_id,
    per_device_train_batch_size,
    task_name,
    model_name_or_path,
    noise_type,
    config_idx,
    output_dir,
    gradient_accumulation_steps,
    clipping_mode,
    few_shot_type,
    data_dir,
    learning_rate,
    num_train_epochs,
    seed,
    non_private,
):
    if task_name == "e2e":
        learning_rate = 2e-3 if task_name=="e2e" else 5e-4
        num_train_epochs = 10 if task_name=="e2e" else 15
        max_seq_len = 100 if task_name=="e2e" else 120
    elif task_name == "dart":
        learning_rate = 5e-4
        num_train_epochs = 15
        max_seq_len = 120

    if data_dir is None:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        foldername = "e2e_data" if task_name == "e2e" else "dart"
        data_dir = f"table2text/prefix-tuning/data/{foldername}"
    
    cmd = f'''
CUDA_VISIBLE_DEVICES={gpu_id} python -m table2text.run_language_modeling \
  --per_device_train_batch_size {per_device_train_batch_size} --tokenizer_name {model_name_or_path} \
  --task_mode {task_name} --model_name_or_path {model_name_or_path} \
  --noise_type {noise_type} --config_idx {config_idx} \
  --non_private {non_private} --output_dir {output_dir} --overwrite_output_dir \
  --gradient_accumulation_steps {gradient_accumulation_steps} --num_train_epochs {num_train_epochs} \
  --learning_rate {learning_rate} --clipping_mode {clipping_mode} --data_folder {data_dir} \
  --seed {seed} --eval_steps 100 --lr_decay "no" --max_seq_len {max_seq_len} --do_train --do_eval \
  --per_device_eval_batch_size 100 --evaluation_strategy epoch --evaluate_before_training True --evaluate_during_training "yes" \
  --line_by_line --save_steps 100 --save_total_limit 1 --save_at_last no \
  --logging_dir {output_dir} --logging_steps -1 --eval_epochs 2 --max_eval_batches 100 \
  --max_generations 9223372036854775807 --max_generations_train 10 --max_generations_valid 9223372036854775807 \
  --max_train_examples 9223372036854775807 --max_valid_examples 9223372036854775807 --max_eval_examples 9223372036854775807 \
  --format_mode cat --cache_dir ../cache'''
    return cmd


def main(
    gpu_id=0,
    per_device_train_batch_size=200,
    task_name="e2e",
    model_name_or_path="gpt2", # distilgpt2, gpt2, gpt2-medium, gpt2-large
    noise_type="Gaussian",
    config_idx=0,
    output_dir="results",
    gradient_accumulation_steps=64,
    learning_rate=None,
    num_train_epochs=None,
    clipping_mode="ghost",
    few_shot_type="finetune", # "prompt"
    data_dir=None,
    seed=42,
    non_private="yes",
):
    
    if noise_type == "non" or int(config_idx) == 0:
        non_private = "yes"
    elif noise_type == "Gaussian" or noise_type == "PLRVO":
        assert int(config_idx)>0
        non_private = "no"
    else:
        print("reinput noise type. exit...")
    
    if non_private == "yes":
        assert int(config_idx) == 0
    else:
        assert int(config_idx) > 0
        assert os.path.exists(f"../plrvo/configs/{config_idx}.json")
    
    command = _get_command(
        gpu_id=gpu_id,
        per_device_train_batch_size=per_device_train_batch_size,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        noise_type=noise_type,
        config_idx=config_idx,
        output_dir=output_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clipping_mode=clipping_mode,
        few_shot_type=few_shot_type,
        data_dir=data_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        seed=seed,
        non_private=non_private,
    )
    print('Running command:')
    print(command)
    os.system(command)


if __name__ == "__main__":
    # gradient_accumulation_steps = batch_size // per_device_train_batch_size
    fire.Fire(main)
