# Copyright (c) Xuechen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper launcher script."""

import os

import fire

from .src import common


def _get_command(
    gpu_id,
    per_device_train_batch_size,
    task_name,
    model_name_or_path,
    noise_type,
    config_idx,
    output_dir,
    gradient_accumulation_steps,
    learning_rate,
    clipping_mode,
    few_shot_type,
    data_dir,
    seed,
    randomly_initialize,
    store_grads,
    attention_only,
    static_lm_head,
    static_embedding,
    eval_steps,
    eval_spectrum,
    max_spectrum_batches,
    max_lanczos_iter,
    num_train_epochs,
    orthogonal_projection_path,
    orthogonal_projection_rank,
    non_private,
):
    if task_name == "sst2": task_name = "sst-2"
    data_dir = f"{data_dir}/{common.task_name2suffix_name[task_name]}"
    template = {
        "sst-2": "*cls**sent_0*_It_was*mask*.*sep+*",
        "mnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qqp": "*cls**sent-_0**mask*,*+sentl_1**sep+*",
    }[task_name]

    cmd = f'''
CUDA_VISIBLE_DEVICES={gpu_id} python -m classification.run_classification_final \
  --per_device_train_batch_size {per_device_train_batch_size} \
  --task_name {task_name} --model_name_or_path {model_name_or_path} \
  --noise_type {noise_type} --config_idx {config_idx} \
  --non_private {non_private} --output_dir {output_dir} --overwrite_output_dir \
  --gradient_accumulation_steps {gradient_accumulation_steps} --num_train_epochs {num_train_epochs} \
  --learning_rate {learning_rate} --clipping_mode {clipping_mode} --few_shot_type {few_shot_type} --data_dir {data_dir} \
  --seed {seed} --randomly_initialize {randomly_initialize} --store_grads {store_grads} --template {template} \
  --attention_only {attention_only} --static_lm_head {static_lm_head} --static_embedding {static_embedding} \
  --eval_steps {eval_steps} --eval_spectrum {eval_spectrum} --max_spectrum_batches {max_spectrum_batches} --max_lanczos_iter {max_lanczos_iter} \
  --weight_decay 0 --lr_decay no --adam_epsilon 1e-08 --max_seq_len 256 --per_device_eval_batch_size 100 \
  --evaluation_strategy steps --evaluate_before_training True --do_train --do_eval --num_sample 1 --num_k 1 \
  --first_sent_limit 200 --other_sent_limit 200 --truncate_head yes '''
    if orthogonal_projection_path is not None:
        cmd += f' --orthogonal_projection_path {orthogonal_projection_path}'
        cmd += f' --orthogonal_projection_rank {orthogonal_projection_rank}'
    return cmd


def main(
    gpu_id=0,
    per_device_train_batch_size=170,
    task_name="sst2",
    model_name_or_path="roberta-base",
    noise_type="Gaussian",
    config_idx=0,
    output_dir="results",
    gradient_accumulation_steps=6,
    learning_rate=1e-3,
    num_train_epochs=3,
    clipping_mode="ghost",
    few_shot_type="finetune", # "prompt"
    data_dir="classification/data/original",
    seed=42,
    randomly_initialize="no",
    store_grads="no",
    attention_only="no",
    static_lm_head="no",
    static_embedding="no",
    eval_steps=100,
    eval_spectrum="no",
    max_spectrum_batches=2,
    max_lanczos_iter=2,
    orthogonal_projection_path=None,
    orthogonal_projection_rank=100,
):

    if noise_type == "non" or int(config_idx)==0:
        non_private = "yes"
    elif noise_type == "Gaussian" or noise_type == "PLRVO":
        assert int(config_idx)>0

        non_private = "no"
    else:
        print("reinput noise type. exit...")
        exit(0)
    
    print(non_private)
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
        learning_rate=learning_rate,
        clipping_mode=clipping_mode,
        few_shot_type=few_shot_type,
        data_dir=data_dir,
        seed=seed,
        randomly_initialize=randomly_initialize,
        store_grads=store_grads,
        attention_only=attention_only,
        static_lm_head=static_lm_head,
        static_embedding=static_embedding,
        eval_steps=eval_steps,
        eval_spectrum=eval_spectrum,
        max_spectrum_batches=max_spectrum_batches,
        max_lanczos_iter=max_lanczos_iter,
        num_train_epochs=num_train_epochs,
        orthogonal_projection_path=orthogonal_projection_path,
        orthogonal_projection_rank=orthogonal_projection_rank,
        non_private=non_private,
    )
    print('Running command:')
    print(command)
    os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
