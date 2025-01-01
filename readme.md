## set parameters in scripts/run_config_idx.sh and then "cd scripts" and "bash run_config_idx.sh"

## run_7.**.sh is an example to use previous version but check run_config_idx.sh for this update.

1. run config_idx: 15 - 18 for

(1) task_name: qqp, mnli
(2) modelname: roberta-large bert-large-uncased [I will run this by myself: roberta-base bert-base-uncased]


2. For each config_idx, the estimation time is:
roberta-large:      mnli [11h] & qqp [13h]
bert-large-uncased: mnli [11h] & qqp [13h]


3. if using 8 GPUs, 36 hrs can run all the settings.
