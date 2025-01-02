import sys; sys.path.append("..")
import sys; sys.path.append("../..")
from private_transformers.accounting import accounting_manager
from private_transformers.accounting import rdp_accounting

DEFAULT_ALPHAS = tuple(1 + x / 10.0 for x in range(1, 100)) + tuple(range(12, 64))  # RDP.

dataset="sst-2"
dataset="qnli"
dataset="mnli"
dataset="qqp"
dataset="e2e"
dataset="dart"
dataset="cifar10"
# dataset="p100"
from configs.datainfo import hyperparameters

start = hyperparameters[dataset]['start']
end = hyperparameters[dataset]['end']

import os, json
for filename in os.listdir("../configs"):
    if not filename.endswith(".json"):
        continue
    
    if int(filename.split(".json")[0]) > end or int(filename.split(".json")[0]) < start:
        continue
    
    file_path = os.path.join("../configs", filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        string = json.load(file)

    if "eps_check" in string or "target_epsilon" in string:
        if "eps_check" in string:
            target_epsilon=string['eps_check']
        else:
            target_epsilon=string['target_epsilon']

        manager = accounting_manager.RDPManager(DEFAULT_ALPHAS)
        noise_multiplier = manager.compute_sigma(
            target_epsilon=target_epsilon,
            target_delta=hyperparameters[dataset]["target_delta"],
            sample_rate=hyperparameters[dataset]["sample_rate"],
            steps=hyperparameters[dataset]["steps"]
        )

        print(noise_multiplier)

        string['paired_noise_multiplier'] = noise_multiplier
        string['paired_sigma'] = noise_multiplier * float(string['C'])

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(string, file, ensure_ascii=False, indent=4)
