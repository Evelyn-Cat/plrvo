import json
import os; os.environ["TRANSFORMERS_CACHE"] = "/mnt/backup/home/qiy22005/PRO/plrvo/plrvo/cache"
import sys;sys.path.append("/mnt/backup/home/qiy22005/PRO/plrvo")
from private_transformers.accounting import accounting_manager
from private_transformers.accounting import rdp_accounting
DEFAULT_ALPHAS = tuple(1 + x / 10.0 for x in range(1, 100)) + tuple(range(12, 64))  # RDP.

exit(0)

idx = 14
filepath = f"configs/{idx}.json"
with open(filepath, 'r', encoding='utf-8') as file:
    data = json.load(file)
    print(data)

if "sigma" not in data:
    target_epsilon = data['eps']
    target_delta = data['delta']
    sample_rate = data['sample_rate']
    steps = data['steps']
    L2_sensitivity = data['C']
    
    manager = accounting_manager.RDPManager(DEFAULT_ALPHAS)
    noise_multiplier = manager.compute_sigma(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=steps
    ) 
    sigma = noise_multiplier * L2_sensitivity
    print(noise_multiplier)
    print(sigma)

    data["noise_multiplier"] = noise_multiplier
    data["sigma"] = sigma

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
