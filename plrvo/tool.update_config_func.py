import json
# import os; os.environ["TRANSFORMERS_CACHE"] = "add your path"
import sys;sys.path.append("..")
from private_transformers.accounting import accounting_manager
from private_transformers.accounting import rdp_accounting
DEFAULT_ALPHAS = tuple(1 + x / 10.0 for x in range(1, 100)) + tuple(range(12, 64))  # RDP.

import sys; idx = int(sys.argv[1]) if len(sys.argv) > 1 else exit(0)

# idx = 14
filepath = f"configs/{idx}.json"
with open(filepath, 'r', encoding='utf-8') as file:
    data = json.load(file)
    print(data)


if "??" in data:
    data["rdp_order"] = data.pop("??")
    

if "sigma" not in data:
    if "target_epsilon" in data:
        target_epsilon = data["target_epsilon"]
    elif "eps" in data:
        target_epsilon = data["eps"]
        
    if "target_delta" in data:
        target_delta = data["target_delta"]
    elif "delta" in data:
        target_delta = data['delta']

    if "q" in data:
        sample_rate = data['q']
    elif "sample_rate" in data:
        sample_rate = data['sample_rate']
    
    if "T" in data:
        steps = data['T']
    elif "Ts" in data:
        steps = data['Ts']
    elif "step" in data:
        steps = data['steps']
    elif "steps" in data:
        steps = data['steps']
    
    if "clipping" in data:
        L2_sensitivity = data['clipping']
    elif "C" in data:
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
