## run 20241229 for clip>1 but close to 1
## the results are from meisam is in 20241230/Top4.mat for SST2
import time, copy, json
import pandas as pd
import numpy as np
from scipy.io import loadmat

P0 = loadmat("Top4.mat")['sampled_data']
P0 = pd.DataFrame(P0)
P0.columns = ["target_epsilon", "distortion_PLRV", "target_delta", "C", "sample_rate", "G_k", "G_theta", "steps", "??", "distortion"]

for jdx in P0.index.tolist():
    dict_P = P0.loc[jdx, :].to_dict()
    
    idx = jdx + 15
    with open(f"/mnt/backup/home/qiy22005/PRO/plrvo/plrvo/configs/{idx}.json", "w") as json_file:
        json.dump(dict_P, json_file, indent=4, ensure_ascii=False)
    