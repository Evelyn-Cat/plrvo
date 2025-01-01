## run this for clip>1 but close to 1 and the results are from meisam is in 20241230/Top4.mat for SST2

# gaussian_eps, gaussian_q, gaussian_sigma, clip, gaussian_dist

import time, copy, json
import pandas as pd
import numpy as np
from scipy.io import loadmat

P0 = loadmat("Q_192.mat")['Q']
P0 = pd.DataFrame(P0)
P0.columns = ["target_epsilon", "distortion_PLRV", "target_delta", "C", "sample_rate", "G_k", "G_theta", "steps"]

G0 = loadmat("Gaussiancomprehensive.mat")['gaussian_results']
G0 = pd.DataFrame(G0)
G0.columns = ["gaussian_eps", "gaussian_q", "gaussian_sigma", "clip", "gaussian_dist"]
G0["sigma"] = G0["distortion_Gaussian"] / np.sqrt(2 / np.pi)


for jdx, fix_epsilon in enumerate([1.9, 1.7, 1.4, 1.1, 0.8, 0.7]):
    print(f"\n\n\nfixed epsilon is {fix_epsilon}\n")
    
    P = copy.deepcopy(P0)
    G = copy.deepcopy(G0)

    P = P[P["target_epsilon"] < fix_epsilon]
    G = G[G["epsilon"] < fix_epsilon]

    print(P.head(10))
    print(G.head(10))
    
    dict_G = G.iloc[0].to_dict()
    print(dict_G)
    
    P["diff_dist"] = dict_G["distortion_Gaussian"] - P["distortion_PLRV"]
    top_P = P.nlargest(1, 'diff_dist').index
    print(P.loc[top_P, :])
    dict_P = P.loc[top_P, :].iloc[0].to_dict()
    
    idx = int(jdx+8)
    dict_P["a_G"] = 1
    dict_P["distributions"] = ["Gamma"]
    dict_P.update(dict_G)
    print(dict_P)
    with open(f"/mnt/backup/home/qiy22005/PRO/plrvo/plrvo/configs/{idx}.json", "w") as json_file:
        json.dump(dict_P, json_file, indent=4, ensure_ascii=False)
    