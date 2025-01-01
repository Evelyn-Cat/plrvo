import copy, json
import numpy as np
import pandas as pd
from scipy.io import loadmat


P0 = loadmat("e2e_both.mat")["Qn"]
P0 = pd.DataFrame(P0)
P0.columns = ["target_epsilon", "distortion_PLRV", "target_delta", "C", "sample_rate", "G_k", "G_theta", "steps", "??", "distortion"]

G0 = loadmat("e2e_both.mat")["gaussian_results"]
G0 = pd.DataFrame(G0)
G0.columns = ["gaussian_eps", "gaussian_q", "gaussian_sigma", "clip", "distortion_Gaussian"]


for jdx, fix_epsilon in enumerate([2.5, 1.9, 1.7, 1.4, 1.1, 0.8]): #  0.7, 0.5, 0.2 cannot find solution
    print(f"\n\n\nfixed epsilon is {fix_epsilon}\n")
    
    P = copy.deepcopy(P0)
    G = copy.deepcopy(G0)

    P = P[P["target_epsilon"] < fix_epsilon]
    G = G[G["gaussian_eps"] < fix_epsilon]

    print(P.head(10))
    print(G.head(10))
    
    dict_G = G.iloc[0].to_dict()
    print(dict_G)
    
    P["diff_dist"] = dict_G["distortion_Gaussian"] - P["distortion_PLRV"]
    top_P = P.nlargest(1, 'diff_dist').index
    print(P.loc[top_P, :])
    dict_P = P.loc[top_P, :].iloc[0].to_dict()
    
    idx = int(jdx+19)
    dict_P["a_G"] = 1
    dict_P["distributions"] = ["Gamma"]
    dict_P.update(dict_G)
    print(dict_P)
    with open(f"/mnt/backup/home/qiy22005/PRO/plrvo/plrvo/configs/{idx}.json", "w") as json_file:
        json.dump(dict_P, json_file, indent=4, ensure_ascii=False)


