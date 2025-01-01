import copy, json
import numpy as np
import pandas as pd
from scipy.io import loadmat


P0 = loadmat("plrv_50000_147.mat")['Qn']
P0 = pd.DataFrame(P0)
P0.columns = ["target_epsilon", "distortion_PLRV", "target_delta", "C", "sample_rate", "G_k", "G_theta", "steps", "lambda_max", "matching_entries"]

G0 = loadmat("gaussian_50000_147.mat")['gaussian_results']
G0 = pd.DataFrame(G0)
G0.columns = ["gaussian_eps", "gaussian_q", "gaussian_sigma", "gaussian_clip", "distortion_Gaussian"]


# assert P0.shape == (507608, 10)
# assert G0.shape == (6000, 5)
print(P0.shape) # (374034, 10)
print(G0.shape) # (6000, 5)


for jdx, fix_epsilon in enumerate([2.5, 1.9, 1.7, 1.4, 1.1, 0.8, 0.5, 0.2]): #   cannot find solution
    print(f"\n\n\nfixed epsilon is {fix_epsilon}\n")
    
    P = copy.deepcopy(P0)
    G = copy.deepcopy(G0)

    P = P[P["target_epsilon"] < fix_epsilon]
    G = G[G["gaussian_eps"] < fix_epsilon]

    P["dist_to_fix_eps"] = fix_epsilon - P["target_epsilon"]
    G["dist_to_fix_eps"] = fix_epsilon - G["gaussian_eps"]

    P = P.sort_values(by='dist_to_fix_eps', ascending=True)
    
    for key, group in P.groupby('dist_to_fix_eps'):
        top_P = group.nsmallest(1, 'distortion_PLRV').index
        break
    dict_P = P.loc[top_P[0], :].to_dict()
    print(dict_P)
    
    
    try:
        G = G.sort_values(by='dist_to_fix_eps', ascending=True)
        for key, group in G.groupby('dist_to_fix_eps'):
            print(group)
            top_G = group.nlargest(1, 'distortion_Gaussian').index
            break
        print(top_G[0])
        dict_G = G.loc[top_G[0], :].to_dict()
        print(dict_G)
        dict_P["a_G"] = 1
        dict_P["distributions"] = ["Gamma"]
        dict_P.update(dict_G)
        dict_P['diff_distortion'] = dict_P["distortion_Gaussian"] - dict_P["distortion_PLRV"]
    except:
        dict_P["a_G"] = 1
        dict_P["distributions"] = ["Gamma"]
        
    idx = int(jdx+701)
    
    print(dict_P)
    with open(f"/mnt/backup/home/qiy22005/PRO/plrvo/plrvo/configs/{idx}.json", "w") as json_file:
        json.dump(dict_P, json_file, indent=4, ensure_ascii=False)


