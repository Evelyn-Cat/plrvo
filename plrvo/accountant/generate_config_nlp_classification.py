import copy, json
import numpy as np
import pandas as pd

generate_dict=True
dataset="sst-2"
dataset="qnli"
dataset="mnli"
dataset="qqp"
confix_index_start = {
    "sst-2": 111,
    "qnli": 211,
    "mnli": 311,
    "qqp": 411,
    "e2e": 511,
    "dart": 611,
    "cifar10": 711,
    "p100": 811,
    "cifar100": 911,
}

G0 = pd.read_csv(f"csv/{dataset}_gaussian_results.csv")
G0.columns = ["gaussian_eps", "gaussian_q", "gaussian_sigma", "clip", "gaussian_dist/clip"]

P0 = pd.read_csv(f"csv/{dataset}_Qn.csv")
P0.columns = ["eps_check", "distortion_PLRV/C", "delta", "C", "q", "k", "theta", "T", "lambda_max", "matching_entries/cli"]

# print(G0)
# print(P0)

# print(P0.shape)
# print(G0.shape)
# (1538, 10)
# (19999, 5)


eps_list = [8, 5, 4, 3, 2.5, 2, 1.5, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
for jdx, fix_epsilon in enumerate(eps_list):
    print(f"\n\n\nfixed epsilon is {fix_epsilon}\n")
    
    P = copy.deepcopy(P0)
    G = copy.deepcopy(G0)

    P = P[P["eps_check"] < fix_epsilon]
    G = G[G["gaussian_eps"] < fix_epsilon]
    
    P["dist_to_fix_eps"] = fix_epsilon - P["eps_check"]
    G["dist_to_fix_eps"] = fix_epsilon - G["gaussian_eps"]

    P = P.sort_values(by='distortion_PLRV/C', ascending=True)
    
    try:
        for key, group in P.groupby('dist_to_fix_eps'):
            top_P = group.nsmallest(1, 'distortion_PLRV/C').index
            break
        kdx = P.loc[top_P, :].index.tolist()
        dict_P = P.loc[top_P[0], :].to_dict()
        print(dict_P)
        
        G = G.sort_values(by='dist_to_fix_eps', ascending=True)
        for key, group in G.groupby('dist_to_fix_eps'):
            print(group)
            top_G = group.nlargest(1, 'gaussian_dist/clip').index
            break
        print(top_G[0])
        dict_G = G.loc[top_G[0], :].to_dict()
        print(dict_G)
        dict_P["idx_P"] = int(kdx[0])
        dict_P["a_G"] = 1
        dict_P["distributions"] = ["Gamma"]
        dict_P.update(dict_G)
        dict_P['diff_distortion'] = dict_P["gaussian_dist/clip"] - dict_P["distortion_PLRV/C"]
           
        idx = int(jdx+confix_index_start[dataset])
        if generate_dict:
            with open(f"../configs/{idx}.json", "w") as json_file:
                json.dump(dict_P, json_file, indent=4, ensure_ascii=False)
        else:
            print(dict_P)
    except:
        pass
     

start_idx = idx
P0 = P0.sort_values(by='distortion_PLRV/C', ascending=True)
head_P0 = P0.head(20).index.tolist()
print(head_P0)
import random
random.seed(42)
random_selection = random.sample(head_P0, 10)


for jkdx, index_num in enumerate(random_selection):
    row = P0.loc[index_num, :].to_dict()
    rowp['idx'] = int(index_num)
    
    if generate_dict:
        with open(f"../configs/{int(jkdx+start_idx)}.json", "w") as json_file:
            json.dump(row, json_file, indent=4, ensure_ascii=False)
    else:
        print(random_selection)
        print(row)