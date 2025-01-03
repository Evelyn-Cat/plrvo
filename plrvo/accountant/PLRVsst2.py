# comprehensive configs

import copy, json
import numpy as np
import pandas as pd

generate_dict=True
dataset="sst-2"
# dataset="qnli"
# dataset="mnli"
# dataset="qqp"
confix_index_start = {
    "sst-2": 140,
    # "qnli": 242,
    # "mnli": 346,
    # "qqp": 445,
    # "e2e": 511,
    # "dart": 611,
    # "cifar10": 711,
    # "p100": 811,
    # "cifar100": 911,
}

# universal results
filename = "PLRVsst2.csv"
P0 = pd.read_csv(filename)
P0.columns = ["eps_check", "distortion_PLRV/C", "delta", "C", "q", "k", "theta", "T", "lambda_max", "matching_entries/cli"]

print(P0.shape) # (113209, 10)



eps_list = [8, 5, 4, 3, 2.5, 2, 1.5, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
used_ones = []
for jdx, fix_epsilon in enumerate(eps_list):
    print(f"\n\n\nfixed epsilon is {fix_epsilon}\n")
    
    P = copy.deepcopy(P0)

    P = P[P["eps_check"] < fix_epsilon]
    
    P["dist_to_fix_eps"] = fix_epsilon - P["eps_check"]

    P = P.sort_values(by='distortion_PLRV/C', ascending=True)
    
    try:
        for key, group in P.groupby('dist_to_fix_eps'):
            top_P = group.nsmallest(1, 'distortion_PLRV/C').index
            break
        kdx = P.loc[top_P, :].index.tolist()
        dict_P = P.loc[top_P[0], :].to_dict()
        print(dict_P)
        
        
        dict_P["idx_P"] = int(kdx[0])
        dict_P["a_G"] = 1
        dict_P["distributions"] = ["Gamma"]
        print(dict_P)
        
        idx = int(jdx+confix_index_start[dataset])
        if generate_dict==True:
            with open(f"../configs/{idx}.json", "w") as json_file:
                json.dump(dict_P, json_file, indent=4, ensure_ascii=False)
        else:
            print(dict_P)
        
        used_ones.append(int(kdx[0]))
    except:
        pass
     

# start_idx = idx
# P0 = P0.sort_values(by='distortion_PLRV/C', ascending=True)
# P0_1000 = P0.index.tolist()
# head_P0 = [item for item in P0_1000 if item not in used_ones]

# import random
# random.seed(42)
# random_selection = random.sample(head_P0, 10)

# print(random_selection)
# for jkdx, index_num in enumerate(random_selection):
#     row = P0.loc[index_num, :].to_dict()
#     row['idx'] = int(index_num)
    
#     if generate_dict:
#         output_idx = int(jkdx+start_idx+1)
#         with open(f"../configs/{output_idx}.json", "w") as json_file:
#             json.dump(row, json_file, indent=4, ensure_ascii=False)
#     else:
#         print(index_num)
#         print(row)



# start_idx = int(output_idx+1)
# print(start_idx)
# print(start_idx)
# P0 = P0.sort_values(by='eps_check', ascending=True)
# interval = len(P0.index.tolist()) // 10
# selected_values = P0['eps_check'].iloc[::interval].index.tolist()
# print(selected_values)

# for okkdx, index_num in enumerate(selected_values):
#     row = P0.loc[index_num, :].to_dict()
#     row['idx'] = int(index_num)
    
#     if generate_dict:
#         with open(f"../configs/{int(okkdx+start_idx)}.json", "w") as json_file:
#             json.dump(row, json_file, indent=4, ensure_ascii=False)
#     else:
#         print(index_num)
#         print(row)

