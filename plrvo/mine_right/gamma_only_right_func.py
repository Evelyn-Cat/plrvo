import os, sys, json
import pandas as pd
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path) 

sys.path.append("../..")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(current_dir, "..", "..", "cache")
from private_transformers.accounting import accounting_manager

foldername = sys.argv[1] if len(sys.argv) > 1 else None
delta = sys.argv[2] if len(sys.argv) > 2 else None

if foldername is None:
    print('check path. exit...')
    exit(0)

folderPath = f'gamma_only_right/{foldername}'
alphas = tuple(range(2,256))

for folderName in os.listdir(folderPath):
    if not folderName.startswith("SearchResults"):
        continue
    
    folderName = os.path.join(folderPath, folderName)
    for filename in os.listdir(folderName):
        dataset, sample_size, batch_size, epoch = filename.split(".")[0].split("_")
        sample_size, batch_size, epoch = int(sample_size), int(batch_size), int(epoch)
        print(dataset, sample_size, batch_size, epoch)
        
        filename_prefix = os.path.splitext(filename)[0]
        filepath = os.path.join(folderName, filename)
        if filepath.endswith(".json"):
            with open(filepath, 'r') as file:
                param = json.load(file)  # Parse the JSON data into a Python dictionary

            print(param)
            
            filepath = os.path.join(folderName, filename_prefix + ".csv")
            S = pd.read_csv(filepath)
            
            if S.empty:
                print("empty")
                continue
            else:
                print("not empty")
            
            cnt= 0
            S_new = pd.DataFrame([], columns = S.columns)
            all_epsilon = S.target_epsilon.tolist()
            for idx, i in enumerate(all_epsilon):
                try:
                    if "i" not in i: #  and float(i) < 0.1:
                        S_new.loc[cnt, :] = S.loc[idx, :]
                        cnt = cnt + 1
                except:
                    S_new.loc[cnt, :] = S.loc[idx, :]
                    cnt = cnt + 1
            
            # generate_config.SR.eps20.log
            X=S_new.target_epsilon.tolist()
            print(sorted(X)[:20])
            
            
            # generate_config.SR.all20.log
            S_new_sort = S_new.sort_values("target_epsilon")
            print(S_new_sort.head(20))

            S_new_sort = S_new.sort_values("target_epsilon")
            eps_check = float(S_new_sort.target_epsilon.tolist()[0])
            manager = accounting_manager.RDPManager(alphas=alphas)
            try:
                if delta == None:
                    delta = param['target_delta']
                noise_multiplier = manager.compute_sigma(
                    target_epsilon = eps_check,
                    target_delta = delta,
                    sample_rate = param['sample_rate'],
                    steps = param['steps'],
                )
                
                sigma = noise_multiplier * param["C"]
            except:
                noise_multiplier = None
                sigma = None
            print(f"target_delta: {delta}, noise_multiplier: {noise_multiplier}, sigma: {sigma}\n")
   
