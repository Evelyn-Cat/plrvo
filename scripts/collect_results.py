import os, re, json
import pandas as pd
results = {}

taskname = "sst-2"
taskname = "qnli"
# taskname = "mnli"
# taskname = "qqp"

foldername = "results"
task_types = ["classification"]
# task_types = ["classification", "generation", "cv"]
for task_type in task_types:
    folderpath = os.path.join(foldername, task_type, taskname)

    results[task_type] = {}
    for idx, filename in enumerate(os.listdir(folderpath)):
        results[task_type][idx] = {}
        match = re.match(r"(\w+)\.(\S+)\.clip_(\d+)\.eps_(\d(?:\.\d+)?)", filename)
        noise_type, modelname, C, target_epsilon = match[1], match[2], match[3], match[4]

        if noise_type == "PLRVO":
            config_idx = int(target_epsilon)
            with open(f"../plrvo/configs/{config_idx}.json", 'r', encoding='utf-8') as file:
                data = json.load(file)
                target_epsilon = data['target_epsilon']
        
        filepath = os.path.join(folderpath, filename, "final_results.json")
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        eval_acc = data[taskname]['eval_acc']
        
        # results[task_type][idx]['filename'] = filepath
        # results[task_type][idx]['filepath'] = filepath
        results[task_type][idx]["noise_type"] = noise_type
        results[task_type][idx]["modelname"] = modelname
        results[task_type][idx]["C"] = C
        results[task_type][idx]["target_epsilon"] = target_epsilon
        results[task_type][idx]["eval_acc"] = eval_acc
    
    df = pd.DataFrame(results[task_type]).T
    sorted_df = df.sort_values(by=["modelname"]).reset_index(drop=True)
    print(sorted_df)

    # import ace_tools as tools; tools.display_dataframe_to_user(name="Sorted Noise and Modelname Data", dataframe=sorted_df)


