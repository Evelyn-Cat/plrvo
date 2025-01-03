# latest
from typing import Dict, List, Any, Optional, Literal, Union
from itertools import product
import pandas as pd
import numpy as np
import os, ast, time, copy, json, math

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks")
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from plrvo_transformers.accountant import accounting_manager_check as accounting_manager
from plrvo_transformers.accountant import rdp_accounting_check as rdp_accounting


class plrv():
    def __init__(
        self,
        mode="PLRVO_DPSGD",
        *args,
        **kwargs
    ):
        if mode == "Search_Space_Finder":
            self.SS_Finder(*args, **kwargs)
        elif mode == "Noise_Parameters_Finder":
            self.Noise_Params_Finder(*args, **kwargs)
        elif mode == "PLRVO_DPSGD":
            self.init(*args, **kwargs)
    
    def check_valid(self, alphas, params, L2_sensitivity, distributions):
        valid = True
        for eta in alphas:
            lhs = rdp_accounting.cmp_lhs(params, L2_sensitivity, eta, distributions)
            rhs = rdp_accounting.cmp_rhs(params, L2_sensitivity, eta, distributions)
            
            if rhs == None or lhs == None or lhs < rhs:
                valid = False
                break
        
        return valid, rhs, lhs

    def SS_Finder(
        self,
        array_per_example_max_grad_norm: List[float] = [0.1, 0.5, 1], # clipping threshold - C
        alphas: Optional[float] = tuple(range(2, 256)),
        distributions: Optional[List[Literal["Gamma", "Exponential"]]] = ["Gamma"],
        filepath: Optional[str] = "../search_space_finder.csv",
        figure: Optional[str] = ["../search_space_finder.png", "../search_space.png"],
    ):
        """
        PLRV-O Noise Parameters Search Space Finder
        
        """
        start_time = time.time()


        S = []
        ## PLRV-O Noise Parameters Search Range
        if distributions==["Gamma"] or distributions==["Gamma", "Exponential"]:
            G_k1 = np.logspace(np.log10(1.00001), np.log10(5), 50)
            G_k2 = np.linspace(5, 100, 200)
            G_ks = np.concatenate((G_k1, G_k2))
            for G_k in G_ks:
                G_thetas = np.linspace(1 / (G_k - 1), 2 / (G_k - 1), 200)
                # TODO G_thetas = 1 - y * G_theta < 0 ; y = (eta - 1) * clip
                for G_theta in G_thetas:
                    if distributions == ["Gamma"]:
                        param = {"G_k": G_k, "G_theta": G_theta}
                        added = copy.deepcopy(param)
                        S.append(added)
                    
                    elif distributions == ["Gamma", "Exponential"]:
                        a_Gs = [0.1, 0.2, 0.5]
                        a_Es = [0.1, 0.2, 0.5]
                        E_lambdas = [1, 5, 10, 20]
                        for a_G, a_E, E_lambda in product(a_Gs, a_Es, E_lambdas):
                            param = {
                                "a_G": a_G,
                                "a_E": a_E,
                                "G_k": G_k,
                                "G_theta": G_theta,
                                "E_lambda": E_lambda,
                            }
                            added = copy.deepcopy(param)
                            S.append(added)
        
        elif distributions == ["Exponential"]:
            E_lambdas = [1, 5, 10, 20]
            for E_lambda in E_lambdas:
                param = {"E_lambda": E_lambda}
            added = copy.deepcopy(param)
            S.append(added)


        ## PLRV-O Noise Parameters Search Space
        SS = []
        nSS = []
        S = pd.DataFrame(S)
        for idx in S.index.tolist():
            params = dict(S.loc[idx, :])
            for L2_sensitivity in array_per_example_max_grad_norm:
                valid, rhs, lhs = self.check_valid(alphas, params, L2_sensitivity, distributions)
                
                if valid:
                    params["L2_sensitivity"] = L2_sensitivity
                    params["rhs"] = rhs
                    params["lhs"] = lhs
                    added = copy.deepcopy(params)
                    SS.append(added)
                else:
                    params["L2_sensitivity"] = L2_sensitivity
                    params["rhs"] = None
                    params["lhs"] = None
                    added = copy.deepcopy(params)
                    nSS.append(added)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = elapsed_time / 60
        print(f"Running time: {elapsed_time:.2f} seconds... [about {minutes:.2f} minutes]")
        

        if filepath!=None and SS!=[]:
            SS = pd.DataFrame(SS)
            SS.to_csv(filepath, index=None)
            print(f"Search Space is saved at: {filepath}")
            nSS = pd.DataFrame(nSS)
            
            if figure!=None:
                # figure 1
                morandi_colors = ['#E7D8C9', '#A7BED3']
                morandi_cmap = LinearSegmentedColormap.from_list("MorandiBlueYellow", morandi_colors)

                plt.figure(figsize=(10, 6))
                f_nSS = nSS[nSS['G_theta'] < 1]
                f_nSS = f_nSS[['G_k', 'G_theta']]
                plt.scatter(f_nSS.G_k.tolist(), f_nSS.G_theta.tolist(), color='gray', alpha=0.5, s=50)
                
                G_k = SS.G_k.tolist()
                G_theta = SS.G_theta.tolist()
                plt.scatter(G_k, G_theta, alpha=0.7, edgecolor=None, cmap=morandi_cmap, c=G_k)
                plt.xlabel(r"$G_k$")
                plt.ylabel(r"$G_\theta$")
                plt.colorbar(label="Based on $G_k$")

                plt.gca().set_facecolor('#F9F9F9')
                plt.grid(color='#E0E0E0', linestyle='--', linewidth=0.5)
                
                plt.savefig(figure[0])
                plt.clf()
                plt.cla()

                # figure 2
                morandi_colors = ['#E7D8C9', '#A7BED3']
                morandi_cmap = LinearSegmentedColormap.from_list("MorandiBlueYellow", morandi_colors)

                plt.figure(figsize=(10, 6))
                G_k = SS.G_k.tolist()
                G_theta = SS.G_theta.tolist()
                plt.scatter(G_k, G_theta, alpha=0.7, edgecolor=None, cmap=morandi_cmap, c=G_k)
                plt.xlabel(r"$G_k$")
                plt.ylabel(r"$G_\theta$")
                plt.colorbar(label="Based on $G_k$")

                plt.gca().set_facecolor('#F9F9F9')
                plt.grid(color='#E0E0E0', linestyle='--', linewidth=0.5)
                
                plt.savefig(figure[1])
            

        elif SS == []:
            print(f"valid_search_space is [], please expand seach range ...")
        else:
            return SS
    
    def Noise_Params_Finder(
        self,
        dataset=None,
        epoch=None,
        array_alphas: Optional[float] = tuple(range(2, 256)),
        filepath: Optional[str] = "../search_space_finder.csv",
        prefix: Optional[str] = "../search_space_",
        demo: bool = False,
    ):
        ## load SS
        SS = pd.read_csv(filepath)
        SS['distributions'] = SS['distributions'].apply(ast.literal_eval)
        distributions = SS['distributions'].values[0]
        
        self.idx =  [0, 3, 10, 20, 34, 52, 74, 100, 129, 162, 199, 239, 283, 332, 383, 439, 498, 561, 628, 697, 771, 849, 931, 1016, 1105, 1198, 1293, 1389, 1485, 1581, 1678, 1773, 1869, 1965, 2061, 2157, 2254, 2349, 2445, 2541, 2637, 2734, 2830, 2925, 3022, 3117, 3213, 3309, 3405, 3501, 3597, 3694, 3789, 3885, 3981, 4077, 4173, 4269, 4365, 4461, 4557, 4653, 4750, 4845, 4941, 5038, 5134, 5229, 5325, 5421, 5517, 5613, 5709, 5805, 5902, 5997, 6093, 6189, 6285, 6381, 6477, 6574, 6669, 6765, 6861, 6958, 7054, 7149, 7245, 7341, 7438, 7533, 7629, 7725, 7821, 7917, 8013, 8109, 8205, 8301, 8397, 8493, 8589, 8685, 8781, 8878, 8973, 9069, 9165, 9261, 9357, 9453, 9549, 9645, 9741, 9837, 9934, 10029, 10125, 10221, 10318, 10414, 10509, 10605, 10701, 10797, 10893, 10989, 11085, 11181, 11277, 11373, 11469, 11565, 11662, 11757, 11854, 11949, 12045, 12141, 12237, 12333, 12429, 12525, 12621, 12718, 12813, 12909, 13005, 13101, 13197, 13293, 13389, 13485, 13581]
        

        self.SS = SS
        self.array_alphas = array_alphas
        if demo: SS = SS.head(2)
        
        if ["Gamma"] == distributions:
            columns = ["G_k", "G_theta", "distributions"]
            SS['a_G'] = 1
        elif ["Exponential"] == distributions:
            columns = ["E_lambda", "distributions"]
            SS['a_E'] = 1
        elif ["Gamma", "Exponential"] == distributions:
            columns = ["a_G", "G_k", "G_theta", "a_E", "E_lambda", "distributions"]
        self.columns = columns
        
        self.sample_sizes = {
            "sst-2": 67349, # dev: 872
            "qnli": 104743, # dev: 5463
            "mnli": 392703, # (mnli-m) dev: 
            "qqp": 363847, # dev:
            "e2e": 42061, # dev:
            "dart": 93187, # dev:
            "cifar10": 50000, # dev: 10000
            "cifar100": 50000, # dev: 10000
            "svhn": 73257, # dev: 26032
            "mnist": 60000, # dev: 10000
            "fmnist": 60000, # dev: 10000
            "kmnist": 60000, # dev: 10000
            "p100": 10000, # dev: 10000? (197324, 197324)? 15w?
        }
        
        ## PLRV-O Noise Parameters Finder: Search S from SS
        if dataset != None:
            S = self.cmp_for_dataset(dataset, epoch)
            S.to_csv(prefix+dataset+".csv")
        else:
            for dataset in self.sample_sizes:
                S = self.cmp_for_dataset(dataset, epoch)
                S.to_csv(prefix+dataset+".csv")
    
    def cmp_for_dataset(
        self,
        dataset: str = None,
        epoch: float = None,
    ):
        epochs = {
            "sst-2": 3, "qnli": 3, "mnli": 3, "qqp": 3,
            "e2e": 10, "dart": 10, "cifar10": 10, "cifar100": 10,
            "svhn": 10, "mnist": 10, "fmnist": 10, "kmnist": 10, "p100": 10
        }
        
        try:
            epoch = epoch if epoch!=None else epochs[dataset]
        except:
            print('check the input of dataset name')
        
        sample_size = self.sample_sizes[dataset]
        len_SS = len(self.SS.index.tolist())
        
        # progress_bar = tqdm(self.SS.index.tolist(), desc="Processing target epsilon", position=0)
        progress_bar = tqdm(self.idx, desc="Processing target epsilon", position=0)
        for idx in progress_bar:
            L2_sensitivity = self.SS.loc[idx, "L2_sensitivity"]
            params = dict(self.SS.loc[idx, self.columns])

            target_delta =  1 / (2 * sample_size)
            sample_rate = 1024 / sample_size
            steps = math.ceil(epoch / sample_rate)
            
            self.init(
                params=params,
                distributions=params["distributions"],
                steps=steps,
                sample_rate=sample_rate,
                target_delta=target_delta,
                L2_sensitivity=L2_sensitivity,
                alphas=self.array_alphas,
                debug_mode=False,
            )
            self.pairs(return_noise_type=["g"], figures=False)
            
            print(self.target_epsilon, self.target_epsilon2, self.target_epsilon3)
            print(self.sigma, self.sigma2, self.sigma3)
            
            progress_bar.set_description(f"Processing idx: {idx} / {len_SS}")

            if self.target_epsilon3 < 10:
                self.SS.loc[idx, "target_epsilon"] = self.target_epsilon
                self.SS.loc[idx, "target_epsilon2"] = self.target_epsilon2
                self.SS.loc[idx, "target_epsilon3"] = self.target_epsilon3
                self.SS.loc[idx, "sigma"] = self.sigma
                self.SS.loc[idx, "sigma2"] = self.sigma2
                self.SS.loc[idx, "sigma3"] = self.sigma3
            else:
                self.SS.loc[idx, "target_epsilon"] = None
            
        self.SS.to_csv("../search_space_epsilon.csv")
        SS0 = self.SS.dropna(subset=['target_epsilon3'])
        SS0.to_csv("../search_space_epsilon_drop.csv")
    
    def init(
        self,
        params: Dict[str, Any],
        distributions: Optional[List[Literal["Gamma", "Exponential"]]] = None,
        dimension: int = 4,
        epochs: Optional[float] = 3,
        steps: Optional[int] = None,
        sample_rate: float = 0.01,
        target_delta: Optional[float] = None,
        alphas: Optional[float] = tuple(range(2, 256)),
        L2_sensitivity: Optional[float] = None,
        sample_size: Optional[int] = None,
        debug_mode: Optional[bool] = False,
    ):
        self.noise_type = "p"  # plrv
        self.debug_mode = debug_mode
        
        assert distributions != None and len(distributions) != 0
        if ["Gamma", "Exponential"] == distributions:
            assert "a_G" in params and "a_E" in params
        elif ["Gamma"] == distributions:
            params["a_G"] = 1
        elif ["Exponential"] == distributions:
            params["a_E"] = 1
        
        self.params = params
        self.params["distributions"] = distributions
        self.distributions = distributions
        self.dimension = dimension
        
        assert epochs != None or steps != None
        assert 0 < sample_rate <= 1
        self.epochs = epochs if steps is None else steps * sample_rate
        self.steps = math.ceil(epochs / sample_rate) if steps is None else steps
        self.sample_rate = sample_rate
        
        self.target_delta = 1/(2*sample_size) if target_delta is None else target_delta
        self.alphas = alphas
        self.L2_sensitivity = L2_sensitivity
        
        if debug_mode:
            print("\n", "*"*30, "PLRV noise", "*"*30, "\n")
            print(f"PLRV params:\t{self.params}")
            print(f"Distributions:\t{self.distributions}")
            print(f"Steps:\t\t{self.steps}")
            print(f"Sample rate:\t{self.sample_rate}")
            print(f"Target Detla:\t{self.target_delta}")
            print(f"L2_sensitivity:\t{self.L2_sensitivity}")
            
            self.noise = self.generate()
            if self.dimension > 5:
                print(f"Mean of PLRV:\t{np.mean(np.abs(self.noise))}")
                print(f"PLRV Noises:\t{self.noise[:5]}")
            else:
                print(f"PLRV Noises:\t{self.noise}")
            
        manager = accounting_manager.RDPManager(alphas=self.alphas, L2_sensitivity=self.L2_sensitivity)
        # self.target_epsilon = manager._compute_epsilon_from_distribution(
        #     params=self.params,
        #     sample_rate=self.sample_rate,
        #     target_delta=self.target_delta,
        #     steps=self.steps,
        # )
        # self.target_epsilon2 = manager._compute_epsilon_from_distribution2(
        #     params=self.params,
        #     sample_rate=self.sample_rate,
        #     target_delta=self.target_delta,
        #     steps=self.steps,
        # )
        self.target_epsilon3 = manager._compute_epsilon_from_distribution3(
            params=self.params,
            sample_rate=self.sample_rate,
            target_delta=self.target_delta,
            steps=self.steps,
        )
        
        if debug_mode:
            # print(f"Epsilon:\t{self.target_epsilon}")
            # print(f"Epsilon2:\t{self.target_epsilon2}")
            print(f"Epsilon3:\t{self.target_epsilon3}")

    def generate(
        self,
        dimension=None
    ) -> np.ndarray:
        assert self.params != None and self.distributions != None
        self.dimension = self.dimension if dimension is None else dimension
        
        us = 0
        if "Gamma" in self.distributions:
            us = us + self.params['a_G']*np.random.gamma(self.params["G_k"], self.params["G_theta"], self.dimension)
        if "Exponential" in self.distributions:
            us = us + self.params['a_E']*np.random.exponential(self.params["E_lambda"], self.dimension)
        
        return np.random.laplace(0, 1/us)

    def pairs(
        self,
        return_noise_type: Optional[List[Literal["g", "l"]]] = ["g", "l"],
        figures: Union[bool, int] = False,
    ):
        assert self.params != None and self.distributions != None
        
        manager = accounting_manager.RDPManager(L2_sensitivity=self.L2_sensitivity)
        # self.sigma = manager.compute_sigma(
        #     target_epsilon=self.target_epsilon,
        #     target_delta=self.target_delta,
        #     sample_rate=self.sample_rate,
        #     steps=self.steps
        # ) * self.L2_sensitivity if "g" in return_noise_type else None

        # self.sigma2 = manager.compute_sigma(
        #     target_epsilon=self.target_epsilon2,
        #     target_delta=self.target_delta,
        #     sample_rate=self.sample_rate,
        #     steps=self.steps
        # ) * self.L2_sensitivity if "g" in return_noise_type else None

        self.sigma3 = manager.compute_sigma(
            target_epsilon=self.target_epsilon3,
            target_delta=self.target_delta,
            sample_rate=self.sample_rate,
            steps=self.steps
        ) * self.L2_sensitivity if "g" in return_noise_type else None

        self.scale = manager.compute_scale(
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            sample_rate=self.sample_rate,
            steps=self.steps
        ) if "l" in return_noise_type else None

        if self.debug_mode:
            # print(f"Paired sigma:\t{self.sigma}")
            # print(f"Paried sigma2:\t{self.sigma2}")
            print(f"Paried sigma3:\t{self.sigma3}")
        
        if figures: # TODO
            dimension = int(figures)
            noise_p = self.generate(dimension=dimension)
            if "g" in return_noise_type: noise_g = np.random.normal(loc=0, scale=self.sigma, size=dimension)
            if "l" in return_noise_type: noise_l = np.random.normal(loc=0, scale=self.scale, size=dimension) # TODO
            
            noise_names = {"p": "List1"}
            df_noises = {noise_names['p']: noise_p}
            if "g" in return_noise_type:
                noise_names.update({"g": "List2"})
                df_noises.update({noise_names['g']: noise_g})
            if "l" in return_noise_type:
                noise_names.update({"l": "List3"})
                df_noises.update({noise_names['l']: noise_l})
            
            plt.tight_layout()
            # plt.savefig("plrv.png")
            # plt.savefig("Gaussian.png")
        
    def generate_config(
        self,
        idx: Union[int, List[int]] = 1,
        plrv_params: dict = {},
        filepath: str = "demo.plrv_config.py"
    ):
        data = copy.deepcopy(self.params)
        data.update({"sample_rate": self.sample_rate})
        data.update({"L2_sensitivity": self.L2_sensitivity})
        data.update({"sigma": self.sigma if self.sigma is not None else "None"})
        data.update({"scale": self.scale if self.scale is not None else "None"})
        datas = {idx: data}

        json_str = json.dumps(datas, indent=4)
        if isinstance(idx, int):
            content = f"""# Generated by noise_search/noise/plrv.py\n{plrv_params} = {json_str}\n"""
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        else:
            return f"""{plrv_params} = {json_str}\n"""

if __name__ == "__main__":
    try:
        import sys; mode=sys.argv[1]
    except:
        mode="PLRVO_DPSGD"
    
    if mode == "Search_Space_Finder":
        plrv(mode=mode)
    elif mode == "Noise_Parameters_Finder":
        plrv(mode=mode)
    else:
        params = {
            "a_G": 1,
            "G_k": 2000,
            "G_theta": 5.002501250625312e-04,
            "distributions": ["Gamma"]
        }
        array_alphas = tuple(range(2, 256))
        # L2_sensitivity = 0.1
        L2_sensitivity = 1
        sample_size = 67349
        # batchsize = 1024
        batchsize = 674
        epochs = 3
        
        # target_delta =  1 / (2 * sample_size)
        target_delta = 9.999999999999999e-06
        sample_rate = batchsize / sample_size
        steps = math.ceil(epochs / sample_rate)

        n_p = plrv(
            mode=mode,
            params=params,
            distributions=params["distributions"],
            dimension=1000,
            steps=steps,
            sample_rate=sample_rate,
            target_delta=target_delta,
            L2_sensitivity=L2_sensitivity,
            debug_mode=True,
            alphas=array_alphas,
        )
        n_p.pairs(return_noise_type=["g"], figures=False)
