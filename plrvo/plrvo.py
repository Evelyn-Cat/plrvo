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

# from plrvo_transformers.accountant import accounting_manager
# from plrvo_transformers.accountant import rdp_accounting

import sys;sys.path.append("..")
from private_transformers.accountant import accounting_manager
from private_transformers.accountant import rdp_accounting


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
        return_noise_type: Optional[List[Literal["g", "l"]]] = ["g", "l"]
    ):
        assert self.params != None and self.distributions != None
        
        manager = accounting_manager.RDPManager(L2_sensitivity=self.L2_sensitivity)
        self.sigma = manager.compute_sigma(
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            sample_rate=self.sample_rate,
            steps=self.steps
        ) * self.L2_sensitivity

        if self.debug_mode:
            print(f"Paried sigma3:\t{self.sigma3}")
    
    def accountant(
        self,
    ):  
        # TODO Nicholas
        pass


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
        # n_p.pairs(return_noise_type=["g"], figures=False)
        epsilon = n_p.accountant()
