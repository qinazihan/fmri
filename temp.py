#%%
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from treeple.ensemble import PatchObliqueRandomForestRegressor
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import time
from library.time_func import *
import math

dict =  np.load('stats_dict_within_subject.npy',allow_pickle=True).item()
brainstates = [ 'tsCAP1', 'tsCAP2', 'tsCAP3', 'tsCAP4', 'tsCAP5', 'tsCAP6', 'tsCAP7', 'tsCAP8']
model_list = ['ridge','rf','morf']
SUBJECTS = [str(i + 1).zfill(2) for i in range(22)]
 #%%
overall_min = float('inf')
overall_max = float('-inf')

for key, df in dict.items():
    current_min = df.min().min()
    current_max = df.max().max()
    overall_min = min(overall_min, current_min)
    overall_max = max(overall_max, current_max)

overall_max = math.ceil(overall_max)
overall_min = math.floor(overall_min)

#%%
dme_list = []
dme_index = []
dmh_list = []
dmh_index = []
for model in model_list:
    for subject in SUBJECTS:
        for state in brainstates:
            key = f'{model}-Subject{subject}-{state}'
            if 'dme' in dict[key].index:
                dme_list.append(dict[key].loc['dme'])
                dme_index.append(f'{model}-{subject}-{state}')
            if 'dmh' in dict[key].index:
                dmh_list.append(dict[key].loc['dmh'])
                dmh_index.append(f'{model}-{subject}-{state}')
df_dme = pd.DataFrame(dme_list,index = dme_index)
df_dmh = pd.DataFrame(dmh_list,index = dmh_index)
a = 0


