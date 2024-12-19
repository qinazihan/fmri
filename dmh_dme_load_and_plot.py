from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import time
from library.time_func import *
import math
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

dict_within =  np.load('stats_dict_dmh_dme_within_movie.npy',allow_pickle=True).item()
dict_cross = np.load('stats_dict_dmh_dme_cross.npy',allow_pickle=True).item()
tasks_list = ['dme_run', 'dmh_run']
brainstates = [ 'tsCAP1', 'tsCAP2', 'tsCAP3', 'tsCAP4', 'tsCAP5', 'tsCAP6', 'tsCAP7', 'tsCAP8']
model_list = ['ridge','rf','morf']
SUBJECTS = [str(i + 1).zfill(2) for i in range(22)]
#%%
# get ploting min/max
overall_min = float('inf')
overall_max = float('-inf')

for key, df in dict_within.items():
    current_min = df.min().min()
    current_max = df.max().max()
    overall_min = min(overall_min, current_min)
    overall_max = max(overall_max, current_max)

for key, df in dict_cross.items():
    current_min = df.min().min()
    current_max = df.max().max()
    overall_min = min(overall_min, current_min)
    overall_max = max(overall_max, current_max)

overall_max = math.ceil(overall_max)
overall_min = math.floor(overall_min)

#%%
for model in model_list:
    plt_idx = 1
    plt.figure(figsize=(30,30))
    for state in brainstates:
        df_list = []
        df = dict_within[f'{model}-{tasks_list[0]}-{state}']
        df = df.add_prefix(f'{tasks_list[0]}_')
        df_list.append(df)
        df = dict_cross[f'{model}-train{tasks_list[0]}-test{tasks_list[1]}-{state}']
        df = df.add_prefix(f'{tasks_list[0]}to{tasks_list[1]}_')
        df_list.append(df)
        df = dict_cross[f'{model}-train{tasks_list[1]}-test{tasks_list[0]}-{state}']
        df = df.add_prefix(f'{tasks_list[1]}to{tasks_list[0]}_')
        df_list.append(df)
        df = dict_within[f'{model}-{tasks_list[1]}-{state}']
        df = df.add_prefix(f'{tasks_list[1]}_')
        df_list.append(df)

        CORR_Df = pd.concat(df_list, axis = 1)
        colors = ['blue', 'orange', 'green', 'red'] * 4
        plt.subplot(4,2,plt_idx)
        sns.stripplot(data=CORR_Df, jitter=False, palette=colors)
        i = 0
        plt.plot([i,i+1],[CORR_Df.loc[:,f"{tasks_list[0]}_R2_Score"],CORR_Df.loc[:,f"{tasks_list[0]}_R2_Score_Time"]],color='blue', linestyle='-', marker='o',alpha = 0.2)
        plt.plot([i+2,i+3],[CORR_Df.loc[:,f"{tasks_list[0]}_Corr"],CORR_Df.loc[:,f"{tasks_list[0]}_Corr_Time"]], color='blue', linestyle='-', marker='o',alpha = 0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"{tasks_list[0]}to{tasks_list[1]}_R2_Score"], CORR_Df.loc[:, f"{tasks_list[0]}to{tasks_list[1]}_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"{tasks_list[0]}to{tasks_list[1]}_Corr"], CORR_Df.loc[:, f"{tasks_list[0]}to{tasks_list[1]}_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"{tasks_list[1]}to{tasks_list[0]}_R2_Score"], CORR_Df.loc[:, f"{tasks_list[1]}to{tasks_list[0]}_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"{tasks_list[1]}to{tasks_list[0]}_Corr"], CORR_Df.loc[:, f"{tasks_list[1]}to{tasks_list[0]}_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"{tasks_list[1]}_R2_Score"], CORR_Df.loc[:, f"{tasks_list[1]}_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"{tasks_list[1]}_Corr"], CORR_Df.loc[:, f"{tasks_list[1]}_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)

        plt.ylim([overall_min,overall_max])
        plt.yticks(np.arange(overall_min, overall_max + 0.2, 0.2))
        plt.title(f"{state}")
        plt.xticks(rotation=90)
        plt_idx += 1

    plt.suptitle(f'Model {model}')
    plt.tight_layout()
    plt.show()

#%%
for state in brainstates:
    plt_idx = 1
    plt.figure(figsize=(30,10))
    for model in model_list:
        df_list = []
        df = dict_within[f'{model}-{tasks_list[0]}-{state}']
        df = df.add_prefix(f'{tasks_list[0]}_')
        df_list.append(df)
        df = dict_cross[f'{model}-train{tasks_list[0]}-test{tasks_list[1]}-{state}']
        df = df.add_prefix(f'{tasks_list[0]}to{tasks_list[1]}_')
        df_list.append(df)
        df = dict_cross[f'{model}-train{tasks_list[1]}-test{tasks_list[0]}-{state}']
        df = df.add_prefix(f'{tasks_list[1]}to{tasks_list[0]}_')
        df_list.append(df)
        df = dict_within[f'{model}-{tasks_list[1]}-{state}']
        df = df.add_prefix(f'{tasks_list[1]}_')
        df_list.append(df)

        CORR_Df = pd.concat(df_list, axis = 1)
        colors = ['blue', 'orange', 'green', 'red'] * 4
        plt.subplot(1,3,plt_idx)
        sns.stripplot(data=CORR_Df, jitter=False, palette=colors)
        i = 0
        plt.plot([i,i+1],[CORR_Df.loc[:,f"{tasks_list[0]}_R2_Score"],CORR_Df.loc[:,f"{tasks_list[0]}_R2_Score_Time"]],color='blue', linestyle='-', marker='o',alpha = 0.2)
        plt.plot([i+2,i+3],[CORR_Df.loc[:,f"{tasks_list[0]}_Corr"],CORR_Df.loc[:,f"{tasks_list[0]}_Corr_Time"]], color='blue', linestyle='-', marker='o',alpha = 0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"{tasks_list[0]}to{tasks_list[1]}_R2_Score"], CORR_Df.loc[:, f"{tasks_list[0]}to{tasks_list[1]}_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"{tasks_list[0]}to{tasks_list[1]}_Corr"], CORR_Df.loc[:, f"{tasks_list[0]}to{tasks_list[1]}_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"{tasks_list[1]}to{tasks_list[0]}_R2_Score"], CORR_Df.loc[:, f"{tasks_list[1]}to{tasks_list[0]}_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"{tasks_list[1]}to{tasks_list[0]}_Corr"], CORR_Df.loc[:, f"{tasks_list[1]}to{tasks_list[0]}_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"{tasks_list[1]}_R2_Score"], CORR_Df.loc[:, f"{tasks_list[1]}_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"{tasks_list[1]}_Corr"], CORR_Df.loc[:, f"{tasks_list[1]}_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)

        plt.ylim([overall_min,overall_max])
        plt.yticks(np.arange(overall_min, overall_max + 0.2, 0.2))
        plt.title(f"{model}")
        plt.xticks(rotation=90)
        plt_idx += 1

    plt.suptitle(f'{state}')
    plt.tight_layout()
    plt.show()