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
for subject in SUBJECTS:

    plt.figure(figsize=(30,15))
    plt_idx = 1
    for state in brainstates:
        df_list = []
        for model in model_list:
            df = dict[f'{model}-Subject{subject}-{state}']
            df = df.add_prefix(f'{model}_')
            df_list.append(df)

        CORR_Df = pd.concat(df_list, axis = 1)
        colors = ['blue', 'orange', 'green', 'red'] * 3
        plt.subplot(2,4,plt_idx)
        sns.stripplot(data=CORR_Df, jitter=False, palette=colors)
        i = 0
        for model in model_list:
            plt.plot([i,i+1],[CORR_Df.loc[:,f"{model}_R2_Score"],CORR_Df.loc[:,f"{model}_R2_Score_Time"]],color='blue', linestyle='-', marker='o',alpha = 0.2)
            plt.plot([i+2,i+3],[CORR_Df.loc[:,f"{model}_Corr"],CORR_Df.loc[:,f"{model}_Corr_Time"]], color='blue', linestyle='-', marker='o',alpha = 0.2)
            i += 4

        suffixes = ['R2_Score','R2_Score_Time','Corr','Corr_Time']
        label = [f"{model}_{suffix}" for model in model_list for suffix in suffixes]
        plt.xticks(list(range(0, 12, 1)),  label)
        plt.ylim([overall_min,overall_max])
        plt.yticks(np.arange(overall_min, overall_max + 0.2, 0.2))
        plt.title(f"{state}")
        plt.xticks(rotation=90)
        plt_idx += 1
    plt.suptitle(f'Subject {subject}')
    plt.tight_layout()
    plt.show()

#%%
for state in brainstates:
    plt.figure(figsize=(40,10))
    plt_idx = 1
    for subject in SUBJECTS:
        df_list = []
        for model in model_list:
            df = dict[f'{model}-Subject{subject}-{state}']
            df = df.add_prefix(f'{model}_')
            df_list.append(df)

        CORR_Df = pd.concat(df_list, axis = 1)
        colors = ['blue', 'orange', 'green', 'red'] * 3
        plt.subplot(2,11,plt_idx)
        sns.stripplot(data=CORR_Df, jitter=False, palette=colors)
        i = 0
        for model in model_list:
            plt.plot([i,i+1],[CORR_Df.loc[:,f"{model}_R2_Score"],CORR_Df.loc[:,f"{model}_R2_Score_Time"]],color='blue', linestyle='-', marker='o',alpha = 0.2)
            plt.plot([i+2,i+3],[CORR_Df.loc[:,f"{model}_Corr"],CORR_Df.loc[:,f"{model}_Corr_Time"]], color='blue', linestyle='-', marker='o',alpha = 0.2)
            i += 4

        suffixes = ['R2_Score','R2_Score_Time','Corr','Corr_Time']
        label = [f"{model}_{suffix}" for model in model_list for suffix in suffixes]
        plt.xticks(list(range(0, 12, 1)),  label)
        plt.ylim([overall_min,overall_max])
        plt.yticks(np.arange(overall_min, overall_max + 0.2, 0.2))
        plt.title(f'Subject {subject}')
        plt.xticks(rotation=90)
        plt_idx += 1
    plt.suptitle(f"{state}")
    plt.tight_layout()
    plt.show()