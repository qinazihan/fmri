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
dict =  np.load('stats_dict_within_subject.npy',allow_pickle=True).item()
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

for key, df in dict.items():
    current_min = df.min().min()
    current_max = df.max().max()
    overall_min = min(overall_min, current_min)
    overall_max = max(overall_max, current_max)

overall_max = math.ceil(overall_max)
overall_min = math.floor(overall_min)



# get within subject scross movie stats
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


color_define = ['#0077FF',  # Bright Azure Blue
          '#FF6600',  # Vivid Orange
          '#00CC44',  # Vibrant Emerald Green
          '#FF0044',  # Bright Crimson Red
          '#FFD700',  # Golden Yellow
          '#800080',  # Deep Purple
          '#008080',  # Teal
          '#FF00FF']


#%%
for model in model_list:
    plt_idx = 1
    plt.figure(figsize=(30,30))
    for state in brainstates:
        df_list = []
        df = dict_within[f'{model}-{tasks_list[0]}-{state}']
        df = df.add_prefix(f'{tasks_list[0]}_')
        df_list.append(df)
        df = dict_within[f'{model}-{tasks_list[1]}-{state}']
        df = df.add_prefix(f'{tasks_list[1]}_')
        df_list.append(df)
        df = dict_cross[f'{model}-train{tasks_list[0]}-test{tasks_list[1]}-{state}']
        df = df.add_prefix(f'{tasks_list[0]}to{tasks_list[1]}_')
        df_list.append(df)
        df = dict_cross[f'{model}-train{tasks_list[1]}-test{tasks_list[0]}-{state}']
        df = df.add_prefix(f'{tasks_list[1]}to{tasks_list[0]}_')
        df_list.append(df)


        df = df_dme[df_dme.index.str.contains(model) & df_dme.index.str.contains(state)]
        df = df.add_prefix(f'dme_cross_movie_')
        df_list.append(df)
        df = df_dmh[df_dmh.index.str.contains(model) & df_dmh.index.str.contains(state)]
        df = df.add_prefix(f'dmh_cross_movie_')
        df_list.append(df)

        CORR_Df = pd.concat(df_list, axis = 1)
        colors = color_define * 3
        plt.subplot(4,2,plt_idx)
        sns.stripplot(data=CORR_Df, jitter=False, palette=colors)
        x_ticks = plt.xticks()[0]  # Get the positions of the x-ticks
        for i in range(len(x_ticks) - 1):
            if i in range(8,16):  # Grey for even indices
                plt.gca().add_patch(plt.Rectangle((i - 0.5, -3), 1, 6, color='grey', alpha=0.1))
        i = 0
        plt.plot([i,i+1],[CORR_Df.loc[:,f"{tasks_list[0]}_R2_Score"],CORR_Df.loc[:,f"{tasks_list[0]}_R2_Score_Time"]],color='blue', linestyle='-', marker='o',alpha = 0.2)
        plt.plot([i+2,i+3],[CORR_Df.loc[:,f"{tasks_list[0]}_Corr"],CORR_Df.loc[:,f"{tasks_list[0]}_Corr_Time"]], color='blue', linestyle='-', marker='o',alpha = 0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"{tasks_list[1]}_R2_Score"], CORR_Df.loc[:, f"{tasks_list[1]}_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"{tasks_list[1]}_Corr"], CORR_Df.loc[:, f"{tasks_list[1]}_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)
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
                 [CORR_Df.loc[:, f"dme_cross_movie_R2_Score"], CORR_Df.loc[:, f"dme_cross_movie_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"dme_cross_movie_Corr"], CORR_Df.loc[:, f"dme_cross_movie_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"dmh_cross_movie_R2_Score"], CORR_Df.loc[:, f"dmh_cross_movie_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"dmh_cross_movie_Corr"], CORR_Df.loc[:, f"dmh_cross_movie_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)

        plt.ylim([overall_min,overall_max])
        plt.yticks(np.arange(overall_min, overall_max + 0.2, 0.2))
        plt.title(f"{state}")
        plt.xticks(rotation=90)
        plt.xticks(fontsize=16)
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
        df = dict_within[f'{model}-{tasks_list[1]}-{state}']
        df = df.add_prefix(f'{tasks_list[1]}_')
        df_list.append(df)
        df = dict_cross[f'{model}-train{tasks_list[0]}-test{tasks_list[1]}-{state}']
        df = df.add_prefix(f'{tasks_list[0]}to{tasks_list[1]}_')
        df_list.append(df)
        df = dict_cross[f'{model}-train{tasks_list[1]}-test{tasks_list[0]}-{state}']
        df = df.add_prefix(f'{tasks_list[1]}to{tasks_list[0]}_')
        df_list.append(df)


        df = df_dme[df_dme.index.str.contains(model) & df_dme.index.str.contains(state)]
        df = df.add_prefix(f'dme_cross_movie_')
        df_list.append(df)
        df = df_dmh[df_dmh.index.str.contains(model) & df_dmh.index.str.contains(state)]
        df = df.add_prefix(f'dmh_cross_movie_')
        df_list.append(df)

        CORR_Df = pd.concat(df_list, axis = 1)
        colors = color_define * 3
        plt.subplot(1,3,plt_idx)
        sns.stripplot(data=CORR_Df, jitter=False, palette=colors)
        x_ticks = plt.xticks()[0]  # Get the positions of the x-ticks
        for i in range(len(x_ticks) - 1):
            if i in range(8,16):  # Grey for even indices
                plt.gca().add_patch(plt.Rectangle((i - 0.5, -3), 1, 6, color='grey', alpha=0.1))

        i = 0
        plt.plot([i,i+1],[CORR_Df.loc[:,f"{tasks_list[0]}_R2_Score"],CORR_Df.loc[:,f"{tasks_list[0]}_R2_Score_Time"]],color='blue', linestyle='-', marker='o',alpha = 0.2)
        plt.plot([i+2,i+3],[CORR_Df.loc[:,f"{tasks_list[0]}_Corr"],CORR_Df.loc[:,f"{tasks_list[0]}_Corr_Time"]], color='blue', linestyle='-', marker='o',alpha = 0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"{tasks_list[1]}_R2_Score"], CORR_Df.loc[:, f"{tasks_list[1]}_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"{tasks_list[1]}_Corr"], CORR_Df.loc[:, f"{tasks_list[1]}_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)
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
                 [CORR_Df.loc[:, f"dme_cross_movie_R2_Score"], CORR_Df.loc[:, f"dme_cross_movie_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"dme_cross_movie_Corr"], CORR_Df.loc[:, f"dme_cross_movie_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)
        i += 4
        plt.plot([i, i + 1],
                 [CORR_Df.loc[:, f"dmh_cross_movie_R2_Score"], CORR_Df.loc[:, f"dmh_cross_movie_R2_Score_Time"]],
                 color='blue', linestyle='-', marker='o', alpha=0.2)
        plt.plot([i + 2, i + 3],
                 [CORR_Df.loc[:, f"dmh_cross_movie_Corr"], CORR_Df.loc[:, f"dmh_cross_movie_Corr_Time"]], color='blue',
                 linestyle='-', marker='o', alpha=0.2)

        plt.ylim([overall_min,overall_max])
        plt.yticks(np.arange(overall_min, overall_max + 0.2, 0.2))
        plt.title(f"{model}")
        plt.xticks(rotation=90)
        plt.xticks(fontsize=16)
        plt_idx += 1

    plt.suptitle(f'{state}')
    plt.tight_layout()
    plt.show()