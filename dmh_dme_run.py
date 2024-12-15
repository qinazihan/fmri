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


#%%
print(os.getcwd())
data = np.load('group_data_natview_data_fmri_eyetracking1hz.npy',allow_pickle=True)
data_dict = data.item()
#%%
subjects = data_dict['subjects']
sessions = data_dict['sessions']
tasks = data_dict['tasks']
data_brainstates = data_dict['data_fmri']
data_eyetracking = data_dict['data_eyetracking']
brainstates = data_brainstates['sub-01_ses-01_task-checker_run-01'].columns[1:-2].tolist()
brainstates = [item for item in brainstates if item != 'tsMask']

#%%
# extract data for one session (key) for one brain state
def Data_Handeler(key,brain_state,window=12,mask_thr = 0.75):
    fmri_ts_mask = (data_brainstates[key]['tsMask'].to_numpy() > 0.5)
    eyetrack_mask = (data_eyetracking[key]['tmask'].to_numpy() > 0.5)

    FMRI_TARGET_DATA = data_brainstates[key][brain_state].to_numpy()
    time_fmri = data_brainstates[key]['time'].to_numpy()
    PD_DATA = data_eyetracking[key]['pupil_size'].to_numpy()
    Xpos = data_eyetracking[key]['X_position'].to_numpy()
    Ypos = data_eyetracking[key]['Y_position'].to_numpy()

    PD_DATA_dfirst = np.diff(PD_DATA, prepend=PD_DATA[0])
    PD_DATA_dsecond = np.diff(PD_DATA, n=2, prepend=PD_DATA_dfirst[:2])
    PREDICTION_FEATURES = np.vstack((PD_DATA, PD_DATA_dfirst, PD_DATA_dsecond)).T

    PUPIL = []
    FMRI = []
    TIME = []
    for i in range(0,FMRI_TARGET_DATA.shape[0]-window):
        X = np.reshape(PREDICTION_FEATURES[(i):(i+window),: ].flatten(),(1,-1))
        FEATURE_MASK = eyetrack_mask[(i):(i+window)]
        Y = FMRI_TARGET_DATA[i+window]
        TARGET_MASK = fmri_ts_mask[i+window]
        if np.mean(FEATURE_MASK) >= 0.75 and TARGET_MASK:
            PUPIL.append(X)
            FMRI.append(Y)
            TIME.append(i + window)
            # TIME.append(time_fmri[i+window])
    PUPIL_DF = np.concatenate(PUPIL)
    return PUPIL_DF,FMRI,TIME

#%%
def Train_Test(test_subject_idx,available_list,state, time = False,model = 'ridge'):
    PUPIL_ALL = []
    FMRI_ALL = []
    TIME_ALL = []
    for i in range(len(available_list)):
        id_list = available_list[i]
        sub_id = id_list[0]
        cur_key_list = id_list[1]
        PUPIL = []
        FMRI = []
        TIME = []

        #extract data for one subject across all available session
        for key in cur_key_list:
            PUPIL_curkey, FMRI_curkey, TIME_curkey = Data_Handeler(key, state, window=12, mask_thr=0.75)
            PUPIL.append(PUPIL_curkey)
            FMRI.extend(FMRI_curkey)
            TIME.extend(TIME_curkey)
        PUPIL = np.concatenate(PUPIL)

        if i == test_subject_idx:
            PUPIL_TEST = zscore(PUPIL,axis = 1)
            FMRI_TEST = zscore(FMRI)
            TIME_TEST = TIME
        else:
            PUPIL_ALL.append(PUPIL)
            FMRI_ALL.extend(FMRI)
            TIME_ALL.extend(TIME)

    PUPIL_TRAIN = zscore(np.concatenate(PUPIL_ALL),axis = 1)
    FMRI_TRAIN = zscore(FMRI_ALL)
    TIME_TRAIN = TIME_ALL
    if model == 'ridge':
        reg = Ridge(alpha=1)
    if model == 'rf':
        reg = RandomForestRegressor()
    if model == 'morf':
        reg = PatchObliqueRandomForestRegressor()
    if time:
        TIME_ENCODE_TRAIN = Time_Handeler(TIME_TRAIN)
        PUPIL_TRAIN_TIME = np.hstack((PUPIL_TRAIN,TIME_ENCODE_TRAIN))
        TIME_ENCODE_TEST = Time_Handeler(TIME_TEST)
        PUPIL_TEST_TIME = np.hstack((PUPIL_TEST,TIME_ENCODE_TEST))
    else:
        PUPIL_TRAIN_TIME = PUPIL_TRAIN
        PUPIL_TEST_TIME = PUPIL_TEST

    reg.fit(PUPIL_TRAIN_TIME, FMRI_TRAIN)
    ypred_train = reg.predict(PUPIL_TRAIN_TIME)
    ypred = reg.predict(PUPIL_TEST_TIME)
    corr = np.corrcoef(np.array(FMRI_TEST).T,ypred.T)[0,1]
    return FMRI_TRAIN,ypred_train,FMRI_TEST,ypred,corr


#%%
try:
    stats_dict = np.load('stats_dict_dmh_dme_within_movie.npy',allow_pickle=True).item()
except:
    stats_dict = {}
tasks_list = ['dme_run','dmh_run']
model_list = ['ridge','rf','morf']
SUBJECTS = [str(i + 1).zfill(2) for i in range(22)]
available_subjects_list_all_task = []
for task in tasks_list:
    matching_cur_task_list = [key for key in data_brainstates.keys() if task in key]
    available_subjects_list = []
    for subject in SUBJECTS:
        list = [task_name for task_name in matching_cur_task_list if f'sub-{subject}' in task_name]
        if list != []:
            available_subjects_list.append([subject, list])

    available_subjects_list_all_task.append(available_subjects_list)



#%%
for task_idx in range(len(tasks_list)):
    start_time = time.time()
    cur_task = tasks_list[task_idx]
    cur_task_available_list = available_subjects_list_all_task[task_idx]
    for state in brainstates:
        # TODO temp continue run setting on Dec 13
        if cur_task == 'dme_run' and 'ts' in state or state in ['yeo7net1','yeo7net2']:
            continue
        for model in model_list:
            if (
                    cur_task == 'dme_run' and
                    (
                            'ts' in state or
                            'yeo7' in state or
                            state == 'yeo17net1'
                    )
            ):
                continue
            CORR = []
            CORR_TIME = []
            R2_score = []
            R2_score_TIME = []
            sub_idx_list = []
            for i in range(len(cur_task_available_list)):
                test_sub = cur_task_available_list[i][0]
                sub_idx_list.append(test_sub)
                print(cur_task,model,state,test_sub)
                _,_,FMRI_TEST,ypred,corr = Train_Test(i,cur_task_available_list,state,time = False,model = model)
                R2_score.append(r2_score(FMRI_TEST,ypred))
                CORR.append(corr)
                _,_,FMRI_TEST_time,ypred_time,corr_time = Train_Test(i,cur_task_available_list,state,time = True,model = model)
                CORR_TIME.append(corr_time)
                R2_score_TIME.append(r2_score(FMRI_TEST_time,ypred_time))


            CORR_Df = pd.DataFrame(np.hstack((np.array(R2_score).reshape(-1,1),np.array(R2_score_TIME).reshape(-1,1),np.array(CORR).reshape(-1,1),np.array(CORR_TIME).reshape(-1,1))),columns = ['R2_Score','R2_Score_Time','Corr','Corr_Time'],index = sub_idx_list)
            sns.stripplot(CORR_Df,s = 8,alpha = 0.6,jitter = False)
            for idx, row in CORR_Df.iterrows():
                plt.plot([2, 3], [row['Corr'], row['Corr_Time']], color='blue', linestyle='-', marker='o',alpha = 0.2)
            for idx, row in CORR_Df.iterrows():
                plt.plot([0, 1], [row['R2_Score'], row['R2_Score_Time']], color='blue', linestyle='-', marker='o',alpha = 0.2)
            plt.xticks([0, 1,2,3],  ['R2_Score','R2_Score_Time','Corr','Corr_Time'])
            plt.ylim([-0.4,1])
            plt.yticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
            plt.title(f'{model} Correlation {cur_task} {state}')
            plt.show()
            stats_dict[f'{model}-{cur_task}-{state}'] = CORR_Df
            np.save('stats_dict_dmh_dme_within_movie.npy',stats_dict)
    print('time elapsed',time.time() - start_time)