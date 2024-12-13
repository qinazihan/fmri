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
# load data
print(os.getcwd())
data = np.load('group_data_natview_data_fmri_eyetracking1hz.npy',allow_pickle=True)
data_dict = data.item()
#%%
subjects = data_dict['subjects']
sessions = data_dict['sessions']
tasks = data_dict['tasks']
data_brainstates = data_dict['data_fmri']
data_eyetracking = data_dict['data_eyetracking']
brainstates = [ 'tsCAP1', 'tsCAP2', 'tsCAP3', 'tsCAP4', 'tsCAP5', 'tsCAP6', 'tsCAP7', 'tsCAP8']

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
def Train_Test(test_task_list,all_task_list,unique_list,state, time = False,model = 'ridge'):
    PUPIL_ALL = []
    FMRI_ALL = []
    TIME_ALL = []
    PUPIL_TEST = []
    FMRI_TEST = []
    TIME_TEST = []
    for i in range(len(all_task_list)):
        key = all_task_list[i]
        PUPIL, FMRI, TIME = Data_Handeler(key, state, window=12, mask_thr=0.75)

        if key in test_task_list:
            PUPIL_TEST.append(PUPIL)
            FMRI_TEST.extend(FMRI)
            TIME_TEST.extend(TIME)
        else:
            PUPIL_ALL.append(PUPIL)
            FMRI_ALL.extend(FMRI)
            TIME_ALL.extend(TIME)

    PUPIL_TRAIN = zscore(np.concatenate(PUPIL_ALL),axis = 1)
    FMRI_TRAIN = zscore(FMRI_ALL)
    TIME_TRAIN = TIME_ALL
    PUPIL_TEST = zscore(np.concatenate(PUPIL_TEST),axis = 1)
    FMRI_TEST = zscore(FMRI_TEST)
    TIME_TEST = TIME_TEST

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
    stats_dict = np.load('stats_dict_within_subject.npy',allow_pickle=True).item()
except:
    stats_dict = {}
# tasks_list = ['rest_run','monkey1_run','checker_run','inscapes_run']
model_list = ['ridge','rf','morf']
SUBJECTS = [str(i + 1).zfill(2) for i in range(22)]
available_subjects_list_all_task = []

# collecting runs for each subject
matching_cur_task_list = []
unique_run = []
unique_run_ct = []
for i in range(len(SUBJECTS)):
    subject  = SUBJECTS[i]
    matching_cur_task_list.append([key for key in data_brainstates.keys() if 'sub-'+ subject in key])
    unique_run.append(list(set([item.split('task-')[1].rsplit('_', 1)[0] for item in matching_cur_task_list[i]])))
    unique_run_ct.append(len(matching_cur_task_list[i]))


#%%
for i in range(len(SUBJECTS)):
    subject = SUBJECTS[i]
    subject_task_list = matching_cur_task_list[i]
    start_time = time.time()
    for state in brainstates:
        for model in model_list:
            CORR = []
            CORR_TIME = []
            R2_score = []
            R2_score_TIME = []
            for task in unique_run[i]:
                current_task_list = [i for i in subject_task_list if task in i]
                print(f"Subject: {subject}, Test Task: {task}, Model: {model}")
                _,_,FMRI_TEST,ypred,corr = Train_Test(current_task_list,subject_task_list, unique_run[i],state,time = False,model = model)
                R2_score.append(r2_score(FMRI_TEST,ypred))
                CORR.append(corr)
                _,_,FMRI_TEST_time,ypred_time,corr_time = Train_Test(current_task_list,subject_task_list, unique_run[i],state,time = True,model = model)
                CORR_TIME.append(corr_time)
                R2_score_TIME.append(r2_score(FMRI_TEST_time,ypred_time))

            CORR_Df = pd.DataFrame(np.hstack((np.array(R2_score).reshape(-1,1),np.array(R2_score_TIME).reshape(-1,1),np.array(CORR).reshape(-1,1),np.array(CORR_TIME).reshape(-1,1))),columns = ['R2_Score','R2_Score_Time','Corr','Corr_Time'],index = unique_run[i])
            sns.stripplot(CORR_Df,s = 8,alpha = 0.6,jitter = False)
            for idx, row in CORR_Df.iterrows():
                plt.plot([2, 3], [row['Corr'], row['Corr_Time']], color='blue', linestyle='-', marker='o',alpha = 0.2)
            for idx, row in CORR_Df.iterrows():
                plt.plot([0, 1], [row['R2_Score'], row['R2_Score_Time']], color='blue', linestyle='-', marker='o',alpha = 0.2)
            plt.xticks([0, 1,2,3],  ['R2_Score','R2_Score_Time','Corr','Corr_Time'])
            plt.ylim([-0.4,1])
            plt.yticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
            plt.title(f'{model} Correlation - Subject: {subject} - {state}')
            plt.show()
            stats_dict[f'{model}-Subject{subject}-{state}'] = CORR_Df
            np.save('stats_dict_within_subject.npy',stats_dict, allow_pickle=True)
        print(f"Runtime for one state with all 3 models: {time.time() - start_time}s")