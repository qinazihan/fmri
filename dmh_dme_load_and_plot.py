from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import time
from library.time_func import *
import math

dict =  np.load('stats_dict_dmh_dme_within_movie.npy',allow_pickle=True).item()
dict_cross = np.load('stats_dict_dmh_dme_cross.npy',allow_pickle=True).item()

brainstates = [ 'tsCAP1', 'tsCAP2', 'tsCAP3', 'tsCAP4', 'tsCAP5', 'tsCAP6', 'tsCAP7', 'tsCAP8']