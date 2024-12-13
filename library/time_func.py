# Time Encoding
import numpy as np
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.freqs = (2 * np.pi) / (torch.arange(2, dim + 1, 2))
        self.freqs = self.freqs.unsqueeze(0)

    def forward(self, t):
        self.sin = torch.sin(self.freqs * t)
        self.cos = torch.cos(self.freqs * t)
        return torch.cat([self.sin, self.cos], dim=-1)

def Time_Handeler(t,tdim= 50):
    time_embed = TimeEmbedding(dim=tdim)
    timeembed = []
    for i in range(len(t)):
        timeembed.append(time_embed(t[i]))
    return np.concatenate(timeembed).reshape(-1,tdim)
