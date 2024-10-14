import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
import os
from geopy import distance
from scipy.sparse import dok_matrix
import scipy.sparse as sp
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, trajs):
        self.trajs = trajs
        
    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self, index):
        traj = self.trajs[index]
        return index, traj

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def dataloader(path, batch_size, device):
    train_trajs = np.load(path+'/train.npy', allow_pickle=True).astype(np.int32)
    test_trajs = np.load(path+'/test.npy', allow_pickle=True).astype(np.int32)
    trainData = DataLoader(CustomDataset(train_trajs), batch_size=batch_size, shuffle=True)
    testData = DataLoader(CustomDataset(test_trajs), batch_size=batch_size, shuffle=False)

    return trainData, train_trajs, testData, test_trajs
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def construct_segment_graph(edges):
    segment_graph = np.zeros((len(edges)+1, len(edges)+1), dtype=np.float32) # for start token 0
    for edge in edges.itertuples():
        source, target = edge.u, edge.v    
        edge_id = edge.fid + 1
        neigh_edges = edges.loc[edges.u == target].fid + 1
        for n_edge in neigh_edges:
            segment_graph[edge_id][n_edge] = 1

    # self.segment_graph[0, 1:] = 1
    for i in range(1, segment_graph.shape[0]):
        if segment_graph[i].sum():
            segment_graph[0, i] = 1       
            
    return segment_graph             
    

