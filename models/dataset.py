import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    
    def __init__(self, args, train_seq):
        self.args = args
        self.train_seq = train_seq.tolist()                #* start from 0
        self.max_len = args.max_len


    def __len__(self):
        return len(self.train_seq)

    def __getitem__(self, index):
        sequence = self.train_seq[index] 
        cur_tensors = torch.tensor(sequence, dtype=torch.long)+1
                    
        return cur_tensors