import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np

def one_hot_encode_seq(seq):
    
    ds_out = np.zeros([4,len(seq)])
  
    for i, l in enumerate(seq):
        if (l == 'A'):
            ds_out[0,i] = 1
        if (l == 'G'):
            ds_out[1,i] = 1
        if (l == 'C'):
            ds_out[2,i] = 1
        if (l == 'T'):
            ds_out[3,i] = 1    
            
    return(ds_out)

nfeats = 4
height = 1
nkernels = [320,480,960]
dropouts = [0.2,0.5]

class deep_sea_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=nfeats,      out_channels=nkernels[0], kernel_size=8)
        self.conv2 = nn.Conv1d(in_channels=nkernels[0], out_channels=nkernels[1], kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=nkernels[1], out_channels=nkernels[2], kernel_size=8)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(p=dropouts[0])
        self.drop2 = nn.Dropout(p=dropouts[1])
        self.linear1 = nn.Linear(53*960, 925)
        self.linear2 = nn.Linear(925, 919)
    
    def foward(self, input):
        ## convolution 1 ##
        ds = self.conv1(input)
        ds = F.relu(ds)
        ds = self.maxpool(ds)
        ds = self.drop1(ds)
        
        ## convolution 2 ##
        ds = self.conv2(ds)
        ds = F.relu(ds)
        ds = self.maxpool(ds)
        ds = self.drop1(ds)
        
        ## convolution 3 ##
        ds = self.conv3(ds)
        ds = F.relu(ds)
        ds = self.drop2(ds)
        
        ds = ds.view(-1, 53*960)
        ds = self.linear1(ds)
        ds = F.relu(ds)
        ds = self.linear2(ds)
        
        return ds
        
        
def get_title():
    title = """
    =============================================
    88888                     888888            
    8    8 eeee eeee eeeee    8      eeee eeeee 
    8e   8 8    8    8   8    8eeeee 8    8   8 
    88   8 8eee 8eee 8eee8        88 8eee 8eee8 
    88   8 88   88   88       e   88 88   88  8 
    88eee8 88ee 88ee 88       8eee88 88ee 88  8 
    =============================================
    """
    print(title)