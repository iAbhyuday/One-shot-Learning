import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
#import matplotlib.pyplot as plt

# Siamese net with shared weights
class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv_block = nn.Sequential(
        
        nn.Conv2d(1,64,10),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Conv2d(64,128,7),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Conv2d(128,128,4),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Conv2d(128,256,4),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        )
        
        self.fc_block = nn.Sequential(
        
        nn.Linear(256*6*6,4096),
        nn.Sigmoid()
        
        )
        
        self.out = nn.Linear(4096,1)
    
    def forward(self,x1,x2):
        x1 = self.conv_block(x1)
        x1 = x1.view(-1,256*6*6)
        x1 = self.fc_block(x1)
        
        x2 = self.conv_block(x2)
        x2 = x2.view(-1,256*6*6)
        x2 = self.fc_block(x2)
        
        x = torch.abs(x1-x2)
        
        x = F.sigmoid(self.out(x))
        
        return x