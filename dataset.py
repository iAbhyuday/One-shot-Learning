import torch
from torch.utils.data import DataLoader,Dataset,random_split
import torchvision.transforms as transforms
import numpy as np
import os 
import random
from PIL import Image

class Ominglot(Dataset):
    
    def __init__(self,root,size,transforms=None):
        self.root = root
        self.dir = [(c,os.listdir(self.root+'/'+c)) for c in os.listdir(self.root)]
        self.transforms = transforms
        self.size=size
    def __len__(self):
        return self.size
    def __getitem__(self,idx):
        
        if idx%2:
            alphabet = random.choice(self.dir)
            character = random.choice(alphabet[1])
            character1 = random.choice(os.listdir(self.root+'/'+alphabet[0]+'/'+character))
            character2 = random.choice(os.listdir(self.root+'/'+alphabet[0]+'/'+character))
                                       
            img1 = Image.open(self.root+'/'+alphabet[0]+'/'+character+'/'+character1)
            
            img2 = Image.open(self.root+'/'+alphabet[0]+'/'+character+'/'+character2)
            
            label= 1.
        else:
            
            alphabet1,alphabet2 = random.sample(self.dir,2)
            character1 = random.choice(alphabet1[1])
            
            character2 = random.choice(alphabet2[1])
            while(character1==character2 and alphabet1[0]==alphabet2[0]):
                character1 = random.choice(alphabet1[1])
            
                character2 = random.choice(alphabet2[1])
            file1 = random.choice(os.listdir(self.root+'/'+alphabet1[0]+'/'+character1))
            file2 = random.choice(os.listdir(self.root+'/'+alphabet2[0]+'/'+character2))
            img2 = Image.open(self.root+'/'+alphabet2[0]+'/'+character2+'/'+file2)
            img1 = Image.open(self.root+'/'+alphabet1[0]+'/'+character1+'/'+file1)
            label=0.
        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        
        return img1,img2,torch.from_numpy(np.array([label],dtype=np.float32))


# N-way-Dataset

class NwayEvaluation(Dataset):
    
    def __init__(self,root,nway,size,transforms=None):
        self.root = root
        self.nway = nway
        self.dir = [(c,os.listdir(self.root+'/'+c)) for c in os.listdir(self.root)]
        self.transforms = transforms
        self.size=size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self,idx):
        
        alphabet = random.choice(self.dir)
        characters = random.choice(alphabet[1])
        character1 = random.choice(os.listdir(self.root+'/'+alphabet[0]+'/'+characters))
        
        mainImg = Image.open(self.root+'/'+alphabet[0]+'/'+characters+'/'+character1)
        test_set=[]
        label = np.random.randint(self.nway)
        for i in range(self.nway):
            if i==label:
                testChar = random.choice(os.listdir(self.root+'/'+alphabet[0]+'/'+characters))
                testImg = Image.open(self.root+'/'+alphabet[0]+'/'+characters+'/'+testChar)
            else:
                alphabet1 = random.choice(self.dir)
                characters1 = random.choice(alphabet1[1])
                while(characters==characters1):
                    characters1 = random.choice(alphabet1[1])
                testChar = random.choice(os.listdir(self.root+'/'+alphabet1[0]+'/'+characters1))
                testImg = Image.open(self.root+'/'+alphabet1[0]+'/'+characters1+'/'+testChar)
                
            
            testImg = transform(testImg)
            test_set.append(testImg)
        
        if self.transforms is not None:
            mainImg = self.transforms(mainImg)
            
        
        return mainImg,test_set,label