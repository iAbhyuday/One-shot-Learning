import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from dataset import Ominglot,NwayEvaluation
from model import Net
from utils import save_checkpoint,load_checkpoint

# initialize compute device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# PIL image to torch.Tensor
transform =transforms.Compose([transforms.ToTensor()])

omniglot = Ominglot(root='/content/images_background',size=10000,transforms=transform)
train_data,val_data = random_split(omniglot,[7500,2500])
train_data = DataLoader(train_data,batch_size=128,num_workers=16,shuffle=True)
val_data = DataLoader(val_data,batch_size=2,num_workers=16,shuffle=True)

# 20-way test evaluation
testSize = 5000 
numWay = 20
test_set = NwayEvaluation('/content/images_evaluation',numWay,testSize,transforms=transform)
test_loader = DataLoader(test_set, batch_size = 1, num_workers = 2, shuffle=True)



model = Net().to(device)
optim = torch.optim.Adam(model.parameters(),lr=0.006)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=10, gamma=0.1)

def train(epochs):
    train_loss=[]
    val_loss=[]

    for epoch in range(epochs):
        running_loss=0.0
        model.train()
        best_val_loss = float("Inf")
        
        for img1,img2,label in train_data:
            
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)
            
            out = model(img1,img2)
            loss = criterion(out,label)
            
            optim.zero_grad()
            loss.backward()
            
            optim.step()
            
            running_loss+=loss.item()
        avg_loss = running_loss/len(train_data)
        train_loss.append(avg_loss)
        val_running_loss=0.
        
        
        with torch.no_grad():
            model.eval()
            for img1,img2,label in val_data:
            
                img1 = img1.to(device)
                img2 = img2.to(device)
                label = label.to(device)
            
                out = model(img1,img2)
                loss = criterion(out,label)
                val_running_loss+=loss.item()
        avg_val_loss = val_running_loss/len(val_data)
        val_loss.append(avg_val_loss)
        if avg_val_loss<best_val_loss:

            best_val_loss = avg_val_loss
            save_checkpoint('model_check.pt', model, optim, best_val_loss)

        scheduler.step()
        print(f"Epoch {epoch+1} : Train loss : {avg_loss} Validation loss : {avg_val_loss}\n")
    
    return train_loss,val_loss


def eval(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0
        print('Starting Iteration')
        count = 0
        for mainImg, imgSets, label in test_loader:
            mainImg = mainImg.to(device)
            predVal = 0
            pred = -1
            for i, testImg in enumerate(imgSets):
                testImg = testImg.to(device)
                output = model(mainImg, testImg)
                if output > predVal:
                    pred = i
                    predVal = output
            label = label.to(device)
            if pred == label:
                correct += 1
            count += 1
            if count % 20 == 0:
                print("Current Count is: {}".format(count))
                print('Accuracy on n way: {}'.format(correct/count))



if __name__=='__main__':
    train_loss,val_loss = train(50)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel("No. of epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.show()


