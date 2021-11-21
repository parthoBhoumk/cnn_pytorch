import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict
#from runbuilder import RunBuilder
#from runmanager import RunManager

torch.manual_seed(102)

                                          # declaring model
model_1 = torchvision.models.resnet152(pretrained = True)
for param in model_1.parameters():
  param.requires_grad = False

                              # changing the output featers as my class number(10 for MNIST)

num_features = model_1.fc.in_features
model_1.fc = nn.Linear(num_features,100)
model_1.fc.out_features 

                                     


digit_train_set = torchvision.datasets.CIFAR100(
    
        "./data_cifar",    
        train = True,
        download=True
        ,transform = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),(0.2623, 0.2513, 0.2714))
            
            
        ])
    
)

digit_valid_set = torchvision.datasets.CIFAR100(
    
        "./data_cifar",    
        train = False,
        download=True
        ,transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5088, 0.4874, 0.4419),(0.2631, 0.2521, 0.2723))
            
            
        ])
    
)



digit_train_loader = DataLoader(digit_train_set, batch_size=100,num_workers=1, shuffle=True)
digit_valid_loader = DataLoader(digit_valid_set, batch_size=100, num_workers=1)

model_1 = model_1.cuda()

optimizer = optim.Adam(model_1.parameters(), lr=0.01)

def get_num_correct(preds,labels):
    
    return preds.argmax(dim = 1).eq(labels).sum().item()

epochs = 20
for epoch in range(epochs):

    #-----------------------------------Training loop----------------------------------------
    model_1.train()
    train_loss = 0
    train_correct = 0
    for batch in digit_train_loader:                      # Get Batch
        
        images = batch[0].cuda()
        labels = batch[1].cuda()

        preds = model_1(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights
        train_loss += loss.item()
        
        train_correct += get_num_correct(preds,labels)
    train_accuracy = train_correct/len(digit_train_set)

    #-------------------------------------validation loop-------------------------------------
    model_1.eval()
    valid_loss =  0
    valid_correct = 0
    with torch.no_grad():

        for v_batch in digit_valid_loader:                      # Get Batch
            
            v_images = v_batch[0].cuda()
            v_labels = v_batch[1].cuda()

            v_preds = model_1(v_images) # Pass Batch
            v_loss = F.cross_entropy(v_preds, v_labels) # Calculate Loss
            
            valid_loss += v_loss.item()
            
            valid_correct += get_num_correct(v_preds,v_labels)
        valid_accuracy = valid_correct/len(digit_valid_set)

    print("epoch: ", epoch+1 , "train loss:", train_loss, "train_accuracy:" ,train_accuracy, "valid_loss:", valid_loss, "valid_accuracy", valid_accuracy )
