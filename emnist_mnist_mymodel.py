import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
#from IPython.display import display, clear_output
#import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict

# from runbuilder import RunBuilder
# from runmanager import RunManager

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

torch.manual_seed(102)
#declaring model
test_net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.BatchNorm2d(32),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.BatchNorm2d(64),
    nn.Dropout2d(0.4),
    Flatten(),
    nn.Linear(in_features=64 * 4 * 4, out_features=1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Dropout(0.5),
    nn.Linear(in_features=1024, out_features=512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(in_features=512, out_features=120),
    nn.ReLU(),
    nn.BatchNorm1d(120),
    nn.Linear(in_features=120, out_features=37),
)

# loading data from url of Mnist

# loading normalized data
mnist_train= torchvision.datasets.MNIST(
    
        "./data_mnist",    
        train = True
        ,download=True
        ,transform = transforms.Compose([
           # transforms.Resize((224,224)),
            #transforms.Grayscale(num_output_channels=3),
            
            transforms.ToTensor()
        ])
    
)

emnist_train= torchvision.datasets.EMNIST(

        "./data_emnist", 
        "letters",
        train = True
        ,download=True
        ,transform = transforms.Compose([
            #transforms.Resize((224,224)),
            #transforms.Grayscale(num_output_channels=3),
            
            transforms.ToTensor()
        ])
    
)
emnist_train.targets += 10

#---------------------loading valid set--------------------------

mnist_valid= torchvision.datasets.MNIST(
    
        "./data_mnist",    
        train = False
        ,download=True
        ,transform = transforms.Compose([
          #  transforms.Resize((224,224)),
           # transforms.Grayscale(num_output_channels=3),
            
            transforms.ToTensor()
        ])
    
)


emnist_valid= torchvision.datasets.EMNIST(

        "./data_emnist", 
        "letters",
        train = False
        ,download=True
        ,transform = transforms.Compose([
           # transforms.Resize((224,224)),
            #transforms.Grayscale(num_output_channels=3),
            
            transforms.ToTensor()
        ])
    
)
emnist_valid.targets += 10

train_set = ConcatDataset([emnist_train,mnist_train])
valid_set = ConcatDataset([emnist_valid,mnist_valid])

#test_net = torch.load("./test_net/test_net.pth")
test_net = test_net.cuda()

train_loader = DataLoader(train_set, batch_size=1000, num_workers=1, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=1000, num_workers=1)

optimizer = optim.Adam(test_net.parameters(), lr=0.001)


def get_num_correct(preds, labels):

    return preds.argmax(dim=1).eq(labels).sum().item()


epochs = 20
for epoch in range(epochs):

    # -----------------------------------Training loop----------------------------------------
    test_net.train()
    train_loss = 0
    train_correct = 0
    for batch in train_loader:  # Get Batch

        images = batch[0].cuda()
        labels = batch[1].cuda()

        preds = test_net(images)  # Pass Batch
        loss = F.cross_entropy(preds, labels)  # Calculate Loss
        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights
        train_loss += loss.item()

        train_correct += get_num_correct(preds, labels)
    train_accuracy = train_correct / len(train_set)

    # -------------------------------------validation loop-------------------------------------
    test_net.eval()
    valid_loss = 0
    valid_correct = 0
    with torch.no_grad():

        for v_batch in valid_loader:  # Get Batch

            v_images = v_batch[0].cuda()
            v_labels = v_batch[1].cuda()

            v_preds = test_net(v_images)  # Pass Batch
            v_loss = F.cross_entropy(v_preds, v_labels)  # Calculate Loss

            valid_loss += v_loss.item()

            valid_correct += get_num_correct(v_preds, v_labels)
        valid_accuracy = valid_correct / len(valid_set)

    print(
        "epoch: ",
        epoch + 1,
        "train loss:",
        train_loss,
        "train_accuracy:",
        train_accuracy,
        "valid_loss:",
        valid_loss,
        "valid_accuracy",
        valid_accuracy,
    )

#torch.save(test_net,"./test_net/test_net.pth")
#print(test_net)
