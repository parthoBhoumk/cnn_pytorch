import torch 
import numpy as np
import pandas  as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
# -------------------- loading test set--------------------------------

mnist_train= torchvision.datasets.MNIST(
    
        "./data_mnist",    
        train = True
        ,download=True
        ,transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            
            transforms.ToTensor()
        ])
    
)


emnist_train= torchvision.datasets.EMNIST(

        "./data_emnist", 
        "letters",
        train = True
        ,download=True
        ,transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            
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
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            
            transforms.ToTensor()
        ])
    
)


emnist_valid= torchvision.datasets.EMNIST(

        "./data_emnist", 
        "letters",
        train = False
        ,download=True
        ,transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            
            transforms.ToTensor()
        ])
    
)
emnist_valid.targets += 10

train_set = ConcatDataset([emnist_train,mnist_train])
valid_set = ConcatDataset([emnist_valid,mnist_valid])

train_loader = DataLoader(train_set,batch_size=100,num_workers=1,shuffle = True)
valid_loader = DataLoader(valid_set,batch_size=100,num_workers=1)

batch = next(iter(train_loader))
images , labels = batch 
for image in images:
  transforms.ToPILImage(image)

grid = torchvision.utils.make_grid(images,nrow = 5)
plt.figure(figsize = (15,15))
plt.imshow(np.transpose(grid,(1,2,0)))
print('labels:' , labels)


torch.manual_seed(102)

                                          # declaring model
model_1 = torchvision.models.resnet152(pretrained = True)
for param in model_1.parameters():
  param.requires_grad = False

                              # changing the output featers as my class number(10 for MNIST)

num_ftrs = model_1.fc.in_features
model_1.fc = nn.Linear(num_ftrs,37)


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
    for batch in train_loader:                      # Get Batch
        
        images = batch[0].cuda()
        labels = batch[1].cuda()

        preds = model_1(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights
        train_loss += loss.item()
        
        train_correct += get_num_correct(preds,labels)
    train_accuracy = train_correct/len(train_set)

    #-------------------------------------validation loop-------------------------------------
    model_1.eval()
    valid_loss =  0
    valid_correct = 0
    with torch.no_grad():

        for v_batch in valid_loader:                      # Get Batch
            
            v_images = v_batch[0].cuda()
            v_labels = v_batch[1].cuda()

            v_preds = model_1(v_images) # Pass Batch
            v_loss = F.cross_entropy(v_preds, v_labels) # Calculate Loss
            
            valid_loss += v_loss.item()


            valid_correct += get_num_correct(v_preds,v_labels)
        valid_accuracy = valid_correct/len(valid_set)

    print("epoch: ", epoch+1 , "train loss:", train_loss, "train_accuracy:" ,train_accuracy, "valid_loss:", valid_loss, "valid_accuracy", valid_accuracy )
    
    
