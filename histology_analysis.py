import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization,  Dropout, RandomCrop
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as pltn
import torch
import tensorflow as tf
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import pandas as pd
import argparse
import os
import random
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast
from torchvision.transforms import RandAugment


def count_files(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        num_files = len(files)
        return num_files
    else:
        return "Directory does not exist or is not a valid directory."



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# High learning rate
# High test batch size
# 404 - 74.76 - Normal
# 406 - 72.91 Normalize
# 409 - 70 - Normal 
# 411 - 72.45 - Weight Decay - 5e-4 
# 412 - 70.6 - LR - 0.1  
# 413 - ?? - LR - 0.01  

# num_epochs = 36
# learning_rate = 0.06
num_classes=7
num_epochs = 130
learning_rate = 0.05
size=256


# Define transformations to apply to the images
transform_train = transforms.Compose([
    transforms.Resize((size, size)),  # Resize images to a consistent size  ############################
    transforms.ToTensor() ,
    #RandAugment(num_ops=4)  ,  # Convert images to PyTorch tensors
    transforms.RandomHorizontalFlip(),transforms.RandomCrop(size,padding_mode='reflect'),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    ])
transform = transforms.Compose([
    transforms.Resize((size, size)),  # Resize images to a consistent size  ############################
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

# Path to your dataset containing subdirectories for each class
data_dir = '/home/Student/s4737925/Nan/her2_analysis/data/train'
data_dir_test = '/home/Student/s4737925/Nan/her2_analysis/data/test'
# Load the dataset using ImageFolder and apply transformations
dataset_train = datasets.ImageFolder(root=data_dir, transform=transform_train)
dataset_test = datasets.ImageFolder(root=data_dir_test, transform=transform)

# Split the dataset into training, validation, and test sets
total_len = len(dataset_test)
#train_len = int(0.7 * total_len)  # 70% for training
valid_len = int(0.5 * total_len)  # 15% for validation
test_len = total_len - valid_len  # Remaining for testing

valid_set, test_set = torch.utils.data.random_split(dataset_test, [valid_len, test_len])
train_set=dataset_train

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=100)
test_loader = DataLoader(test_set, batch_size=100)

print(len(train_loader),len(dataset_train),len(test_loader))

import torch.nn as nn
import torch.nn.functional as F

# %%
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, size//8)  # Adjust pooling based on the new input size (64x64)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def Resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

from resnet import *
model = Resnet18() #Resnet50()
# %%

#%%
# if 1==1:
#     from modules import CCT ############ VIT
#     model = CCT(
#         img_size = (size, size),
#         embedding_dim = 192,
#         n_conv_layers = 2,
#         kernel_size = 7,
#         stride = 2,
#         padding = 3,
#         pooling_kernel_size = 3,
#         pooling_stride = 2,
#         pooling_padding = 1,
#         num_layers = 2,
#         num_heads = 6,
#         mlp_ratio = 3.,
#         num_classes = num_classes,
#         positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
#     )

#%%
model = model.to(device)

# model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)




##################################################################################################################################################################
##################################################################################################################################################################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
p=[]
l=[]
total_step = len(train_loader)
scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=learning_rate,steps_per_epoch=total_step, epochs=num_epochs)

scaler = torch.cuda.amp.GradScaler()

# Train the model
model.train()
print("> Training")
start = time.time()  # time generation
for epoch in range(num_epochs):
    br = None
    total=0
    correct=0
    l_v=True
    model.train()
    for i, (images, labels) in enumerate(train_loader):  # load a batch
        #print(images)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        #################### We go forward and calculate the weights
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
          ## autocast serve as context managers that allow regions of your script to run in mixed precision.
          ## In these regions, CUDA ops run in a dtype chosen by autocast to improve performance while maintaining accuracy.
            #print(images.shape)
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward and optimize
        ###################### We now go back and optimise the values
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        br = loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        #if (i + 1) % 100 == 0:
        if l_v: 
            l.append(loss.item())
            l_v=False
        print("Accuracy:{}, Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(100.*correct/total,epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        scheduler.step()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        p.append(100 * correct / total)
        print('Valid Accuracy: {} %'.format(100 * correct / total))
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed / 60) + " mins in total")
###############################################################################################################################################################
# Test the model
print("> Testing")
start = time.time() #time generation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

print('END')

import matplotlib.pyplot as plt
from datetime import datetime
torch.save(model.train(), str(datetime.now().strftime("%H:%M:%S")) + 'model.pth')
plt.plot(p)
plt.savefig(f'accuracy_{datetime.now()}.png')
plt.clf()
import matplotlib.pyplot as plt2
plt2.plot(l)
plt2.savefig(f'loss_{datetime.now()}.png')

def result(model, device, testloader):
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from resnet18 import Resnet18

    # Define the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.to(device)
    model.eval()


    # Initialize lists to store predicted and true labels
    y_pred = []
    y_true = []

    # Perform inference on the test set
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)

            # Convert logits to class predictions
            _, predicted = torch.max(output, 1)

            # Append predictions and true labels to lists
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

    # Calculate confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(y_pred, y_true)
    # Plot the confusion matrix
    classes = ('0', '1', '2', '3', '4', '5', '6')
    df_cm = pd.DataFrame(cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis], index=classes, columns=classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt=".2f")
    plt.savefig(str(datetime.now().strftime("%H:%M:%S")) + 'confusion.png')

    # Calculate additional classification metrics
    print(cf_matrix.ravel())

result(model,device, test_loader)