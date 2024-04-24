import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F    #relu, tanh => activation func.s
from torch.utils.data import DataLoader 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
from CustomDataset import CatsAndDogDataset

#device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

#Hyperparameters
in_channels =3      # 3 for RBG images
num_classes =10
learning_rate = 1e-3
batch_size =4
num_epochs =1

#Load Data
dataset = CatsAndDogDataset(csv_file='Custom Data Loading/cat&dogs/cats_dogs.csv',
                            root_dir='Custom Data Loading/cat&dogs/cats_dogs_resized',
                            transform=transforms.ToTensor())

#train_set, test_set = torch._utils.data.random_split(dataset,[r1:r2])       # r1:r2 = train-test split ratio
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#rest of code is same