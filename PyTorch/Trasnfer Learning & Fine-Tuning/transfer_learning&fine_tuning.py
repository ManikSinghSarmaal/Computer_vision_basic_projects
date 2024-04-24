import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F    #relu, tanh => activation func.s
from torch.utils.data import DataLoader 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys

#device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

#Hyperparameters
in_channels =3      # 3 for RBG images
num_classes =10
learning_rate = 1e-3
batch_size =64
num_epochs =1

class Identity(nn.Module):      # To pass avgpool layer so it doesn't affects features, Identity = Do Nothing
    def __init__(self) :
       super(Identity,self).__init__()
       
    def forward(self,x):
        return x
    
#load the pretrained-model
model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)  # you can use pretrained=True but this way you'll get the most updated weights
#freezing top layers to not train the whole network
for param in model.parameters():
    param.requires_grad=False
  
model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512,100),
                                 nn.ReLU(),
                                 nn.Linear(100,10))

model.to(device)

train_dataset = datasets.CIFAR10(root= 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader (dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root='dataset/',train=False,transform=transforms.ToTensor(), download=True)
test_loader = DataLoader (dataset=test_dataset,batch_size=batch_size, shuffle=True)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

#train the network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        scores = model(data)
        #print(f'The tagets shape is {targets.shape},The scores shape is {scores.shape} and The data shape is {data.shape}')
        #break
        loss = criterion(scores, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        #gradient descent
        optimizer.step()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
    
def check_accuracy(loader,model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
        
    num_correct =0
    num_samples = 0
    model.eval()
    with torch.no_grad(): 
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions== y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) *100: 2f}')
        model.train()
        
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
