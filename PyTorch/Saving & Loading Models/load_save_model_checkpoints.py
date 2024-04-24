import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F    #relu, tanh => activation func.s
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#doing for MNIST dataset so in_channels=1 because greayscale

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1), padding=(1,1))      #same convolution
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)       # 16*7*7 because we are applying 2 maxpooling layers in forward pass

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

#device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

#Hyperparameters
in_channels =1
num_classes =10
learning_rate = 0.001
batch_size =64
num_epochs =1
load_model = True       # to load model from a checkpoint and start from there

# Load data
train_dataset = datasets.MNIST(root= 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader (dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(), download=True)
test_loader = DataLoader (dataset=test_dataset,batch_size=batch_size, shuffle=True)

#NETWORK
model = CNN().to(device)

#saving checkpoint function
def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    print('=> Saving the model checkpoint')
    torch.save(state,filename)
    
#loading a checkpoint
def load_checkpoint(checkpoint_path):
    print('=> Loading the model checkpoint')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

if load_model:
    load_checkpoint('my_checkpoint.pth.tar')

#train the network
for epoch in range(num_epochs):
    losses =[]
    
    if epoch % 3 == 0:        # save checkpoint and overwrite checkpoint file after every 3 epochs
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
        
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        scores = model(data)
        #print(f'The tagets shape is {targets.shape},The scores shape is {scores.shape} and The data shape is {data.shape}')
        #break
        loss = criterion(scores, targets)
        losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        #gradient descent
        optimizer.step()
        
        #mean_loss = sum(losses)/len(losses)
        #print(f'Loss at epoch {epoch} is {mean_loss:.5f}')
        
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











device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')