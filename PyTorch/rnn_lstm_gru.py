#MNIST Dataset looks like this when loaded => Nx1x28x28 (batch_size,num_channels,width,height)

import torch
import torch.nn as nn       #Conv2D, Linear layers and all
import torch.optim as optim     #for optimization algorithms like SGD, Adam,etc
import torch.nn.functional as F    #relu, tanh => activation func.s
from torch.utils.data import DataLoader     #create mini batches
import torchvision.datasets as datasets
import torchvision.transforms as transforms


#Hyperparameters
input_size = 28
sequence_length = 28
num_layers =2
hidden_size = 256
num_classes =10
learning_rate = 0.001
batch_size =64
num_epochs =1

#device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

#RNN
class RNN(nn.Module):
    def __init__(self, input_size,num_layers, hidden_size, num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)       #feel free to replace RNN with GRU, gru here gives a good accuracy than rnn
        #but you might get a loss value of NaN which can be due to gradient clipping so be aware of that
        
        #input => Nxtime_seqxfeatures
        self.batch_norm = nn.BatchNorm1d(hidden_size * sequence_length)     # BatchNorm layer can reduce accuracy on your data but gives a better generalisation
        self.fc1 = nn.Linear(hidden_size*sequence_length,num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)  # _ is the hidden state in every RNN step, and we don't need to store it
        out = out.contiguous().view(x.size(0), -1)  # Flatten the output
        out = self.fc1(out)
        return out
    
class LSTM(nn.Module):
    def __init__(self, input_size,num_layers, hidden_size, num_classes):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)       #feel free to replace RNN with GRU, gru here gives a good accuracy than rnn
        #but you might get a loss value of NaN which can be due to gradient clipping so be aware of that
        
        #input => Nxtime_seqxfeatures
        self.batch_norm = nn.BatchNorm1d(hidden_size * sequence_length)     # BatchNorm layer can reduce accuracy on your data but gives a better generalisation
        self.fc1 = nn.Linear(hidden_size*sequence_length,num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0,c0))  # _ is the hidden state in every RNN step, and we don't need to store it
        out = out.reshape(x.size(0), -1)  # Flatten the output
        out = self.fc1(out)
        return out
    
# Load data
train_dataset = datasets.MNIST(root= 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader (dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(), download=True)
test_loader = DataLoader (dataset=test_dataset,batch_size=batch_size, shuffle=True)

#NETWORK

#RNN
#model = RNN(input_size=input_size,num_layers=num_layers,hidden_size=hidden_size, num_classes=num_classes).to(device)

#LSTM
model = LSTM(input_size=input_size,num_layers=num_layers,hidden_size=hidden_size, num_classes=num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

#train the network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device).squeeze(1)
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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions== y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) *100: 2f}')
        model.train()
        
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
        