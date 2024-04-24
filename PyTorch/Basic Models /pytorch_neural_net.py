import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F    #relu, tanh => activation func.s
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self,input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)     
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


#Hyperparameters
input_size = 784
num_classes =10
learning_rate = 0.001
batch_size =64
num_epochs =1

# Load data
train_dataset = datasets.MNIST(root= 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader (dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(), download=True)
test_loader = DataLoader (dataset=test_dataset,batch_size=batch_size, shuffle=True)

#NETWORK
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

#train the network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        data = data.reshape(data.shape[0],-1)
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
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions== y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) *100: 2f}')
        model.train()
        
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)