import torch

checkpoint = torch.load('my_checkpoint.pth.tar')
print(checkpoint.keys())