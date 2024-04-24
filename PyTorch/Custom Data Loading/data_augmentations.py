import torch
from torch.utils.data import DataLoader 
from torchvision.utils import save_image
import torchvision.transforms as transforms
from CustomDataset import CatsAndDogDataset



my_transforms = transforms.Compose([
    transforms.ToPILImage(),      # because all the transforms work only on PIL Image
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomCrop((224,224)),
    transforms.RandomGrayscale(p=0.2),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])  #normalises image pixels as (pixel_value - Mean)/std
])

#Load Data
dataset = CatsAndDogDataset(csv_file='cat&dogs/cats_dogs.csv',
                            root_dir='cat&dogs/cats_dogs_resized',
                            transform=my_transforms)
img_num=0
for _ in range(10):     #range(10) so 10 new images per any image
    for img, label in dataset:
        save_image(img,'augmented_images/img'+str(img_num)+'.png')       #remember to create an augmented_images folder where you want to store aug_images
        img_num+=1
