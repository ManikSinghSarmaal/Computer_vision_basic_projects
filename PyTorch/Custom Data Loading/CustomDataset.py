import os
import pandas as pd 
import torch 
from torch.utils.data import Dataset
from skimage import io


#loading data from a csv file
class CatsAndDogDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform = None):
        self.annotations = pd.read_csv(csv_file)        #refer csv_file for data structure
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])      # csv file's ith row i.e. index and 1st coloumn which is the name f the img
        img = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))     #again refer to csv file, 2nd row is label of image
        
        if self.transform:
            img = self.transform(img)
        return (img, y_label)