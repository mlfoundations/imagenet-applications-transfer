import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

import os
import cv2
from PIL import Image

class HPADataset(torch.utils.data.Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.file_names = df['Id'].values
        self.labels = df['Target'].values
        self.transform = transform
        self.data_path = data_path
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_path, file_name + '.png')
        image = cv2.imread(file_path)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = [0 for _ in range(28)]
        for l in [int(x) for x in self.labels[idx].split(' ')]:
            label[l] = 1
        label = torch.tensor(np.array(label))
        return image, label


def get_hpa():
    image_key = 'Id'
    label_key = 'Target'
    num_classes = 28
    data_path = "hpa"
    train_path = os.path.join(data_path, 'rgby', 'train')
    test_path = os.path.join(data_path, 'rgby', 'test')
    train_csv = os.path.join('splits', 'hpa', 'split.stratified.small.0.csv')
    split_df = pd.read_csv(train_csv)
    train_df = split_df[split_df['Split'] =='train']
    val_df = split_df[split_df['Split'] == 'val']
    test_val_df = split_df[split_df['Split'] == 'test_val']
    
    img_size = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    valset = HPADataset(df=val_df, data_path=train_path, transform=transform)
    testvalset = HPADataset(df=test_val_df, data_path=train_path, transform=transform)

    transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    trainset = HPADataset(df=train_df, data_path=train_path, transform=transform_train)

    return trainset, valset, testvalset
