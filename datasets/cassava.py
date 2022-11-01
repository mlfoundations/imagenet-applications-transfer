import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

import os
import cv2
from PIL import Image

class LeafDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return image, label

def get_cassava():
    num_classes = 5
    metadata = False
    data_path = "cassava-leaf-disease-classification"
    train_path = os.path.join(data_path, 'train_images')
    train_csv = os.path.join(data_path, 'train.csv')
    train_df = pd.read_csv(train_csv)

    np.random.seed(414)
    train_size = len(train_df.values)
    indices_by_label = []
    validation_idx = []
    for i in range(num_classes):
        label_idx = [idx for idx, l in enumerate(train_df['label'].values) if l == i]
        validation_idx.extend(np.random.choice(label_idx, int(0.2*len(label_idx)), replace=False).tolist())

    train_idx = [i for i in range(train_size) if i not in validation_idx]
    train_keys = train_df['image_id'].values[train_idx]
    validation_keys = train_df['image_id'].values[validation_idx]
    train_split_df = train_df[train_df['image_id'].isin(train_keys.tolist())]
    validation_split_df = train_df[train_df['image_id'].isin(validation_keys.tolist())]

    img_size = 256

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    testset = LeafDataset(df=validation_split_df, data_path=train_path, transform=transform)

    transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    trainset = LeafDataset(df=train_split_df, data_path=train_path, transform=transform_train)

    return trainset, testset
