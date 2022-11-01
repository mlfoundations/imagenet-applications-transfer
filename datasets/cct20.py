import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

import os
import cv2
from PIL import Image
import json
import time

class CCTDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None):
        assert split in ['train', 'trans', 'cis', 'test']
        path = 'CCT-20'
        if split == 'train':
            annotations = json.load(open(os.path.join(path, 'eccv_18_annotation_files', 'train_annotations.json')))
        elif split == 'trans': # locations not seen in training (validation)
            annotations = json.load(open(os.path.join(path, 'eccv_18_annotation_files', 'trans_val_annotations.json')))
        elif split == 'cis': # locations seen in training (validation)
            annotations = json.load(open(os.path.join(path, 'eccv_18_annotation_files', 'cis_val_annotations.json')))
        elif split == 'test': # locations seen in training (cis test)
            annotations = json.load(open(os.path.join(path, 'eccv_18_annotation_files', 'cis_test_annotations.json')))
        else:
            print("Not supported split")
            annotations = None
            exit()

        classes = [1, 3, 5, 6, 7, 8, 9, 10, 11, 16, 21, 33, 34, 51, 99]
        # classes = [1, 3, 5, 6, 7, 8, 9, 10, 11, 16, 21, 30, 33, 34, 51, 99] # class numbers including empty
        data = [(anno['image_id'], anno['category_id']) for anno in annotations['annotations'] if anno['category_id'] != 30] # don't use empty
        self.file_names, self.labels = list(zip(*data))
        self.labels = [classes.index(l) for l in self.labels]
        self.transform = transform
        self.data_path = os.path.join(path, 'eccv_18_all_images_sm')
        print("Split: " + split)
        print("Dataset length: " + str(len(self.labels)))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_path, file_name + '.jpg')
        image = cv2.imread(file_path)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return image, label

def get_cct20():
    img_size = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    # testset_cis = CCTDataset(split='cis', transform=transform) # cis val
    testset_true = CCTDataset(split='test', transform=transform)

    transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    trainset = CCTDataset(split='train', transform=transform_train)

    return trainset, testset_true
