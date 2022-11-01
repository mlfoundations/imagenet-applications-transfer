import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def get_eurosat():
    num_classes = 10
    img_size = 224

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    testset_base = torchvision.datasets.ImageFolder(root='eurosat/2750', transform=transform)
    np.random.seed(42)
    all_labels = testset_base.targets
    validation_idx = []
    val_size = 0.2
    for i in range(num_classes):
        label_idx = [idx for idx, l in enumerate(all_labels) if l == i]
        validation_idx.extend(np.random.choice(label_idx, int(val_size*len(label_idx)), replace=False).tolist())
        
    train_idx = [i for i in range(len(all_labels)) if i not in validation_idx]
    assert len(set(train_idx).intersection(set(validation_idx))) == 0
    assert len(set(all_labels)) == num_classes
    testset = torch.utils.data.Subset(testset_base, validation_idx)
    
    transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    trainset_base = torchvision.datasets.ImageFolder(root='eurosat/2750', transform=transform_train)
    trainset = torch.utils.data.Subset(trainset_base, train_idx)

    return trainset, testset

