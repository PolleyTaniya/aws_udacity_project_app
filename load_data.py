from torchvision import transforms 
import sys
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
import torchvision.models as models
import os

batch_size = 32

def load_data(data_dir):
    """
    Loads training, validation, and testing datasets, applies transformations, and returns DataLoaders.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    ##Define transforms for the training, validation, and testing sets
    
    data_transforms = {
    'train':
    transforms.Compose([transforms.RandomResizedCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomRotation(30),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ]),
    'valid':
    transforms.Compose([transforms.Resize(256),
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ]),
    'test':
    transforms.Compose([transforms.Resize(256),
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ]),
    
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train' : ImageFolder(train_dir, data_transforms['train']),
        'test' : ImageFolder(test_dir, data_transforms['test']),
        'valid' : ImageFolder(valid_dir, data_transforms['valid'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    
    dataloaders = {
        'train' : DataLoader(image_datasets['train'], batch_size = batch_size, shuffle = True, num_workers = 2),
        'valid' : DataLoader(image_datasets['valid'], batch_size = batch_size, shuffle = False, num_workers = 2),
        'test' : DataLoader(image_datasets['test'], batch_size = batch_size, shuffle = True),
    }
    trainLoader = dataloaders['train']
    validLoader = dataloaders['valid']
    testLoader = dataloaders['test']
    
    Image_set_train = image_datasets['train']
    return trainLoader, validLoader, testLoader, image_datasets

