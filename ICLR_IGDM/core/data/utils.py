import os
import re
import numpy as np

import torch

from core.utils.utils import np_load
from core.utils.utils import NumpyToTensor
from Externals.robustbench.data import load_cifar10, load_cifar100, load_imagenet
    
class AdversarialDatasetWithPerturbation(torch.utils.data.Dataset):
    """
    Torch dataset for reading examples with corresponding perturbations.
    Arguments:
        root (str): path to saved data.
        transform (torch.nn.Module): transformations to be applied to input.
        target_transform (torch.nn.Module): transformations to be applied to target.
    """
    def __init__(self, root, transform=NumpyToTensor(), target_transform=None):
        super(AdversarialDatasetWithPerturbation, self).__init__()
        
        x_path = re.sub(r'adv_(\d)+', 'adv_0', root)   
        if os.path.isfile(os.path.join(root, 'x.npy')):
            data = np_load(x_path)
        elif os.path.isfile(os.path.join(x_path, 'x.npy')):
            data = np_load(x_path)
        else:
            raise FileNotFoundError('x, y not found at {} and {}.'.format(root, x_path))
        self.data = data['x']
        self.targets = data['y']
        
        data = np_load(root)
        self.r = data['r']
        self.transform = transform
        self.target_transform = target_transform
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
            r = self.transform(self.r[idx])
        if self.target_transform:
            label = self.target_transform(label)
        return image, r, label







def evaluate_aa(model, dataset = 'CIFAR100'):
    n_samples = 1000
    if dataset == 'CIFAR10':
        x_test, y_test = load_cifar10(n_examples=n_samples)
    elif dataset == 'CIFAR100':
        x_test, y_test = load_cifar100(n_examples=n_samples)
    x_test = x_test.cuda()
    y_test = y_test.cuda()
    model.eval() # edit
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce'])
    adversary.apgd.n_restarts = 1
    adversary.apgd.n_iter = 20
    adversary.apgd.topk = 3
    x_adv, y_adv, clean_acc = adversary.run_standard_evaluation(
    x_test, y_test, bs = 128, return_labels = True)
    accuracy = (y_test != y_adv).sum()
    clean_acc = clean_acc * 100
    robust_accuracy = (1 - accuracy/n_samples) * 100.0
    idx = (y_test == y_adv)
    x_test = x_test[idx]
    y_test = y_test[idx]

    if x_test.shape[0] == 0 :
        return clean_acc, 0.0

    
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-t'])
    adversary.apgd.n_restarts = 1
    adversary.apgd.n_iter = 20
    adversary.apgd.topk = 3
    x_adv, y_adv, _ = adversary.run_standard_evaluation(
    x_test, y_test, bs = 128, return_labels = True)

    accuracy += (y_test != y_adv).sum()
    robust_accuracy = (1 - accuracy/n_samples) * 100.0

    return clean_acc, robust_accuracy