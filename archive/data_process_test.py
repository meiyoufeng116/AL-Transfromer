import torch
import os
import codecs
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data

import os
import random
import numpy as np
#from visdom import Visdom
import matplotlib.pyplot as plt

def sample_norm(data, md):
    for i in range(len(data)):
        if data[i] != 0:
            data[i] = data[i] / (md + data[i])
    return data
def standardization(data):
    mu = data.mean()
    sigma = data.std()
    return (data - mu) / sigma
def data_norm(data):
    samples = []
    for s in range(len(data)):
        samples.extend(data[s])
    samples = torch.FloatTensor(samples)
    samples = samples.permute(1, 0)
    for  i, sample in enumerate(samples):
        #t = sample[sample>0]
        #t = np.mean(t)
        if i  == len(samples)-4:
            break
        #print('norm:' + str(i))
        #print('np.median(sample):' + str(t))
        samples[i] = standardization(sample) #sample_norm(sample, t)
    samples = samples.permute(1, 0).numpy().tolist()
    data_normed_list = []
    max_length = max([len(x) for x in data])
    mask = torch.zeros(len(data), max_length, len(data[0][0]))
    data_normed_tensor = torch.zeros(len(data), max_length, len(data[0][0]))
    print('start list to tensor!')
    n = 0
    for i in range(len(data)):
        print('data:' + str(i))
        temp = []
        for j in range(len(data[i])):
            mask[i][j] = torch.ones(len(data[0][0]))
            data_normed_tensor[i][j] = torch.FloatTensor(samples[n])
            #temp.append(data[i][j])
            temp.append(samples[n])
            n = n + 1
        data_normed_list.append(temp)
    assert len(data) == len(data_normed_list), "Data length!!"
    assert len(data[0]) == len(data_normed_list[0]), "Data length!!"
    assert len(data[100]) == len(data_normed_list[100]), "Data length!!"
    return data_normed_list, data_normed_tensor, mask  # list tensor tensor

def dataset_split(data, mask):  #data == tensor   mask == torch    b t input
    # train_data = data.permute(1,0,2)[:40].permute(1,0,2)
    # train_mask = mask.permute(1,0,2)[:40].permute(1,0,2)
    # valid_data = data.permute(1,0,2)[40:50].permute(1,0,2)
    # valid_mask = mask.permute(1,0,2)[40:50].permute(1,0,2)
    # test_data = data.permute(1,0,2)[50:].permute(1,0,2)
    # test_mask = mask.permute(1,0,2)[50:].permute(1,0,2)

    # train_data = train_data.permute(2,1,0)[3:53]
    # train_mask = train_mask.permute(2,1,0)[3:53]
    # valid_data = valid_data.permute(2,1,0)[3:53]
    # valid_mask = valid_mask.permute(2,1,0)[3:53]
    # test_data = test_data.permute(2,1,0)[3:53]
    # test_mask = test_mask.permute(2,1,0)[3:53]  # input b t 

    train_data = data[:10734, :, 3:53].permute(2,1,0)
    train_mask = mask[:10734, :, 3:53].permute(2,1,0)
    valid_data = data[10734:12075, :, 3:53].permute(2,1,0)
    valid_mask = mask[10734:12075, :, 3:53].permute(2,1,0)
    test_data = data[12075:, :, 3:53].permute(2,1,0)
    test_mask = mask[12075:, :, 3:53].permute(2,1,0)

    train_data = train_data[torch.arange(train_data.size(0))!=36] 
    train_mask = train_mask[torch.arange(train_mask.size(0))!=36]
    valid_data = valid_data[torch.arange(valid_data.size(0))!=36]
    valid_mask = valid_mask[torch.arange(valid_mask.size(0))!=36]
    test_data = test_data[torch.arange(test_data.size(0))!=36]
    test_mask = test_mask[torch.arange(test_mask.size(0))!=36]

    train_data = train_data.permute(2,1,0)
    train_mask = train_mask.permute(2,1,0)
    valid_data = valid_data.permute(2,1,0)
    valid_mask = valid_mask.permute(2,1,0)
    test_data = test_data.permute(2,1,0)
    test_mask = test_mask.permute(2,1,0)
    return train_data, train_mask, valid_data, valid_mask, test_data, test_mask
    

