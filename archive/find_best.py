#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import data_process_test


path = 'model'


for filename in os.listdir(path):
    content = codecs.open(path+'/'+filename, 'r', 'utf-8').read().split('train_loss:')
    dev = []
    test = []
    for n, sample in enumerate(content[1:]):
        dev.append(sample.split('dev_loss_all: ')[1].split(' ')[0])
        if n == len(content[1:])-1:
            test.append(sample.split('test_loss_all: ')[1])
        else:
            test.append(sample.split('test_loss_all: ')[1].split('\n')[0])
    print(test[dev.index(min(dev))])



print()
