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


# In[2]:


SEED = 1234
torch.backends.cudnn.enabled = False
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


###sample id: [1] icustayid

###Long-term: [2-10]
###age needs to map to original range
###consider gender[3], age[4], elixhauser[5]


###Short-term:[11-49] [54-64]
### ignore: Shock_Index[48] SIRS[64]
### embedding v: mechvent [47]

###Treatment[50-53]:
###median_dose_vaso,max_dose_vaso,input_total,input_4hourly

### sample: long [0:3] + short [3:51] + treatment [51:55]


#output = open('myfile.pkl', 'wb')
#pickle.dump(mydict, output)
#output.close()
#pkl_file = open('myfile.pkl', 'rb')
#mydict2 = pickle.load(pkl_file)

def read_data(path):
    data = codecs.open(path,'r').read().split('\n')[1:-1]
    dataset = []
    samples = []
    current_id = 1
    for i, content in enumerate(data):
        content = [float(v) for v in content.split(',')]
        sample = content[3:6]
        print(content[3:6])
        sample.extend(content[11:48])
        print(content[11:48])
        sample.extend(content[49:50])
        print(content[49:50])
        sample.extend(content[54:64])
        print(content[54:64])
        sample.extend(content[50:54])
        print(content[50:54])
        break
        if content[1] != current_id:
            dataset.append(samples)
            samples = []
            samples.append(sample)
            current_id = content[1]
            print(len(dataset))
        else:
            samples.append(sample)
        if i == len(data)-1:
            dataset.append(samples)
    print(dataset[0])
    return dataset

def save_dataset(dataset, path):
    output = open(path, 'wb')
    pickle.dump(dataset, output)
    output.close()

def read_dataset(path):
    pkl_file = open(path, 'rb')
    return pickle.load(pkl_file)

#dataset = read_data('../data/data/MIMIC_N14594_7-23-2017_1hours.csv')
#print(dae)
#save_dataset(dataset, 'dataset/dataset.pkl')

#dataset = read_dataset('dataset/dataset.pkl')

#_, dataset, mask = data_process_test.data_norm(dataset)
#print(dae)
#save_dataset((dataset, mask), 'dataset/normalized_dataset_new.pkl')
#print(dae)

# In[4]:


class Dataset(data.Dataset):
  #'Characterizes a dataset for PyTorch'
    def __init__(self, dataset, mask, valid_data, valid_mask, test_data, test_mask):
        #'Initialization'
        #self.X = torch.load(path)
        #self.X = pickle.load(open(path, 'rb'))
        self.X = dataset
        self.M = mask
        self.vx = valid_data
        self.vm = valid_mask
        self.tx = test_data
        self.tm = test_mask


    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample
        return self.X[index], self.M[index], self.vx[index], self.vm[index], self.tx[index], self.tm[index]

class TrainDataset(Dataset):
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.mask[index]
    
class VaildDataset(Dataset):
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.mask[index]
    
class TestDataset(Dataset):
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.mask[index]

# In[ ]:


class EncoderDecoder(nn.Module):
    def __init__(self,en_hidden_size=64, en_num_layers=2, en_dropout=0.2, de_hidden_size=64, de_num_layers=2, de_dropout=0.2):
        super(EncoderDecoder, self).__init__()
        self.en_hidden_size = en_hidden_size
        self.en_num_layers = en_num_layers
        self.en_dropout = en_dropout
        self.de_hidden_size = de_hidden_size
        self.de_num_layers = de_num_layers
        self.de_dropout = de_dropout
        self.encoder = nn.LSTM(
            input_size=49,
            hidden_size=self.en_hidden_size,     # rnn hidden unit
            num_layers=self.en_num_layers,       # number of rnn layer
            dropout=self.en_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.decoder = nn.LSTM(
            input_size=self.en_hidden_size,
            hidden_size=self.de_hidden_size,     # rnn hidden unit
            num_layers=self.de_num_layers,       # number of rnn layer
            dropout=self.de_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.drop_layer = nn.Dropout(p=self.de_dropout)
        self.out = nn.Linear(self.de_hidden_size, 49)

    def forward(self, x, h_state=None):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.encoder(x, h_state)
        #return r_out, h_state
        r_out = self.drop_layer(r_out)
        if self.en_num_layers == self.de_num_layers:
            r_out, h_state = self.decoder(r_out, h_state)
        else:
            r_out, h_state = self.decoder(r_out, (torch.unsqueeze(h_state[0][(self.en_num_layers-self.de_num_layers):], 0), torch.unsqueeze(h_state[1][(self.en_num_layers-self.de_num_layers)],0)))
        r_out = self.drop_layer(r_out)
        r_out = self.out(r_out)
        return r_out, h_state



class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, ResidualBlock, num_classes=2, response_size=49, window_sizes = [1,2,3,5,8,12], feature_size=64,window=12):
        super(ResNet18, self).__init__()
        self.inchannel = feature_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(response_size, feature_size, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(),
        )
        self.window=window
        self.num_classes=num_classes
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        result = torch.zeros(x.shape[1]-self.window, x.shape[0], self.num_classes).cuda()  # t b output_size
        x = x.permute(1,2,0) #-> t input_size b
        for n, sample in enumerate([x[i:i+self.window] for i in range(x.shape[0]-self.window)]):
            sample = sample.permute(2,1,0)  #batch_size x input_size x 24(maxlength)
            out = self.conv1(sample)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool1d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            result[n] = out   # -> t b c
        result = result.permute(1,0,2)  #-> b t c
        return result




class Classifier(nn.Module):
    """ Simple network"""
    def __init__(self, response_size=49, window_sizes = [1,2,3,5,8], feature_size=128,num_classes=2,window=12):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.window = window
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(response_size, feature_size, kernel_size=h, stride=1, padding=1), # 28   #1
            nn.BatchNorm1d(feature_size),
            nn.ReLU(),
            nn.Conv1d(feature_size, feature_size, kernel_size=h, stride=1, padding=1), # 28   #1
            nn.BatchNorm1d(feature_size),
            nn.MaxPool1d(kernel_size=self.window-h+1))
            for h in window_sizes])

        self.classifier = nn.Sequential(
            #nn.Linear(feature_size*len(window_sizes), feature_size),
            #nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(feature_size, self.num_classes)
        )

    def forward(self, x):   #x (batch, time_step, input_size)
        result = torch.zeros(x.shape[1]-self.window+1, x.shape[0], self.num_classes).cuda()  # t b output_size
        x = x.permute(1,2,0) #-> t input_size b
        for n, sample in enumerate([x[i:i+self.window] for i in range(x.shape[0]-self.window+1)]):
            sample = sample.permute(2,1,0)  #batch_size x input_size x 24(maxlength)
            sample = [conv(sample) for conv in self.convs]  # -->out[i]:batch_size x feature_size x 1 #self.convs(x)
            #print(sample.shape)
            sample = torch.cat(sample, dim=1)
            sample = sample.view(-1, sample.size(1))
            sample = self.classifier(sample)  #-> batch * 2
            result[n] = sample   # -> t b 2
        result = result.permute(1,0,2)  #-> b t 2
        return result   # b t 2

class ADRNN_old(nn.Module):
    def __init__(self, hidden_size=128, response_size=47, r_layers=2, r_dropout=0.5, treat_size=2, t_layers=2, t_dropout=0.5):
        super(ADRNN_old, self).__init__()
        self.response_size = response_size
        self.r_layers = r_layers
        self.r_dropout = r_dropout
        self.treat_size = treat_size
        self.t_layers = t_layers
        self.t_dropout = t_dropout
        self.r_rnn = nn.LSTM(
            input_size=self.response_size + self.treat_size,
            hidden_size=self.response_size,     # rnn hidden unit
            num_layers=self.r_layers,       # number of rnn layer
            dropout=self.r_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.t_rnn = nn.LSTM(
            input_size=self.response_size*2 + self.treat_size,
            hidden_size=self.treat_size,     # rnn hidden unit
            num_layers=self.t_layers,       # number of rnn layer
            dropout=self.t_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.hidden_size, input_size)


    def forward(self, x_r, x_t, h_r=None, h_t=None,is_train=True, gen_length=None):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_output = torch.zeros(x_r.shape[1], x_r.shape[0], self.response_size).cuda()  #   t b response_size
        t_output = torch.zeros(x_t.shape[1], x_t.shape[0], self.treat_size).cuda()  #   t b treat_size
        batch = x_t.shape[0]
        x_r = x_r.permute(1,0,2)  # -> t b in
        x_t = x_t.permute(1,0,2)  # -> t b in

        if is_train == False:
            r_output = torch.zeros(gen_length, batch, self.response_size).cuda()  #   t b response_size
            t_output = torch.zeros(gen_length, batch, self.treat_size).cuda()  #   t b treat_size
            for i, (r, t) in enumerate(zip(x_r, x_t)):
                r = torch.unsqueeze(r, 0) # -> 1 b in
                t = torch.unsqueeze(t, 0) # -> 1 b in
                r = r.permute(1,0,2) # -> b 1 in
                t = t.permute(1,0,2) # -> b 1 in
                r_input = torch.cat((r,t), 2) # b 1 in+in
                r_out, h_r = self.r_rnn(r_input, h_r)  #r_out = b 1 response_size

                t_input = torch.cat((r_input,r_out), 2) # b 1 in+2response_size
                t_out, h_t = self.t_rnn(t_input, h_t)
            for i in range(gen_length):
                r_input = torch.cat((r_out,t_out), 2) # b 1 in+in
                r_out, h_r = self.r_rnn(r_input, h_r)  #r_out = b 1 response_size
                t_input = torch.cat((r_input,r_out), 2) # b 1 in+2response_size
                t_out, h_t = self.t_rnn(t_input, h_t)
                r_output[i] = r_out.permute(1,0,2)[0]
                t_output[i] = t_out.permute(1,0,2)[0]
            return r_output.permute(1,0,2), t_output.permute(1,0,2)

        for i, (r, t) in enumerate(zip(x_r, x_t)):
            r = torch.unsqueeze(r, 0) # -> 1 b in
            t = torch.unsqueeze(t, 0) # -> 1 b in
            r = r.permute(1,0,2) # -> b 1 in
            t = t.permute(1,0,2) # -> b 1 in
            r_input = torch.cat((r,t), 2) # b 1 in+in
            r_out, h_r = self.r_rnn(r_input, h_r)  #r_out = b 1 response_size

            t_input = torch.cat((r_input,r_out), 2) # b 1 in+2response_size
            t_out, h_t = self.t_rnn(t_input, h_t)

            r_output[i] = r_out.permute(1,0,2)[0]
            t_output[i] = t_out.permute(1,0,2)[0]

        return r_output.permute(1,0,2), t_output.permute(1,0,2)

class ADRNN(nn.Module):
    def __init__(self, hidden_size=128, response_size=47, r_layers=2, r_dropout=0.5, treat_size=2, t_layers=2, t_dropout=0.5):
        super(ADRNN, self).__init__()
        self.hidden_size = hidden_size
        self.response_size = response_size
        self.r_layers = r_layers
        self.r_dropout = r_dropout
        self.treat_size = treat_size
        self.t_layers = t_layers
        self.t_dropout = t_dropout
        self.r_rnn = nn.LSTM(
            input_size=self.response_size + self.treat_size,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.r_layers,       # number of rnn layer
            dropout=self.r_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.t_rnn = nn.LSTM(
            input_size=self.response_size*2 + self.treat_size,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.t_layers,       # number of rnn layer
            dropout=self.t_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.drop_layer = nn.Dropout(0)
        self.linear_r = nn.Linear(self.hidden_size, self.response_size)
        self.linear_t = nn.Linear(self.hidden_size, self.treat_size)


    def forward(self, x_r, x_t, h_r=None, h_t=None,is_train=True, gen_length=None):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_output = torch.zeros(x_r.shape[1], x_r.shape[0], self.response_size).cuda()  #   t b response_size
        t_output = torch.zeros(x_t.shape[1], x_t.shape[0], self.treat_size).cuda()  #   t b treat_size
        batch = x_t.shape[0]
        x_r = x_r.permute(1,0,2)  # -> t b in
        x_t = x_t.permute(1,0,2)  # -> t b in

        if is_train == False:
            r_output = torch.zeros(gen_length, batch, self.response_size).cuda()  #   t b response_size
            t_output = torch.zeros(gen_length, batch, self.treat_size).cuda()  #   t b treat_size
            for i, (r, t) in enumerate(zip(x_r, x_t)):
                r = torch.unsqueeze(r, 0) # -> 1 b in
                t = torch.unsqueeze(t, 0) # -> 1 b in
                r = r.permute(1,0,2) # -> b 1 in
                t = t.permute(1,0,2) # -> b 1 in
                r_input = torch.cat((r,t), 2) # b 1 in+in
                r_out, h_r = self.r_rnn(r_input, h_r)  #r_out = b 1 response_size
                r_out=self.drop_layer(r_out)
                r_out = self.linear_r(r_out)
                t_input = torch.cat((r_input,r_out), 2) # b 1 in+2response_size
                t_out, h_t = self.t_rnn(t_input, h_t)
                t_out=self.drop_layer(t_out)
                t_out = self.linear_t(t_out)
            r_output[0] = r_out.permute(1,0,2)[0]
            t_output[0] = t_out.permute(1,0,2)[0]
            for i in range(gen_length-1):
                r_input = torch.cat((r_out,t_out), 2) # b 1 in+in
                r_out, h_r = self.r_rnn(r_input, h_r)  #r_out = b 1 response_size
                r_out=self.drop_layer(r_out)
                r_out = self.linear_r(r_out)
                t_input = torch.cat((r_input,r_out), 2) # b 1 in+2response_size
                t_out, h_t = self.t_rnn(t_input, h_t)
                t_out=self.drop_layer(t_out)
                t_out = self.linear_t(t_out)
                r_output[i+1] = r_out.permute(1,0,2)[0]
                t_output[i+1] = t_out.permute(1,0,2)[0]
            return r_output.permute(1,0,2), t_output.permute(1,0,2)

        for i, (r, t) in enumerate(zip(x_r, x_t)):
            r = torch.unsqueeze(r, 0) # -> 1 b in
            t = torch.unsqueeze(t, 0) # -> 1 b in
            r = r.permute(1,0,2) # -> b 1 in
            t = t.permute(1,0,2) # -> b 1 in
            r_input = torch.cat((r,t), 2) # b 1 in+in
            r_out, h_r = self.r_rnn(r_input, h_r)  #r_out = b 1 response_size
            r_out=self.drop_layer(r_out)
            r_out = self.linear_r(r_out)

            t_input = torch.cat((r_input, r_out), 2)############################### # b 1 in+2response_size
            #if i < len(x_r)-1:
            #    t_input = torch.cat((r_input, torch.unsqueeze(x_r[i+1],1)), 2)
            #else:
            #    t_input = torch.cat((r_input, r_out), 2)
            t_out, h_t = self.t_rnn(t_input, h_t)
            t_out=self.drop_layer(t_out)
            t_out = self.linear_t(t_out)

            r_output[i] = r_out.permute(1,0,2)[0]
            t_output[i] = t_out.permute(1,0,2)[0]

        return r_output.permute(1,0,2), t_output.permute(1,0,2)

class ADRNN_new(nn.Module):
    def __init__(self, hidden_size=128, response_size=47, r_layers=2, r_dropout=0.5, treat_size=2, t_layers=2, t_dropout=0.5):
        super(ADRNN_new, self).__init__()
        self.hidden_size = hidden_size
        self.response_size = response_size
        self.r_layers = r_layers
        self.r_dropout = r_dropout
        self.treat_size = treat_size
        self.t_layers = t_layers
        self.t_dropout = t_dropout
        self.r_rnn = nn.LSTM(
            input_size=self.response_size + self.treat_size,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.r_layers,       # number of rnn layer
            dropout=self.r_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.t_rnn = nn.LSTM(
            input_size=self.response_size + self.treat_size,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.t_layers,       # number of rnn layer
            dropout=self.t_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.drop_layer = nn.Dropout(0)
        self.linear_r = nn.Linear(self.hidden_size, self.response_size)
        self.linear_t = nn.Linear(self.hidden_size, self.treat_size)


    def forward(self, x_r, x_t, h_r=None, h_t=None,is_train=True, gen_length=None):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_output = torch.zeros(x_r.shape[1], x_r.shape[0], self.response_size).cuda()  #   t b response_size
        t_output = torch.zeros(x_t.shape[1], x_t.shape[0], self.treat_size).cuda()  #   t b treat_size
        batch = x_t.shape[0]
        x_r = x_r.permute(1,0,2)  # -> t b in
        x_t = x_t.permute(1,0,2)  # -> t b in

        if is_train == False:
            r_output = torch.zeros(gen_length, batch, self.response_size).cuda()  #   t b response_size
            t_output = torch.zeros(gen_length, batch, self.treat_size).cuda()  #   t b treat_size
            for i, (r, t) in enumerate(zip(x_r, x_t)):
                r = torch.unsqueeze(r, 0) # -> 1 b in
                t = torch.unsqueeze(t, 0) # -> 1 b in
                r = r.permute(1,0,2) # -> b 1 in
                t = t.permute(1,0,2) # -> b 1 in
                r_input = torch.cat((r,t), 2) # b 1 in+in
                r_out, h_r = self.r_rnn(r_input, h_r)  #r_out = b 1 response_size
                r_out=self.drop_layer(r_out)
                r_out = self.linear_r(r_out)
                t_input = torch.cat((t,r_out), 2) # b 1 in+2response_size
                t_out, h_t = self.t_rnn(t_input, h_t)
                t_out=self.drop_layer(t_out)
                t_out = self.linear_t(t_out)
            r_output[0] = r_out.permute(1,0,2)[0]
            t_output[0] = t_out.permute(1,0,2)[0]
            for i in range(gen_length-1):
                r_input = torch.cat((r_out,t_out), 2) # b 1 in+in
                r_out, h_r = self.r_rnn(r_input, h_r)  #r_out = b 1 response_size
                r_out=self.drop_layer(r_out)
                r_out = self.linear_r(r_out)
                t_input = torch.cat((t_out,r_out), 2) # b 1 in+2response_size
                t_out, h_t = self.t_rnn(t_input, h_t)
                t_out=self.drop_layer(t_out)
                t_out = self.linear_t(t_out)
                r_output[i+1] = r_out.permute(1,0,2)[0]
                t_output[i+1] = t_out.permute(1,0,2)[0]
            return r_output.permute(1,0,2), t_output.permute(1,0,2)

        for i, (r, t) in enumerate(zip(x_r, x_t)):
            r = torch.unsqueeze(r, 0) # -> 1 b in
            t = torch.unsqueeze(t, 0) # -> 1 b in
            r = r.permute(1,0,2) # -> b 1 in
            t = t.permute(1,0,2) # -> b 1 in
            r_input = torch.cat((r,t), 2) # b 1 in+in
            r_out, h_r = self.r_rnn(r_input, h_r)  #r_out = b 1 response_size
            r_out=self.drop_layer(r_out)
            r_out = self.linear_r(r_out)

            t_input = torch.cat((t, r_out), 2)############################### # b 1 in+2response_size
            #if i < len(x_r)-1:
            #    t_input = torch.cat((r_input, torch.unsqueeze(x_r[i+1],1)), 2)
            #else:
            #    t_input = torch.cat((r_input, r_out), 2)
            t_out, h_t = self.t_rnn(t_input, h_t)
            t_out=self.drop_layer(t_out)
            t_out = self.linear_t(t_out)

            r_output[i] = r_out.permute(1,0,2)[0]
            t_output[i] = t_out.permute(1,0,2)[0]

        return r_output.permute(1,0,2), t_output.permute(1,0,2)

class ADRNN_Linear(nn.Module):
    def __init__(self, encoder_hidden = 32, response_size=47, r_layers=2, r_dropout=0.5, treat_size=2, t_layers=2, t_dropout=0.5):
        super(ADRNN_Linear, self).__init__()
        self.encoder_hidden = encoder_hidden
        self.response_size = response_size
        self.r_layers = r_layers
        self.r_dropout = r_dropout
        self.treat_size = treat_size
        self.t_layers = t_layers
        self.t_dropout = t_dropout
        self.r_rnn = nn.LSTM(
            input_size=self.encoder_hidden,
            hidden_size=self.response_size,     # rnn hidden unit
            num_layers=self.r_layers,       # number of rnn layer
            dropout=self.r_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.t_rnn = nn.LSTM(
            input_size=self.encoder_hidden,
            hidden_size=self.treat_size,     # rnn hidden unit
            num_layers=self.t_layers,       # number of rnn layer
            dropout=self.t_dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.drop_layer = nn.Dropout(0)
        self.encoder1 = nn.Linear(self.response_size, self.encoder_hidden)
        self.encoder2 = nn.Linear(self.treat_size, self.encoder_hidden)
        self.encoder3 = nn.Linear(2*self.encoder_hidden, self.encoder_hidden)
        self.encoder4 = nn.Linear(2*self.encoder_hidden, self.encoder_hidden)

    def forward(self, x_r, x_t, h_r=None, h_t=None, is_train=True, gen_length=None):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_output = torch.zeros(x_r.shape[1], x_r.shape[0], self.response_size).cuda()  #   t b response_size
        t_output = torch.zeros(x_t.shape[1], x_t.shape[0], self.treat_size).cuda()  #   t b treat_size
        batch = x_t.shape[0]
        x_r = x_r.permute(1,0,2)  # -> t b in
        x_t = x_t.permute(1,0,2)  # -> t b in

        if is_train == False:
            r_output = torch.zeros(gen_length, batch, self.response_size).cuda()  #   t b response_size
            t_output = torch.zeros(gen_length, batch, self.treat_size).cuda()  #   t b treat_size
            for i, (r, t) in enumerate(zip(x_r, x_t)):
                r = torch.unsqueeze(r, 0) # -> 1 b in
                t = torch.unsqueeze(t, 0) # -> 1 b in
                r = r.permute(1,0,2) # -> b 1 in
                t = t.permute(1,0,2) # -> b 1 in
                r_input_pre = self.encoder1(r)
                t_input = self.encoder2(t)
                r_input_pre=self.drop_layer(r_input_pre)
                t_input=self.drop_layer(t_input)
                input_r = self.encoder3(torch.cat((r_input_pre,t_input),2))
                r_out, h_r = self.r_rnn(input_r, h_r)  #r_out = b 1 response_size
                r_input = self.encoder1(r_out)
                r_input = self.drop_layer(r_input)
                input_t = self.encoder4(torch.cat((input_r,r_input),2))
                t_out, h_t = self.t_rnn(input_t, h_t)
            for i in range(gen_length):
                r_input_pre = self.encoder1(r_out)
                t_input = self.encoder2(t_out)
                r_input_pre=self.drop_layer(r_input_pre)
                t_input=self.drop_layer(t_input)
                input_r = self.encoder3(torch.cat((r_input_pre,t_input),2))
                r_out, h_r = self.r_rnn(input_r, h_r)  #r_out = b 1 response_size
                r_input = self.encoder1(r_out)
                r_input=self.drop_layer(r_input)
                input_t = self.encoder4(torch.cat((input_r,r_input),2))
                t_out, h_t = self.t_rnn(input_t, h_t)   #t_out = b 1 t_size
                r_output[i] = r_out.permute(1,0,2)[0]
                t_output[i] = t_out.permute(1,0,2)[0]
            return r_output.permute(1,0,2), t_output.permute(1,0,2)

        for i, (r, t) in enumerate(zip(x_r, x_t)):
            r = torch.unsqueeze(r, 0) # -> 1 b in
            t = torch.unsqueeze(t, 0) # -> 1 b in
            r = r.permute(1,0,2) # -> b 1 in
            t = t.permute(1,0,2) # -> b 1 in
            r_input_pre = self.encoder1(r)
            t_input = self.encoder2(t)
            r_input_pre=self.drop_layer(r_input_pre)
            t_input=self.drop_layer(t_input)
            input_r = self.encoder3(torch.cat((r_input_pre,t_input),2))
            r_out, h_r = self.r_rnn(input_r, h_r)  #r_out = b 1 response_size
            r_input = self.encoder1(r_out)
            r_input=self.drop_layer(r_input)
            input_t = self.encoder4(torch.cat((t_input,r_input),2))
            t_out, h_t = self.t_rnn(input_t, h_t)
            r_output[i] = r_out.permute(1,0,2)[0]
            t_output[i] = t_out.permute(1,0,2)[0]

        return r_output.permute(1,0,2), t_output.permute(1,0,2)

# In[5]:


class LSTM(nn.Module):
    def __init__(self,hidden_size=64, num_layers=2, dropout=0.2,input_size=49):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.num_layers,       # number of rnn layer
            dropout=self.dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.hidden_size, input_size)

    def forward(self, x, h_state=None, is_train=True, gen_length=None):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        batch = x.shape[0]
        r_out, h_state = self.rnn(x, h_state)
        #return r_out, h_state
        r_out = self.drop_layer(r_out)
        r_out = self.out(r_out) #-> b t inputsize
        if is_train==False:
            r_output = torch.zeros(gen_length, batch, self.input_size).cuda()  #   t b input_size
            r_out = x.permute(1,0,2)[0] #->  b hidden_size
            r_output[0] = r_out
            r_out = r_out.view(batch, 1, -1) #->  b 1 hidden_size
            for i in range(gen_length-1):
                r_out, h_state = self.rnn(r_out, h_state) #r_out  b 1 hidden_size
                r_out = self.drop_layer(r_out)
                r_out = self.out(r_out) #-> b t inputsize
                r_output[i+1]= r_out.permute(1,0,2)[0]
            return r_output.permute(1,0,2), h_state
        return r_out, h_state

# In[6]:


class RNN(nn.Module):
    def __init__(self,hidden_size=64, num_layers=2, dropout=0.2,input_size=49):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.num_layers,       # number of rnn layer
            dropout=self.dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.hidden_size, input_size)

    def forward(self, x, h_state=None, is_train=True, gen_length=None):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        batch = x.shape[0]
        r_out, h_state = self.rnn(x, h_state)
        #return r_out, h_state
        r_out = self.drop_layer(r_out)
        r_out = self.out(r_out) #-> b t inputsize
        if is_train==False:
            r_output = torch.zeros(gen_length, batch, self.input_size).cuda()  #   t b input_size
            r_out = r_out.permute(1,0,2)[-1] #->  b hidden_size
            r_output[0] = r_out
            r_out = r_out.view(batch, 1, -1) #->  b 1 hidden_size
            for i in range(gen_length-1):
                r_out, h_state = self.rnn(r_out, h_state) #r_out  b 1 hidden_size
                r_out = self.drop_layer(r_out)
                r_out = self.out(r_out) #-> b t inputsize
                r_output[i+1]= r_out.permute(1,0,2)[0]
            return r_output.permute(1,0,2), h_state
        return r_out, h_state

class GRU(nn.Module):
    def __init__(self,hidden_size=64, num_layers=2, dropout=0.2,input_size=49):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.num_layers,       # number of rnn layer
            dropout=self.dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.hidden_size, input_size)

    def forward(self, x, h_state=None, is_train=True, gen_length=None):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        batch = x.shape[0]
        r_out, h_state = self.rnn(x, h_state)
        #return r_out, h_state
        r_out = self.drop_layer(r_out)
        r_out = self.out(r_out) #-> b t inputsize
        if is_train==False:
            r_output = torch.zeros(gen_length, batch, self.input_size).cuda()  #   t b input_size
            r_out = r_out.permute(1,0,2)[-1] #->  b hidden_size
            r_output[0] = r_out
            r_out = r_out.view(batch, 1, -1) #->  b 1 hidden_size
            for i in range(gen_length-1):
                r_out, h_state = self.rnn(r_out, h_state) #r_out  b 1 hidden_size
                r_out = self.drop_layer(r_out)
                r_out = self.out(r_out) #-> b t inputsize
                r_output[i+1]= r_out.permute(1,0,2)[0]
            return r_output.permute(1,0,2), h_state
        return r_out, h_state




def evaluate(model, data_x, data_y, mask, loss_fn):  # x (batch, time_step, input_size)
    batch = data_y.shape[0]
    length = data_y.shape[1]
    Pre_y = torch.zeros(length, batch, data_y.shape[2]).cuda()
    Pre_y, _ = model(data_y, is_train=False, gen_length=length) # b t input
    Pre_y = torch.mul(Pre_y,mask)
    eval_loss_r = loss_fn(data_y.permute(2,1,0)[:47], Pre_y.permute(2,1,0)[:47])   # input t b
    eval_loss_t = loss_fn(data_y.permute(2,1,0)[47:], Pre_y.permute(2,1,0)[47:])
    return eval_loss_r, eval_loss_t, data_y, Pre_y  # b t input
def evaluate_rt(model, data_x, data_y, mask, loss_fn):  # x (batch, time_step, input_size)
    batch = data_x.shape[0]
    length = data_y.shape[1]
    Pre_y, _ = model(data_x, is_train=False, gen_length=length) # b t input
    Pre_y = torch.mul(Pre_y,mask)
    eval_loss_r = loss_fn(data_y, Pre_y)
    return eval_loss_r, data_y, Pre_y  # b t input
def evaluate_ADRNN_old(model, data_x_r, data_x_t, data_y, mask, loss_fn):  # x (batch, time_step, input_size)
    batch = data_x_r.shape[0]
    length = data_y.shape[1]
    Pre_y_r = torch.zeros(length, batch, data_x_r.shape[2]).cuda()
    Pre_y_t = torch.zeros(length, batch, data_x_t.shape[2]).cuda()
    for i in range(length):
        y_r, y_t = model(data_x_r, data_x_t)
        data_x_r = torch.cat((data_x_r, y_r.permute(1,0,2)[-1].view(batch, 1, -1)), 1)
        data_x_t = torch.cat((data_x_t, y_t.permute(1,0,2)[-1].view(batch, 1, -1)), 1)
        Pre_y_r[i] = y_r.permute(1,0,2)[-1].view(batch, -1)
        Pre_y_t[i] = y_t.permute(1,0,2)[-1].view(batch, -1)
    Pre_y_r = Pre_y_r.permute(1,0,2)
    Pre_y_t = Pre_y_t.permute(1,0,2)  # -> b t input
    Pre_y = torch.cat((Pre_y_r,Pre_y_t), 2)
    Pre_y = torch.mul(Pre_y,mask)
    eval_loss_r = loss_fn(data_y.permute(2,1,0)[:47], Pre_y.permute(2,1,0)[:47])
    eval_loss_t = loss_fn(data_y.permute(2,1,0)[47:], Pre_y.permute(2,1,0)[47:])
    return eval_loss_r, eval_loss_t
def evaluate_ADRNN(model, data_x_r, data_x_t, data_y, mask, loss_fn):  # x (batch, time_step, input_size)
    batch = data_x_r.shape[0]
    length = data_y.shape[1]
    #Pre_y_r = torch.zeros(length, batch, data_x_r.shape[2]).cuda()
    #Pre_y_t = torch.zeros(length, batch, data_x_t.shape[2]).cuda()
    Pre_y_r, Pre_y_t = model(data_x_r, data_x_t,is_train=False, gen_length=length)
    Pre_y = torch.cat((Pre_y_r,Pre_y_t), 2)
    Pre_y = torch.mul(Pre_y,mask)
    eval_loss_r = loss_fn(data_y.permute(2,1,0)[:47], Pre_y.permute(2,1,0)[:47])
    eval_loss_t = loss_fn(data_y.permute(2,1,0)[47:], Pre_y.permute(2,1,0)[47:])
    return eval_loss_r, eval_loss_t

def Yc_pred2label(Yc1_pred,Yc2_pred): # -> b t 2
    _, Yc1_pred = torch.max(Yc1_pred, 2)  # -> b t
    _, Yc2_pred = torch.max(Yc2_pred, 2)
    return torch.cat((Yc1_pred.unsqueeze(2),Yc2_pred.unsqueeze(2)), 2)  # -> b t 4

def score2Mlabel(x):  # x: b t c  m: b t input
    label = torch.zeros(x.shape[0], x.shape[1], x.shape[2]).cuda()
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            for c in range(x.shape[2]):
                if x[b][t][c] > 0:
                    label[b][t][c] = 1
                else:
                    label[b][t][c] = 0
    return label
def accuracy(ground_truth, Pre_yc, mask):  # P: t  b  input(4)   mask:b t input
    #print(ground_truth.tolist())
    #print(Pre_yc.tolist())
    correct = Pre_yc.eq(ground_truth.view_as(Pre_yc))
    #print(correct.tolist())

    correct = torch.mul(correct.type(torch.cuda.FloatTensor), mask).sum()
    #print(correct.tolist())
    #print()
    base = mask.sum()
    Accu = correct.item() / base.item()
    return Accu
def evaluate_ADRNN_ltCNN(model, C1, C2, data_x_r, data_x_t, data_y, mask, loss_fn):  # x (batch, time_step, input_size)
    batch = data_x_r.shape[0]
    length = data_y.shape[1]
    ground_truth = data_y.permute(2,1,0)[47:] # input(4) t b
    ground_truth = ground_truth.permute(1,2,0)  # t b input(4)
    ground_mask = mask.permute(2,1,0)[47:]
    ground_mask = ground_mask.permute(1,2,0)
    Pre_y_r = torch.zeros(length, batch, data_x_r.shape[2]).cuda()
    Pre_y_t = torch.zeros(length, batch, data_x_t.shape[2]).cuda()
    Pre_yc = torch.zeros(length, batch, ground_truth.shape[2]).cuda()   # t b input(4)
    for i in range(length):
        y_r, y_t = model(data_x_r, data_x_t)    #y_t b t 4
        C_input = torch.cat((data_x_r.permute(1,0,2),data_x_t.permute(1,0,2)),2) # t b input
        C_input = C_input[-13:] # t[-24:] b input
        C_input = C_input.permute(1,0,2) # t[-24:] b input
        Yc1_pred = C1(C_input)  # -> b t 2
        Yc2_pred = C2(C_input)  # -> b t 2
        Yc1_pred = Yc1_pred.max(2, keepdim=True)[1] #F.log_softmax(Yc1_pred)
        Yc2_pred = Yc2_pred.max(2, keepdim=True)[1] #F.log_softmax(Yc2_pred)
        Yc_pred = torch.cat((Yc1_pred,Yc2_pred),2).permute(1,0,2).type(torch.cuda.FloatTensor)  # t b 2
        Yc_pred = Yc_pred[-1]  # b 2
        Pre_yc[i] = Yc_pred
        Yc_pred = Yc_pred.type(torch.FloatTensor).cuda()
        #Yc_pred = torch.cat((Yc_pred.type(torch.FloatTensor).cuda(), torch.ones(Yc_pred.shape[0],Yc_pred.shape[1]).cuda()),1)

        data_x_r = torch.cat((data_x_r, y_r.permute(1,0,2)[-1].view(batch, 1, -1)), 1)
        data_x_t = torch.cat((data_x_t, y_t.permute(1,0,2)[-1].view(batch, 1, -1)), 1)
        Pre_y_r[i] = y_r.permute(1,0,2)[-1].view(batch, -1)
        Pre_y_t[i] = torch.mul(y_t.permute(1,0,2)[-1].view(batch, -1), Yc_pred)
    Pre_y_r = Pre_y_r.permute(1,0,2)
    Pre_y_t = Pre_y_t.permute(1,0,2)  # -> b t input
    Pre_y = torch.cat((Pre_y_r,Pre_y_t), 2)
    Pre_y = torch.mul(Pre_y,mask)
    eval_loss_r = loss_fn(data_y.permute(2,1,0)[:47], Pre_y.permute(2,1,0)[:47])
    eval_loss_t = loss_fn(data_y.permute(2,1,0)[47:], Pre_y.permute(2,1,0)[47:])
    Accu_r = accuracy(ground_truth.permute(2,1,0)[0], Pre_yc.permute(2,1,0)[0], ground_mask.permute(2,1,0)[0])
    Accu_t = accuracy(ground_truth.permute(2,1,0)[1], Pre_yc.permute(2,1,0)[1], ground_mask.permute(2,1,0)[1])
    return eval_loss_r, eval_loss_t, Accu_r, Accu_t
def evaluate_ADRNN_CNN_MultiLabel(model, C1, data_x_r, data_x_t, data_y, mask, loss_fn):  # x (batch, time_step, input_size)
    batch = data_x_r.shape[0]
    length = data_y.shape[1]
    ground_truth = data_y.permute(2,1,0)[47:] # input(4) t b
    ground_truth = ground_truth.permute(1,2,0)  # t b input(4)
    ground_mask = mask.permute(2,1,0)[47:]
    ground_mask = ground_mask.permute(1,2,0)
    Pre_y_r = torch.zeros(length, batch, data_x_r.shape[2]).cuda()
    Pre_y_t = torch.zeros(length, batch, data_x_t.shape[2]).cuda()
    Pre_yc = torch.zeros(length, batch, ground_truth.shape[2]).cuda()   # t b input(4)
    for i in range(length):
        y_r, y_t = model(data_x_r, data_x_t,is_train=False, gen_length=length)    #y_t b t 4
        C_input = torch.cat((data_x_r.permute(1,0,2),data_x_t.permute(1,0,2)),2) # t b input
        C_input = C_input[-13:] # t[-24:] b input
        C_input = C_input.permute(1,0,2) # t[-24:] b input
        Yc1_pred = C1(C_input)  # -> b t c
        Yc1_pred = F.sigmoid(Yc1_pred)
        Yc_pred = (Yc1_pred>0.5).mul_(1)
        Yc_pred = Yc_pred.permute(1,0,2)  # t b c
        Yc_pred = Yc_pred[-1]  # b 4
        Pre_yc[i] = Yc_pred
        #Yc_pred = torch.cat((Yc_pred.type(torch.FloatTensor).cuda(), torch.ones(Yc_pred.shape[0],Yc_pred.shape[1]).cuda()),1)

        data_x_r = torch.cat((data_x_r, y_r.permute(1,0,2)[-1].view(batch, 1, -1)), 1)
        data_x_t = torch.cat((data_x_t, y_t.permute(1,0,2)[-1].view(batch, 1, -1)), 1)
        Pre_y_r[i] = y_r.permute(1,0,2)[-1].view(batch, -1)
        Pre_y_t[i] = torch.mul(y_t.permute(1,0,2)[-1].view(batch, -1), Yc_pred.type(torch.FloatTensor).cuda())
    Pre_y_r = Pre_y_r.permute(1,0,2)
    Pre_y_t = Pre_y_t.permute(1,0,2)  # -> b t input
    Pre_y = torch.cat((Pre_y_r,Pre_y_t), 2)
    Pre_y = torch.mul(Pre_y,mask)
    eval_loss_r = loss_fn(data_y.permute(2,1,0)[:47], Pre_y.permute(2,1,0)[:47])
    eval_loss_t = loss_fn(data_y.permute(2,1,0)[47:], Pre_y.permute(2,1,0)[47:])
    Accu_r = accuracy(ground_truth.permute(2,1,0)[0], Pre_yc.permute(2,1,0)[0], ground_mask.permute(2,1,0)[0])
    Accu_t = accuracy(ground_truth.permute(2,1,0)[1], Pre_yc.permute(2,1,0)[1], ground_mask.permute(2,1,0)[1])
    return eval_loss_r, eval_loss_t, Accu_r, Accu_t

def evaluate_ADRNN_to_CNN_MultiLabel(model, C1, data_x_r, data_x_t, data_y, mask, loss_fn):  # x (batch, time_step, input_size)
    batch = data_x_r.shape[0]
    length = data_y.shape[1]
    ground_truth = data_y.permute(2,1,0)[47:] # input(2) t b
    ground_truth = ground_truth.permute(1,2,0)  # t b input(2)
    ground_truth = score2Mlabel(ground_truth)
    ground_mask = mask.permute(2,1,0)[47:]
    ground_mask = ground_mask.permute(1,2,0)

    Pre_y_r, Pre_y_t = model(data_x_r, data_x_t,is_train=False, gen_length=length)
    Pre_y = torch.cat((Pre_y_r,Pre_y_t), 2)

    C_input = torch.cat((data_x_r.permute(1,0,2),data_x_t.permute(1,0,2)),2) # t b input
    C_input = torch.cat((C_input,Pre_y.permute(1,0,2)), 0)  # t b input
    C_input = C_input[-(length+12):-1] # t[-24:] b input
    C_input = C_input.permute(1,0,2) # t[-24:] b input
    Yc1_pred = C1(C_input)  # -> b t c
    Yc1_pred = F.sigmoid(Yc1_pred)
    Yc_pred = (Yc1_pred>0.2).mul_(1)
    Yc_pred = Yc_pred.type(torch.FloatTensor).cuda()  # b t c

    Accu_r = accuracy(ground_truth.permute(2,1,0)[0], Yc_pred.permute(2,0,1)[0], ground_mask.permute(2,1,0)[0])
    Accu_t = accuracy(ground_truth.permute(2,1,0)[1], Yc_pred.permute(2,0,1)[1], ground_mask.permute(2,1,0)[1])

    Pre_y = torch.mul(Pre_y,mask)
    eval_loss_r = loss_fn(data_y.permute(2,1,0)[:47], Pre_y.permute(2,1,0)[:47])
    eval_loss_t_old = loss_fn(data_y.permute(2,1,0)[47:], Pre_y.permute(2,1,0)[47:])
    #print("eval_loss_t_old" + str(eval_loss_t_old.item()))
    p = Pre_y.permute(2,1,0)[47:]
    #print(p.tolist())
    Pre_y = torch.mul(Pre_y.permute(2,1,0)[47:],Yc_pred.permute(2,1,0))
    eval_loss_t = loss_fn(data_y.permute(2,1,0)[47:], Pre_y)
    #print("eval_loss_t" + str(eval_loss_t.item()))
    #print(Pre_y.tolist())
    #print(data_y.permute(2,1,0)[47:].tolist())

    #print()
    return eval_loss_r, eval_loss_t, Accu_r, Accu_t


# In[8]:
def RMSE(Y, Y_pred):  # b t input
    num = 0
    for y in Y:
        num += y.shape[0]   #batch number
    ground_truth = Y[0]
    predict = Y_pred[0]
    for i, y in enumerate(Y[1:]):
        ground_truth = torch.cat((ground_truth, y),0)
        predict = torch.cat((predict, Y_pred[i+1]),0)
    g_r = ground_truth.permute(2,1,0)[:47]
    g_t = ground_truth.permute(2,1,0)[47:]
    y_r = predict.permute(2,1,0)[:47]
    y_t = predict.permute(2,1,0)[47:]
    g_r = torch.unsqueeze(g_r, 0)
    g_t = torch.unsqueeze(g_t, 0)
    y_r = torch.unsqueeze(y_r, 0)
    y_t = torch.unsqueeze(y_t, 0)
    l2_r = F.mse_loss(g_r,y_r)
    l2_t = F.mse_loss(g_t,y_t)
    l1_r = F.l1_loss(g_r,y_r)
    l1_t = F.l1_loss(g_t,y_t)
    return l1_r.item(), l1_t.item(), l2_r.item(), l2_t.item()


def run_RNN(iterator, model, path):
    if torch.cuda.is_available():
        model=model.cuda()
    print(str(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0005)   # optimize all cnn parameters
    loss_fn = nn.MSELoss()
    h_state = None # for initial hidden state
    num_epochs = 50
    #####################
    # Train model
    #####################

    file_w = codecs.open(path+'.txt', "w", "utf-8")
    file_w.write(str(model))

    for t in range(num_epochs):
        train_loss = 0
        train_loss_r = 0
        train_loss_t = 0
        loss = 0
        model = model.train()
        for b, batch in enumerate(iterator):
            batch, mask, _, _, _, _ = batch  # batch = (batch, time_step, input_size)


            batch = batch.permute(1,0,2)  # batch = (t, b, input)
            mask = mask.permute(1,0,2)
            X_train = batch[:-1]
            X_train = X_train.permute(1,0,2) # > b t input
            Y_train = batch[1:]
            Y_train = Y_train.permute(1,0,2) # > b t input
            mask = mask[1:]
            mask = mask.permute(1,0,2) # > b t input

            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            #model.hidden = model.init_hidden()

            # Forward pass
            if torch.cuda.is_available():
                X_train = X_train.cuda()
                mask = mask.cuda()
                Y_train = Y_train.cuda()
            Y_pred,_ = model(X_train) # X_train = (batch, time_step, input_size)
            Y_pred = torch.mul(Y_pred,mask)

            loss_r = loss_fn(Y_pred.permute(2,1,0)[:47], Y_train.permute(2,1,0)[:47])
            loss_t = loss_fn(Y_pred.permute(2,1,0)[47:], Y_train.permute(2,1,0)[47:])
            loss = loss_r + loss_t
            train_loss += loss.item()
            train_loss_r += loss_r.item()
            train_loss_t += loss_t.item()
            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
        print("Epoch ", t, "  train_loss: ", train_loss, "  train_loss_r: ", train_loss_r, "  train_loss_t: ", train_loss_t)
        file_w.write("Epoch "+str(t)+ "  train_loss: "+str(train_loss)+ "  train_loss_r: "+str( train_loss_r)+ "  train_loss_t: "+str( train_loss_t) +'\n')

        ##evaluation
        model = model.eval()
        with torch.set_grad_enabled(False):
            Y = []
            Y_pred = []
            for b, batch in enumerate(iterator):
                data_x, _, data_y, mask, _, _  = batch
                if torch.cuda.is_available():
                    data_x = data_x.cuda()
                    data_y = data_y.cuda()
                    mask = mask.cuda()
                eval_loss_r, eval_loss_t, y, y_pred = evaluate(model, data_x, data_y, mask, loss_fn)
                Y.append(y)
                Y_pred.append(y_pred)
            l1_r,l1_t,l2_r,l2_t = RMSE(Y, Y_pred)
            print("Epoch ", t,"  dev_loss_all: ", str(l1_r+l1_t+l2_r+l2_t),  "  valid_l1_r: ", l1_r, "  valid_l1_t: ", l1_t,  "  valid_l2_r: ", l2_r, "  valid_l2_t: ", l2_t)
            file_w.write("Epoch "+str(t)+"  dev_loss_all: "+str(l1_r+l1_t+l2_r+l2_t)+ "  valid_l1_r: "+str(l1_r)+ "  valid_l1_t: "+str(l1_t)+"  valid_l2_r: "+str(l2_r)+ "  valid_l2_t: "+str( l2_t)+"\n")

            ##test
            Y = []
            Y_pred = []
            for b, batch in enumerate(iterator):
                data_x, _, data_y, _, data_z, mask  = batch
                data_xy = torch.cat((data_x,data_y), 1)
                if torch.cuda.is_available():
                    data_xy = data_xy.cuda()
                    data_z = data_z.cuda()
                    mask = mask.cuda()
                eval_loss_r, eval_loss_t, y, y_pred = evaluate(model, data_xy, data_z, mask, loss_fn)
                Y.append(y)
                Y_pred.append(y_pred)
            l1_r,l1_t,l2_r,l2_t = RMSE(Y, Y_pred)
            print("Epoch ", t,"  test_loss_all: ", str(l1_r+l1_t+l2_r+l2_t),  "  test_l1_r: ", l1_r, "  test_l1_t: ", l1_t,  "  test_l2_r: ", l2_r, "  test_l2_t: ", l2_t)
            file_w.write("Epoch "+str(t)+"  test_loss_all: "+str(l1_r+l1_t+l2_r+l2_t)+ "  test_l1_r: "+str( l1_r)+ "  test_l1_t: "+str( l1_t)+  "  test_l2_r: "+str( l2_r)+ "  test_l2_t: "+str( l2_t)+"\n")
    file_w.close()

    return model

def run_RNN_rt(train_iterator,vaild_iterator,test_iterator, model_r, model_t, path):
    if torch.cuda.is_available():
        model_r=model_r.cuda()
        model_t=model_t.cuda()
    print(model_r)
    print(model_t)

    optimizer = torch.optim.Adam([
                {'params': model_r.parameters()},
                {'params': model_t.parameters()},], lr=0.0002, weight_decay=0.0005)   # optimize all cnn parameters
    loss_fn = nn.MSELoss()

    h_state = None # for initial hidden state
    num_epochs = 50

    #####################
    # Train model
    #####################
    file_w = codecs.open(path+'.txt', "w", "utf-8")
    file_w.write(str(model_r)+"\n")
    file_w.write(str(model_t)+"\n")

    for t in range(num_epochs):
        train_loss = 0
        train_loss_r = 0
        train_loss_t = 0
        model_r = model_r.train()
        model_t = model_t.train()
        for b, batch in enumerate(train_iterator):
            batch, mask= batch # batch = (b, t, input)
            batch = batch.permute(1,0,2)  # batch = (t, b, input)
            mask = mask.permute(1,0,2)
            X_train = batch[:-1]
            X_train = X_train.permute(1,0,2)
            X_train_r = X_train.permute(2,1,0)[:47] # input[:51] t b
            X_train_t = X_train.permute(2,1,0)[47:] # input[51:] t b
            X_train_r = X_train_r.permute(2,1,0)  # b t input[:51]
            X_train_t = X_train_t.permute(2,1,0)  # b t input[51:]

            Y_train = batch[1:]
            Y_train = Y_train.permute(1,0,2)
            Y_train_r = Y_train.permute(2,1,0)[:47]
            Y_train_t = Y_train.permute(2,1,0)[47:]
            Y_train_r = Y_train_r.permute(2,1,0)
            Y_train_t = Y_train_t.permute(2,1,0)
            mask = mask[1:]  # t b input
            mask = mask.permute(1,0,2)   #b t input
            mask_r = mask.permute(2,1,0)[:47]   #input t b
            mask_r = mask_r.permute(2,1,0)
            mask_t = mask.permute(2,1,0)[47:]   #input t b
            mask_t = mask_t.permute(2,1,0)

            # Forward pass
            if torch.cuda.is_available():
                X_train_r = X_train_r.cuda()
                X_train_t = X_train_t.cuda()
                mask_t = mask_t.cuda()
                mask_r = mask_r.cuda()
                Y_train_r = Y_train_r.cuda()
                Y_train_t = Y_train_t.cuda()
            Y_pred_r,_ = model_r(X_train_r)
            Y_pred_t,_ = model_t(X_train_t)
            Y_pred_r = torch.mul(Y_pred_r,mask_r)
            Y_pred_t = torch.mul(Y_pred_t,mask_t)

            loss_r = loss_fn(Y_pred_r, Y_train_r)
            loss_t = loss_fn(Y_pred_t, Y_train_t)
            loss = loss_r + loss_t
            train_loss = train_loss + loss_r.item() + loss_t.item()
            train_loss_r += loss_r.item()
            train_loss_t += loss_t.item()

            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
        print("Epoch ", t, "  train_loss: ", train_loss, "  train_loss_r: ", train_loss_r, "  train_loss_t: ", train_loss_t)
        file_w.write("Epoch "+str(t)+ "  train_loss: "+str(train_loss)+ "  train_loss_r: "+str( train_loss_r)+ "  train_loss_t: "+str( train_loss_t) +'\n')
        ##evaluation
        valid_loss_r = 0
        valid_loss_t = 0
        model_r = model_r.eval()
        model_t = model_t.eval()
        with torch.set_grad_enabled(False):
            Y = []
            Y_pred = []
            for b, batch in enumerate(vaild_iterator):
                data_y, mask = batch
                if torch.cuda.is_available():
                    # data_x = data_x.cuda()
                    data_y = data_y.cuda()
                    mask = mask.cuda()
                # data_x_r = data_x.permute(2,1,0)[:47]
                # data_x_r = data_x_r.permute(2,1,0)
                # data_x_t = data_x.permute(2,1,0)[47:]
                # data_x_t = data_x_t.permute(2,1,0)
                data_y_r = data_y.permute(2,1,0)[:47]
                data_y_r = data_y_r.permute(2,1,0)
                data_y_t = data_y.permute(2,1,0)[47:]
                data_y_t = data_y_t.permute(2,1,0)
                mask_r = mask.permute(2,1,0)[:47]
                mask_r = mask_r.permute(2,1,0)
                mask_t = mask.permute(2,1,0)[47:]
                mask_t = mask_t.permute(2,1,0)
                eval_loss_r, yr, yr_pred = evaluate_rt(model_r, data_y_r, data_y_r, mask_r, loss_fn)
                eval_loss_t, yt, yt_pred = evaluate_rt(model_t, data_y_t, data_y_t, mask_t, loss_fn)

                Y.append(torch.cat((yr,yt),2))
                Y_pred.append(torch.cat((yr_pred, yt_pred),2))
            l1_r,l1_t,l2_r,l2_t = RMSE(Y, Y_pred)
            print("Epoch ", t,"  dev_loss_all: ", str(l1_r+l1_t+l2_r+l2_t),  "  valid_l1_r: ", l1_r, "  valid_l1_t: ", l1_t,  "  valid_l2_r: ", l2_r, "  valid_l2_t: ", l2_t)
            file_w.write("Epoch "+str(t)+"  dev_loss_all: "+str(l1_r+l1_t+l2_r+l2_t)+ "  valid_l1_r: "+str(l1_r)+ "  valid_l1_t: "+str(l1_t)+"  valid_l2_r: "+str(l2_r)+ "  valid_l2_t: "+str( l2_t)+"\n")
            ##test
            Y = []
            Y_pred = []
            for b, batch in enumerate(test_iterator):
                data_z, mask  = batch
                # data_xy = torch.cat((data_x,data_y), 1)
                if torch.cuda.is_available():
                    # data_xy = data_xy.cuda()
                    data_z = data_z.cuda()
                    mask = mask.cuda()
                # data_xy_r = data_xy.permute(2,1,0)[:47]
                # data_xy_r = data_xy_r.permute(2,1,0)
                # data_xy_t = data_xy.permute(2,1,0)[47:]
                # data_xy_t = data_xy_t.permute(2,1,0)
                data_z_r = data_z.permute(2,1,0)[:47]
                data_z_r = data_z_r.permute(2,1,0)
                data_z_t = data_z.permute(2,1,0)[47:]
                data_z_t = data_z_t.permute(2,1,0)
                mask_r = mask.permute(2,1,0)[:47]
                mask_r = mask_r.permute(2,1,0)
                mask_t = mask.permute(2,1,0)[47:]
                mask_t = mask_t.permute(2,1,0)
                eval_loss_r, yr, yr_pred = evaluate_rt(model_r, data_z_r, data_z_r, mask_r, loss_fn)
                eval_loss_t, yt, yt_pred = evaluate_rt(model_t, data_z_t, data_z_t, mask_t, loss_fn)
                Y.append(torch.cat((yr,yt),2))
                Y_pred.append(torch.cat((yr_pred, yt_pred),2))
            l1_r,l1_t,l2_r,l2_t = RMSE(Y, Y_pred)

            print("Epoch ", t,"  test_loss_all: ", str(l1_r+l1_t+l2_r+l2_t),  "  test_l1_r: ", l1_r, "  test_l1_t: ", l1_t,  "  test_l2_r: ", l2_r, "  test_l2_t: ", l2_t)
            file_w.write("Epoch "+str(t)+"  test_loss_all: "+str(l1_r+l1_t+l2_r+l2_t)+ "  test_l1_r: "+str( l1_r)+ "  test_l1_t: "+str( l1_t)+  "  test_l2_r: "+str( l2_r)+ "  test_l2_t: "+str( l2_t)+"\n")
    file_w.close()
    return model_r, model_t


def run_ADRNN(iterator, model):
    if torch.cuda.is_available():
        model=model
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0005)   # optimize all cnn parameters
    loss_fn = nn.MSELoss()

    h_state = None # for initial hidden state
    num_epochs = 50
    #RNN_Vis = Visdom(env='RNN')
    #RNN_Vis.line(X=batch_num,Y=AlexNet_Train_Acc,update='append',win='lenet train acc vs batch')

    #####################
    # Train model
    #####################

    for t in range(num_epochs):
        train_loss = 0
        train_loss_r = 0
        train_loss_t = 0
        model = model.train()
        for b, batch in enumerate(iterator):
            batch, mask, _, _, _, _ = batch # batch = (b, t, input)
            batch = batch.permute(1,0,2)  # batch = (t, b, input)
            mask = mask.permute(1,0,2)
            X_train = batch[:-1]
            X_train = X_train.permute(1,0,2)
            X_train_r = X_train.permute(2,1,0)[:47] # input[:51] t b
            X_train_t = X_train.permute(2,1,0)[47:] # input[51:] t b
            X_train_r = X_train_r.permute(2,1,0)  # b t input[:51]
            X_train_t = X_train_t.permute(2,1,0)  # b t input[51:]

            Y_train = batch[1:]
            Y_train = Y_train.permute(1,0,2)
            mask = mask[1:]
            mask = mask.permute(1,0,2)

            # Forward pass
            if torch.cuda.is_available():
                X_train_r = X_train_r.cuda()
                X_train_t = X_train_t.cuda()
                mask = mask.cuda()
                Y_train = Y_train.cuda()
            Y_pred_r,Y_pred_t = model(X_train_r, X_train_t)  # X_train = (batch, time_step, input_size)   Y_p_r = b t input[:51]    Y_p_t = b t input[51:]
            Y_pred = torch.cat((Y_pred_r, Y_pred_t),2)  # b t input
            Y_pred = torch.mul(Y_pred,mask)

            loss_r = loss_fn(Y_pred.permute(2,1,0)[:47], Y_train.permute(2,1,0)[:47])
            loss_t = loss_fn(Y_pred.permute(2,1,0)[47:], Y_train.permute(2,1,0)[47:])
            loss = loss_r + loss_t

            train_loss = train_loss + loss.item()
            train_loss_r += loss_r.item()
            train_loss_t += loss_t.item()
            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
        print("Epoch ", t, "  train_loss: ", train_loss, "  train_loss_r: ", train_loss_r, "  train_loss_t: ", train_loss_t)


        ##evaluation
        valid_loss_r = 0
        valid_loss_t = 0
        model = model.eval()
        with torch.set_grad_enabled(False):
            for b, batch in enumerate(iterator):
                data_x, _, data_y, mask, _, _  = batch

                data_x_r = data_x.permute(2,1,0)[:47] # input[:51] t b
                data_x_t = data_x.permute(2,1,0)[47:] # input[51:] t b
                data_x_r = data_x_r.permute(2,1,0)  # b t input[:51]
                data_x_t = data_x_t.permute(2,1,0)  # b t input[51:]
                if torch.cuda.is_available():
                    data_x_r = data_x_r.cuda()
                    data_x_t = data_x_t.cuda()
                    data_y = data_y.cuda()
                    mask = mask.cuda()
                eval_loss_r, eval_loss_t = evaluate_ADRNN(model, data_x_r, data_x_t, data_y, mask, loss_fn)
                valid_loss_r += eval_loss_r.item()
                valid_loss_t += eval_loss_t.item()
            print("Epoch ", t, "  valid_loss_r: ", valid_loss_r, "  valid_loss_t: ", valid_loss_t)

            ##test
            test_loss_r = 0
            test_loss_t = 0
            for b, batch in enumerate(iterator):
                data_x, _, data_y, _, data_z, mask  = batch
                data_xy = torch.cat((data_x,data_y), 1)

                data_xy_r = data_xy.permute(2,1,0)[:47] # input[:51] t b
                data_xy_t = data_xy.permute(2,1,0)[47:] # input[51:] t b
                data_xy_r = data_xy_r.permute(2,1,0)  # b t input[:51]
                data_xy_t = data_xy_t.permute(2,1,0)  # b t input[51:]
                if torch.cuda.is_available():
                    data_xy_r = data_xy_r.cuda()
                    data_xy_t = data_xy_t.cuda()
                    data_z = data_z.cuda()
                    mask = mask.cuda()
                eval_loss_r, eval_loss_t  = evaluate_ADRNN(model, data_xy_r, data_xy_t, data_z, mask, loss_fn)
                test_loss_r += eval_loss_r.item()
                test_loss_t += eval_loss_t.item()
            print("Epoch ", t, "  test_loss_r: ", test_loss_r, "  test_loss_t: ", test_loss_t)
            test_loss = 0

    return model


def score2label(x):  # x: b t   m: b t input
    label = torch.LongTensor(x.shape[0], x.shape[1]).cuda()
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            if x[b][t] > 0:
                label[b][t] = 1
            else:
                label[b][t] = 0
    return label


def run_ADRNN_CNN(iterator, model, C1, C2,path=None):
    if torch.cuda.is_available():
        model=model
    print(model)
    print(C1)
    print(C2)

    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': C1.parameters()},
                {'params': C2.parameters()},
            ], lr=0.0002, weight_decay=0.0005)   # optimize all cnn parameters
    loss_fn = nn.MSELoss()
    loss_ce =nn.CrossEntropyLoss()

    h_state = None # for initial hidden state
    num_epochs = 50

    for t in range(num_epochs):
        train_loss = 0
        train_loss_r = 0
        train_loss_t = 0
        train_loss_t1 = 0
        train_loss_t2 = 0
        for b, batch in enumerate(iterator):
            #print('batch: ' + str(b))
            batch, mask, _, _, _, _ = batch # batch = (b, t, input)
            batch = batch.permute(1,0,2)  # batch = (t, b, input)
            mask = mask.permute(1,0,2)   # t b input
            X_train = batch[:-1]
            X_train = X_train.permute(1,0,2)
            X_train_r = X_train.permute(2,1,0)[:47] # input[:51] t b
            X_train_t = X_train.permute(2,1,0)[47:] # input[51:] t b
            X_train_r = X_train_r.permute(2,1,0)  # b t input[:51]
            X_train_t = X_train_t.permute(2,1,0)  # b t input[51:]

            Y_train = batch[1:]
            Y_train = Y_train.permute(1,0,2)
            mask = mask[1:]
            mask = mask.permute(1,0,2)  # b t input

            # Forward pass
            if torch.cuda.is_available():
                X_train_r = X_train_r.cuda()
                X_train_t = X_train_t.cuda()
                mask = mask.cuda()
                Y_train = Y_train.cuda()
            Y_pred_r,Y_pred_t = model(X_train_r, X_train_t)  # X_train = (batch, time_step, input_size)   Y_p_r = b t input[:51]    Y_p_t = b t input[51:]
            Y_pred = torch.cat((Y_pred_r, Y_pred_t),2)  # b t input
            Y_pred = torch.mul(Y_pred,mask)
            loss_r = loss_fn(Y_pred.permute(2,1,0)[:47], Y_train.permute(2,1,0)[:47])
            loss_t = loss_fn(Y_pred.permute(2,1,0)[47:], Y_train.permute(2,1,0)[47:])

            Yc1_pred = C1(torch.cat((X_train_r,X_train_t),2))  # -> b t 2
            mask = mask.permute(1,0,2)[12:]
            mask = mask.permute(1,0,2)
            mask_c1 = mask.permute(2,0,1)[47].unsqueeze(2) # b t 1
            mask_c1 = mask_c1.expand(mask_c1.shape[0],mask_c1.shape[1],2)
            Yc1_pred = torch.mul(Yc1_pred,mask_c1)
            Yc2_pred = C2(torch.cat((X_train_r,X_train_t),2))  # -> b t 2
            mask_c2 = mask.permute(2,0,1)[48].unsqueeze(2) # b t 1
            mask_c2 = mask_c2.expand(mask_c2.shape[0],mask_c2.shape[1],2)
            Yc2_pred = torch.mul(Yc2_pred,mask_c2)

            Target = batch[12:-1]  #-> t[24:]  b input
            Target = Target.permute(2,1,0)[47:] #-> input(2) b t[24:]
            Target1 = score2label(Target[0].cuda()) # b t
            Target2 = score2label(Target[1].cuda())


            loss_t1 = loss_ce(Yc1_pred.contiguous().view(-1,2), Target1.view(-1))
            loss_t2 = loss_ce(Yc2_pred.contiguous().view(-1,2), Target2.view(-1))
            loss = loss_r + loss_t + loss_t1 + loss_t2
            train_loss = train_loss + loss.item()
            train_loss_r += loss_r.item()
            train_loss_t += loss_t.item()
            train_loss_t1 += loss_t1.item()
            train_loss_t2 += loss_t2.item()
            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

        print("Epoch ", t, "  train_loss: ", train_loss, "  train_loss_r: ", train_loss_r, "  train_loss_t: ", train_loss_t)
        print("Epoch ", t, "  train_loss_t1: ", train_loss_t1, "  train_loss_t2: ", train_loss_t2)


        ##evaluation
        eval_loss = 0
        valid_loss_r = 0
        valid_loss_t = 0
        valid_accu_0 = []
        valid_accu_1 = []
        for b, batch in enumerate(iterator):
            data_x, _, data_y, mask, _, _  = batch

            data_x_r = data_x.permute(2,1,0)[:47] # input[:51] t b
            data_x_t = data_x.permute(2,1,0)[47:] # input[51:] t b
            data_x_r = data_x_r.permute(2,1,0)  # b t input[:51]
            data_x_t = data_x_t.permute(2,1,0)  # b t input[51:]
            if torch.cuda.is_available():
                data_x_r = data_x_r.cuda()
                data_x_t = data_x_t.cuda()
                data_y = data_y.cuda()
                mask = mask.cuda()
            eval_loss_r, eval_loss_t, accu_0, accu_1  = evaluate_ADRNN_ltCNN(model, C1, C2, data_x_r, data_x_t, data_y, mask, loss_fn)
            valid_loss_r += eval_loss_r.item()
            valid_loss_t += eval_loss_t.item()
            valid_accu_0.append(accu_0)
            valid_accu_1.append(accu_1)
        print("Epoch ", t, "  valid_loss_r: ", valid_loss_r, "  valid_loss_t: ", valid_loss_t, " valid_accu_0: ", sum(valid_accu_0)/len(valid_accu_0), " valid_accu_1: ", sum(valid_accu_1)/len(valid_accu_1))

        ##test
        test_loss_r = 0
        test_loss_t = 0
        test_accu_0 = []
        test_accu_1 = []
        for b, batch in enumerate(iterator):
            data_x, _, data_y, _, data_z, mask  = batch
            data_xy = torch.cat((data_x,data_y), 1)

            data_xy_r = data_xy.permute(2,1,0)[:47] # input[:51] t b
            data_xy_t = data_xy.permute(2,1,0)[47:] # input[51:] t b
            data_xy_r = data_xy_r.permute(2,1,0)  # b t input[:51]
            data_xy_t = data_xy_t.permute(2,1,0)  # b t input[51:]
            if torch.cuda.is_available():
                data_xy_r = data_xy_r.cuda()
                data_xy_t = data_xy_t.cuda()
                data_z = data_z.cuda()
                mask = mask.cuda()
            eval_loss_r, eval_loss_t, accu_0, accu_1 = evaluate_ADRNN_ltCNN(model, C1, C2, data_xy_r, data_xy_t, data_z, mask, loss_fn)
            test_loss_r += eval_loss_r.item()
            test_loss_t += eval_loss_t.item()
            test_accu_0.append(accu_0)
            test_accu_1.append(accu_1)
        print("Epoch ", t, "  test_loss_r: ", test_loss_r, "  test_loss_t: ", test_loss_t, " test_accu_0: ", sum(test_accu_0)/len(test_accu_0), " test_accu_1: ", sum(test_accu_1)/len(test_accu_1))

    return model, C1, C2

def run_ADRNN_CNN_MultiLabel(iterator, model, C1,path=None):
    if torch.cuda.is_available():
        model=model
    #print(model)
    print(C1)

    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': C1.parameters()},
            ], lr=0.0002, weight_decay=0.0005)   # optimize all cnn parameters
    loss_fn = nn.MSELoss()
    loss_ce =nn.BCELoss()

    h_state = None # for initial hidden state
    num_epochs = 50

    for t in range(num_epochs):
        train_loss = 0
        train_loss_r = 0
        train_loss_t = 0
        train_loss_t1 = 0
        train_loss_t2 = 0
        for b, batch in enumerate(iterator):
            #print('batch: ' + str(b))
            batch, mask, _, _, _, _ = batch # batch = (b, t, input)
            batch = batch.permute(1,0,2)  # batch = (t, b, input)
            mask = mask.permute(1,0,2)   # t b input
            X_train = batch[:-1]
            X_train = X_train.permute(1,0,2)
            X_train_r = X_train.permute(2,1,0)[:47] # input[:51] t b
            X_train_t = X_train.permute(2,1,0)[47:] # input[51:] t b
            X_train_r = X_train_r.permute(2,1,0)  # b t input[:51]
            X_train_t = X_train_t.permute(2,1,0)  # b t input[51:]

            Y_train = batch[1:]
            Y_train = Y_train.permute(1,0,2)
            mask = mask[1:]
            mask = mask.permute(1,0,2)  # b t input

            # Forward pass
            if torch.cuda.is_available():
                X_train_r = X_train_r.cuda()
                X_train_t = X_train_t.cuda()
                mask = mask.cuda()
                Y_train = Y_train.cuda()
            Y_pred_r,Y_pred_t = model(X_train_r, X_train_t)  # X_train = (batch, time_step, input_size)   Y_p_r = b t input[:51]    Y_p_t = b t input[51:]
            Y_pred = torch.cat((Y_pred_r, Y_pred_t),2)  # b t input
            Y_pred = torch.mul(Y_pred,mask)
            loss_r = loss_fn(Y_pred.permute(2,1,0)[:47], Y_train.permute(2,1,0)[:47])
            loss_t = loss_fn(Y_pred.permute(2,1,0)[47:], Y_train.permute(2,1,0)[47:])

            Yc1_pred = C1(torch.cat((X_train_r,X_train_t),2))  # -> b t c
            Yc1_pred = F.sigmoid(Yc1_pred)
            Yc1_pred = Yc1_pred.contiguous().view(-1,2)

            mask_c = mask.permute(1,0,2)[12:]  # b t input -> t b input
            mask_c = mask_c.permute(2,1,0)[-2:]  #   input b t
            mask_c = mask_c.permute(1,2,0) # b t c
            Yc1_pred = torch.mul(Yc1_pred, mask_c.contiguous().view(-1,2)) # -1 c

            Target = batch[12:-1]  #-> t[24:]  b input
            Target = Target.permute(2,1,0)[47:] #-> input(2) b t[24:]
            Target = Target.permute(1,2,0) #-> b t[24:] c
            Target = score2Mlabel(Target.cuda()) # b t[24:] c
            Target = Target.view(-1,2) # -1 c

            #loss_t1 = loss_ce(Yc1_pred.view(2,-1)[0], Target.view(2,-1)[0])
            #loss_t2 = loss_ce(Yc1_pred.view(2,-1)[1], Target.view(2,-1)[1])
            loss_t1 = loss_ce(Yc1_pred, Target)

            loss = loss_r + loss_t + loss_t1 # + loss_t2
            train_loss = train_loss + loss.item()
            train_loss_r += loss_r.item()
            train_loss_t += loss_t.item()
            train_loss_t1 += loss_t1.item()
            #train_loss_t2 += loss_t2.item()

            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

        print("Epoch ", t, "  train_loss: ", train_loss, "  train_loss_r: ", train_loss_r, "  train_loss_t: ", train_loss_t)
        print("Epoch ", t, "  train_loss_t1: ", train_loss_t1, "  train_loss_t2: ", train_loss_t2)


        ##evaluation
        eval_loss = 0
        valid_loss_r = 0
        valid_loss_t = 0
        valid_accu_0 = 0
        valid_accu_1 = 0
        for b, batch in enumerate(iterator):
            data_x, _, data_y, mask, _, _  = batch

            data_x_r = data_x.permute(2,1,0)[:47] # input[:51] t b
            data_x_t = data_x.permute(2,1,0)[47:] # input[51:] t b
            data_x_r = data_x_r.permute(2,1,0)  # b t input[:51]
            data_x_t = data_x_t.permute(2,1,0)  # b t input[51:]
            if torch.cuda.is_available():
                data_x_r = data_x_r.cuda()
                data_x_t = data_x_t.cuda()
                data_y = data_y.cuda()
                mask = mask.cuda()
            eval_loss_r, eval_loss_t, accu_0, accu_1 = evaluate_ADRNN_CNN_MultiLabel(model, C1, data_x_r, data_x_t, data_y, mask, loss_fn)
            valid_loss_r += eval_loss_r.item()
            valid_loss_t += eval_loss_t.item()
            valid_accu_0 += accu_0
            valid_accu_1 += accu_1
        print("Epoch ", t, "  valid_loss_r: ", valid_loss_r, "  valid_loss_t: ", valid_loss_t, " valid_accu_0: ", valid_accu_0, " valid_accu_1: ", valid_accu_1)


        ##test
        test_loss_r = 0
        test_loss_t = 0
        test_accu_0 = 0
        test_accu_1 = 0
        for b, batch in enumerate(iterator):
            data_x, _, data_y, _, data_z, mask  = batch
            data_xy = torch.cat((data_x,data_y), 1)

            data_xy_r = data_xy.permute(2,1,0)[:47] # input[:51] t b
            data_xy_t = data_xy.permute(2,1,0)[47:] # input[51:] t b
            data_xy_r = data_xy_r.permute(2,1,0)  # b t input[:51]
            data_xy_t = data_xy_t.permute(2,1,0)  # b t input[51:]
            if torch.cuda.is_available():
                data_xy_r = data_xy_r.cuda()
                data_xy_t = data_xy_t.cuda()
                data_z = data_z.cuda()
                mask = mask.cuda()
            eval_loss_r, eval_loss_t, accu_0, accu_1 = evaluate_ADRNN_CNN_MultiLabel(model, C1, data_xy_r, data_xy_t, data_z, mask, loss_fn)
            test_loss_r += eval_loss_r.item()
            test_loss_t += eval_loss_t.item()
            test_accu_0 += accu_0
            test_accu_1 += accu_1
        print("Epoch ", t, "  test_loss_r: ", test_loss_r, "  test_loss_t: ", test_loss_t, "  test_accu_0: ", test_accu_0, "  test_accu_1: ", test_accu_1)

    return model, C1


def run_ADRNN_to_CNN_MultiLabel(iterator, model, C1,path=None):
    if torch.cuda.is_available():
        model=model
    print(model)
    print(C1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0005)   # optimize all cnn parameters
    loss_fn = nn.MSELoss()

    h_state = None # for initial hidden state
    num_epochs = 30
    for t in range(num_epochs):
        if os.path.exists('model_test/run_ADRNN_to_CNN_MultiLabel_CNN_b128.model'):
            model = torch.load('model_test/run_ADRNN_to_CNN_MultiLabel_CNN_b128.model')
            model.cuda()
            break
        train_loss = 0
        train_loss_r = 0
        train_loss_t = 0
        model = model.train()
        for b, batch in enumerate(iterator):
            batch, mask, _, _, _, _ = batch # batch = (b, t, input)
            batch = batch.permute(1,0,2)  # batch = (t, b, input)
            mask = mask.permute(1,0,2)
            X_train = batch[:-1]
            X_train = X_train.permute(1,0,2)
            X_train_r = X_train.permute(2,1,0)[:47] # input[:51] t b
            X_train_t = X_train.permute(2,1,0)[47:] # input[51:] t b
            X_train_r = X_train_r.permute(2,1,0)  # b t input[:51]
            X_train_t = X_train_t.permute(2,1,0)  # b t input[51:]

            Y_train = batch[1:]
            Y_train = Y_train.permute(1,0,2)
            mask = mask[1:]
            mask = mask.permute(1,0,2)

            # Forward pass
            if torch.cuda.is_available():
                X_train_r = X_train_r.cuda()
                X_train_t = X_train_t.cuda()
                mask = mask.cuda()
                Y_train = Y_train.cuda()
            Y_pred_r,Y_pred_t = model(X_train_r, X_train_t)  # X_train = (batch, time_step, input_size)   Y_p_r = b t input[:51]    Y_p_t = b t input[51:]
            Y_pred = torch.cat((Y_pred_r, Y_pred_t),2)  # b t input
            Y_pred = torch.mul(Y_pred,mask)

            loss_r = loss_fn(Y_pred.permute(2,1,0)[:47], Y_train.permute(2,1,0)[:47])
            loss_t = loss_fn(Y_pred.permute(2,1,0)[47:], Y_train.permute(2,1,0)[47:])
            loss = loss_r + loss_t

            train_loss = train_loss + loss.item()
            train_loss_r += loss_r.item()
            train_loss_t += loss_t.item()
            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
        print("Epoch ", t, "  train_loss: ", train_loss, "  train_loss_r: ", train_loss_r, "  train_loss_t: ", train_loss_t)


        ##evaluation
        valid_loss_r = 0
        valid_loss_t = 0
        model = model.eval()
        with torch.set_grad_enabled(False):
            for b, batch in enumerate(iterator):
                data_x, _, data_y, mask, _, _  = batch

                data_x_r = data_x.permute(2,1,0)[:47] # input[:51] t b
                data_x_t = data_x.permute(2,1,0)[47:] # input[51:] t b
                data_x_r = data_x_r.permute(2,1,0)  # b t input[:51]
                data_x_t = data_x_t.permute(2,1,0)  # b t input[51:]
                if torch.cuda.is_available():
                    data_x_r = data_x_r.cuda()
                    data_x_t = data_x_t.cuda()
                    data_y = data_y.cuda()
                    mask = mask.cuda()
                eval_loss_r, eval_loss_t = evaluate_ADRNN(model, data_x_r, data_x_t, data_y, mask, loss_fn)
                valid_loss_r += eval_loss_r.item()
                valid_loss_t += eval_loss_t.item()
            print("Epoch ", t, "  valid_loss_r: ", valid_loss_r, "  valid_loss_t: ", valid_loss_t)

            ##test
            test_loss_r = 0
            test_loss_t = 0
            for b, batch in enumerate(iterator):
                data_x, _, data_y, _, data_z, mask  = batch
                data_xy = torch.cat((data_x,data_y), 1)

                data_xy_r = data_xy.permute(2,1,0)[:47] # input[:51] t b
                data_xy_t = data_xy.permute(2,1,0)[47:] # input[51:] t b
                data_xy_r = data_xy_r.permute(2,1,0)  # b t input[:51]
                data_xy_t = data_xy_t.permute(2,1,0)  # b t input[51:]
                if torch.cuda.is_available():
                    data_xy_r = data_xy_r.cuda()
                    data_xy_t = data_xy_t.cuda()
                    data_z = data_z.cuda()
                    mask = mask.cuda()
                eval_loss_r, eval_loss_t  = evaluate_ADRNN(model, data_xy_r, data_xy_t, data_z, mask, loss_fn)
                test_loss_r += eval_loss_r.item()
                test_loss_t += eval_loss_t.item()
            print("Epoch ", t, "  test_loss_r: ", test_loss_r, "  test_loss_t: ", test_loss_t)
            test_loss = 0

    torch.save(model, 'model_test/run_ADRNN_to_CNN_MultiLabel_CNN_b128.model')
    optimizer = torch.optim.Adam(C1.parameters(), lr=0.0002, weight_decay=0.0005)   # optimize all cnn parameters
    loss_ce =nn.BCELoss()

    h_state = None # for initial hidden state
    num_epochs = 30

    for t in range(num_epochs):
        train_loss = 0
        train_loss_t1 = 0
        train_loss_t2 = 0
        C1 = C1.train()
        for b, batch in enumerate(iterator):
            #print('batch: ' + str(b))
            batch, mask, _, _, _, _ = batch # batch = (b, t, input)
            batch = batch.permute(1,0,2)  # batch = (t, b, input)
            mask = mask.permute(1,0,2)   # t b input
            X_train = batch[:-1]
            X_train = X_train.permute(1,0,2)
            X_train_r = X_train.permute(2,1,0)[:47] # input[:51] t b
            X_train_t = X_train.permute(2,1,0)[47:] # input[51:] t b
            X_train_r = X_train_r.permute(2,1,0)  # b t input[:51]
            X_train_t = X_train_t.permute(2,1,0)  # b t input[51:]


            #mask = mask[1:]
            #mask = mask.permute(1,0,2)  # b t input

            # Forward pass
            if torch.cuda.is_available():
                X_train_r = X_train_r.cuda()
                X_train_t = X_train_t.cuda()
                mask = mask.cuda()

            Yc1_pred = C1(torch.cat((X_train_r,X_train_t),2))  # -> b t c
            Yc1_pred = F.sigmoid(Yc1_pred)
            Yc1_pred = Yc1_pred.contiguous().view(-1,2)

            mask_c = mask[12:]  # b t input -> t b input
            mask_c = mask_c.permute(2,1,0)[-2:]  #   input b t
            mask_c = mask_c.permute(1,2,0) # b t c
            Yc1_pred = torch.mul(Yc1_pred, mask_c.contiguous().view(-1,2)) # -1 c

            Target = batch[12:]  #-> t[24:]  b input
            Target = Target.permute(2,1,0)[47:] #-> input(2) b t[24:]
            Target = Target.permute(1,2,0) #-> b t[24:] c
            Target = score2Mlabel(Target.cuda()) # b t[24:] c
            Target = Target.view(-1,2) # -1 c

            loss_t1 = loss_ce(Yc1_pred, Target)

            loss = loss_t1 # + loss_t2
            train_loss = train_loss + loss.item()
            train_loss_t1 += loss_t1.item()
            #train_loss_t2 += loss_t2.item()

            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

        print("Epoch ", t, "  train_loss_t1: ", train_loss_t1)


        ##evaluation
        eval_loss = 0
        valid_loss_r = 0
        valid_loss_t = 0
        valid_accu_0 = 0
        valid_accu_1 = 0
        model = model.eval()
        C1 = C1.eval()
        with torch.set_grad_enabled(False):
            for b, batch in enumerate(iterator):
                data_x, _, data_y, mask, _, _  = batch

                data_x_r = data_x.permute(2,1,0)[:47] # input[:51] t b
                data_x_t = data_x.permute(2,1,0)[47:] # input[51:] t b
                data_x_r = data_x_r.permute(2,1,0)  # b t input[:51]
                data_x_t = data_x_t.permute(2,1,0)  # b t input[51:]
                if torch.cuda.is_available():
                    data_x_r = data_x_r.cuda()
                    data_x_t = data_x_t.cuda()
                    data_y = data_y.cuda()
                    mask = mask.cuda()
                eval_loss_r, eval_loss_t, accu_0, accu_1 = evaluate_ADRNN_to_CNN_MultiLabel(model, C1, data_x_r, data_x_t, data_y, mask, loss_fn)
                valid_loss_r += eval_loss_r.item()
                valid_loss_t += eval_loss_t.item()
                valid_accu_0 += accu_0
                valid_accu_1 += accu_1
            print("Epoch ", t, "  valid_loss_r: ", valid_loss_r, "  valid_loss_t: ", valid_loss_t, " valid_accu_0: ", valid_accu_0, " valid_accu_1: ", valid_accu_1)


            ##test
            test_loss_r = 0
            test_loss_t = 0
            test_accu_0 = 0
            test_accu_1 = 0
            for b, batch in enumerate(iterator):
                data_x, _, data_y, _, data_z, mask  = batch
                data_xy = torch.cat((data_x,data_y), 1)

                data_xy_r = data_xy.permute(2,1,0)[:47] # input[:51] t b
                data_xy_t = data_xy.permute(2,1,0)[47:] # input[51:] t b
                data_xy_r = data_xy_r.permute(2,1,0)  # b t input[:51]
                data_xy_t = data_xy_t.permute(2,1,0)  # b t input[51:]
                if torch.cuda.is_available():
                    data_xy_r = data_xy_r.cuda()
                    data_xy_t = data_xy_t.cuda()
                    data_z = data_z.cuda()
                    mask = mask.cuda()
                eval_loss_r, eval_loss_t, accu_0, accu_1 = evaluate_ADRNN_to_CNN_MultiLabel(model, C1, data_xy_r, data_xy_t, data_z, mask, loss_fn)
                test_loss_r += eval_loss_r.item()
                test_loss_t += eval_loss_t.item()
                test_accu_0 += accu_0
                test_accu_1 += accu_1
            print("Epoch ", t, "  test_loss_r: ", test_loss_r, "  test_loss_t: ", test_loss_t, "  test_accu_0: ", test_accu_0, "  test_accu_1: ", test_accu_1)


    return model, C1

def data_feature(train, mask): # b t input

    train_1 = train.permute(2,1,0)[51] # t b
    train_m1 = mask.permute(2,1,0)[51] # t b

    train_1 = torch.mul(train_1, train_m1)
    train_1_Accu = (train_1!=0).mul(1)  #pos
    train_1_Accu = torch.sum(train_1_Accu.type(torch.cuda.FloatTensor))  #pos count
    train_1_sum_count = torch.sum(train_m1) #sum count

    train_2 = train.permute(2,1,0)[52] # t b
    train_m2 = mask.permute(2,1,0)[52] # t b
    train_2 = torch.mul(train_2, train_m2)
    train_2_Accu = (train_2!=0).mul(1)  #pos
    train_2_Accu = torch.sum(train_2_Accu.type(torch.cuda.FloatTensor))  #pos count
    train_2_sum_count = torch.sum(train_m2) #sum count

    train_3 = train.permute(2,1,0)[53] # t b
    train_m3 = mask.permute(2,1,0)[53] # t b
    train_3 = torch.mul(train_3, train_m3)
    train_3_Accu = (train_3!=0).mul(1)  #pos
    train_3_Accu = torch.sum(train_3_Accu.type(torch.cuda.FloatTensor))  #pos count
    train_3_sum_count = torch.sum(train_m3) #sum count

    train_4 = train.permute(2,1,0)[54] # t b
    train_m4 = mask.permute(2,1,0)[54] # t b
    train_4 = torch.mul(train_4, train_m4)
    train_4_Accu = (train_4!=0).mul(1)  #pos
    train_4_Accu = torch.sum(train_4_Accu.type(torch.cuda.FloatTensor))  #pos count
    train_4_sum_count = torch.sum(train_m4) #sum count

    print("train: ")
    print("1: "+str(train_1_Accu)+' '+str(train_1_sum_count) +' ' +str(train_1_Accu.item()/train_1_sum_count.item()))
    print("2: "+str(train_2_Accu)+' '+str(train_2_sum_count) +' ' +str(train_2_Accu.item()/train_2_sum_count.item()))
    print("3: "+str(train_3_Accu)+' '+str(train_3_sum_count) +' ' +str(train_3_Accu.item()/train_3_sum_count.item()))
    print("4: "+str(train_4_Accu)+' '+str(train_4_sum_count) +' ' +str(train_4_Accu.item()/train_4_sum_count.item()))



# In[9]:
######### set parameters ################
hidden_size = [128,256,512]
hidden_size_r = [32,64]
num_layers = [1,2]
dropout = [0.2,0.5]
batch_size = [64,128,256]
module = "LSTM_r_t"

os.environ["CUDA_VISIBLE_DEVICES"]="2"

for h in hidden_size:
    for n in num_layers:
        for d in dropout:
            for b in batch_size:
                for h_r in hidden_size_r:
                    ######## DataLoader ################
                    params = {'batch_size': b,
                              'shuffle': True,
                              'num_workers': 6}
                    dataset, mask = read_dataset('/home/mit/alternating_prediction/datasets/normalized_dataset_new.pkl')
                    train_data, train_mask, valid_data, valid_mask, test_data, test_mask = data_process_test.dataset_split(dataset, mask) # b t input
                    print(train_data.shape)
                    print(train_mask.shape)
                    # dataset = Dataset(train_data, train_mask, valid_data, valid_mask, test_data, test_mask)
                    # iterator = data.DataLoader(dataset, **params)
                    train_dataset = TrainDataset(train_data, train_mask)
                    vaild_dataset = VaildDataset(valid_data, valid_mask)
                    test_dataset = TestDataset(test_data, test_mask)

                    train_iterator= data.DataLoader(train_dataset, **params)
                    vaild_iterator=data.DataLoader(vaild_dataset,**params)
                    test_iterator=data.DataLoader(test_dataset,**params)                    

                    ######### Model ################
                    if module == "LSTM_r_t":
                        model_r = LSTM(h, n, d, 47).cuda()
                        model_t = LSTM(h_r, n, d, 2).cuda()
                    if module == "RNN_r_t":
                        model_r = RNN(h, n, d, 47).cuda()
                        model_t = RNN(h_r, n, d, 2).cuda()
                    if module == "GRU_r_t":
                        model_r = GRU(h, n, d, 47).cuda()
                        model_t = GRU(h_r, n, d, 2).cuda()
                    elif module == "EncoderDecoder":
                        model = EncoderDecoder().cuda()
                    elif module == "ADRNN":
                        #model = ADRNN().cuda()
                        model = ADRNN_new().cuda()
                    elif module == "ADRNN_Linear":
                        model = ADRNN_Linear().cuda()
                    elif module == "ADRNN_ltCNN":
                        model = ADRNN().cuda()
                        classifier1 = Classifier(window_sizes = [5], feature_size=64).cuda()
                        classifier2 = Classifier(window_sizes = [5], feature_size=64).cuda()
                    elif module == "ADRNN_ltCNN_MultiLabel" or module == "ADRNN_to_ltCNN_MultiLabel":
                        model = ADRNN_new().cuda()
                        classifier1 = Classifier(window_sizes = [5], feature_size=64, num_classes=2).cuda()##
                    elif module == "ADRNN_ResNet18":
                        model = ADRNN().cuda()
                        classifier1 = ResNet18(ResidualBlock).cuda()
                        classifier2 = ResNet18(ResidualBlock).cuda()
                    elif module == "ADRNN_ResNet18_MultiLabel":
                        model = ADRNN().cuda()
                        classifier1 = ResNet18(ResidualBlock,num_classes=2).cuda()


                    ######### Run ################
                    path = '../trained_models/model_lstm_rt/'+str(module)+'_'+str(h)+'_'+str(n)+'_'+str(d)+'_'+str(b)+"_"+str(h_r)
                    if module == "ADRNN_Linear" or module == "ADRNN":
                        model = run_ADRNN(iterator, model,path)
                    if module == "LSTM_rt" or module == "LSTM_r_t":
                        model_r, model_t = run_RNN_rt(train_iterator,vaild_iterator,test_iterator, model_r, model_t,path)
                    elif module == "ADRNN_ltCNN" or module == "ADRNN_ResNet18" :
                        model, c1, c2 = run_ADRNN_CNN(iterator, model, classifier1, classifier2,path)
                    elif module == "ADRNN_ResNet18_MultiLabel" or module == "ADRNN_ltCNN_MultiLabel":
                        model, c1 = run_ADRNN_CNN_MultiLabel(iterator, model, classifier1,path)
                    elif module == "ADRNN_to_ltCNN_MultiLabel":
                        model, c1 = run_ADRNN_to_CNN_MultiLabel(iterator, model, classifier1,path)
                    else:
                        model = run_RNN(iterator, model,path)

######### Save ################
torch.save(model, 'model_test/'+str(module)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(dropout)+'_'+str(batch_size)+'.model')
torch.save(c1, 'model_test/'+str(module)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(dropout)+'_'+str(batch_size)+'.c')



# In[ ]:





# In[ ]:





# In[ ]:
