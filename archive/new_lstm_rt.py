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
from data_process import dataset_split_h

SEED = 1234
torch.backends.cudnn.enabled = False
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class LSTM1(nn.Module):
    def __init__(self,hidden_size=64, num_layers=2, dropout=0.2,input_size=49):
        super(LSTM1, self).__init__()

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
        if is_train==False:
            r_output = torch.zeros(gen_length, batch, self.input_size).cuda()  #   t b input_size
            r_out = x.permute(1,0,2)[23] #->  b hidden_size
            r_output[:23,:,:] = x.permute(1,0,2)[:23,:,:]
            r_out = r_out.view(batch, 1, -1) #->  b 1 hidden_size
            for i in range(24-1,gen_length-1):
                r_out, h_state = self.rnn(r_out, h_state) #r_out  b 1 hidden_size
                r_out = self.drop_layer(r_out)
                r_out = self.out(r_out) #-> b t inputsize
                r_output[i+1]= r_out.permute(1,0,2)
            return r_output.permute(1,0,2), h_state
        # batch = x.shape[0]
        r_out, h_state = self.rnn(x, h_state)
        #return r_out, h_state
        r_out = self.drop_layer(r_out)
        r_out = self.out(r_out) #-> b t inputsize

        return r_out, h_state

class LSTM2(nn.Module):
    def __init__(self,hidden_size=64, num_layers=2, dropout=0.2,input_size=47):
        super(LSTM2, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.num_layers,       # number of rnn layer
            dropout=self.dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.rnn2 = nn.LSTM(
            input_size=47,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.num_layers,       # number of rnn layer
            dropout=self.dropout,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.drop_layer2 = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.hidden_size, 47)
        self.out2 = nn.Linear(self.hidden_size, 2)

    def forward(self, x, h_state=None,t_state=None, is_train=True, gen_length=None):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        batch = x.shape[0]
        x=x[:,0:gen_length,:]
        if is_train==False:
            for i in range(gen_length):
                Input=x[:,i:i+gen_length,:]
                r_out, h_state = self.rnn1(Input, h_state)
                #return r_out, h_state
                r_out = self.drop_layer(r_out)
                r_out = self.out(r_out) #-> b t inputsize
                t_out, t_state = self.rnn2(r_out[:,-1:,:], t_state)
                t_out = self.drop_layer2(t_out)
                t_out = self.out2(t_out)
                
                B_out=torch.cat((r_out[:,-1:,:],t_out[:,-1:,:]),dim=2)
                x=torch.cat((x,B_out[:,-1:,:]),dim=1)

            return x[:,-gen_length:,:], h_state
        # batch = x.shape[0]
        
        
        for i in range(gen_length):
            Input=x[:,i:i+gen_length,:]
            r_out, h_state = self.rnn1(Input, h_state)
            #return r_out, h_state
            r_out = self.drop_layer(r_out)
            r_out = self.out(r_out) #-> b t inputsize
            t_out, t_state = self.rnn2(r_out[:,-1:,:], t_state)
            t_out = self.drop_layer2(t_out)
            t_out = self.out2(t_out)
            
            B_out=torch.cat((r_out[:,-1:,:],t_out[:,-1:,:]),dim=2)
            x=torch.cat((x,B_out[:,-1:,:]),dim=1)

        return x[:,-gen_length:,:], h_state

def run_RNN_rt(train_iterator,vaild_iterator,test_iterator, model_r, model_t, path):
    if torch.cuda.is_available():
        model_r=model_r.cuda()
        model_t=model_t.cuda()
    print(model_r)
    print(model_t)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam([
                {'params': model_r.parameters()},
                {'params': model_t.parameters()},], lr=0.0002, weight_decay=0.0005)   # optimize all cnn parameters
    optimizer_r=torch.optim.Adam(model_r.parameters(),lr=0.0002)
    optimizer_t=torch.optim.Adam(model_t.parameters(),lr=0.0002)
 

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
            X_train_r = X_train.permute(2,1,0)[:49] # input[:51] t b
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
                mask=mask.cuda()
                Y_train_r = Y_train_r.cuda()
                Y_train_t = Y_train_t.cuda()

            Y_pred_r,_ = model_t(X_train_r,gen_length=12)
            
            Y_pred=torch.mul(Y_pred_r,mask[:,12:24,:])
            
            Y_pred_r=Y_pred[:,:,:47]
            Y_pred_t=Y_pred[:,:,47:]


            loss_r = loss_fn(Y_pred_r, Y_train_r[:,12:24,:])
            loss_t = loss_fn(Y_pred_t, Y_train_t[:,12:24,:])
            loss = loss_r + loss_t
            train_loss = train_loss + loss_r.item() + loss_t.item()
            train_loss_r += loss_r.item()
            train_loss_t += loss_t.item()

            optimizer_t.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
        print("Epoch ", t, "  train_loss: ", train_loss, "  train_loss_r: ", train_loss_r, "  train_loss_t: ", train_loss_t,)
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
                mse_loss_r,_, yr, yr_pred,yt,yt_pred = evaluate_rt(model_t, data_y, data_y, mask, loss_fn)
                # eval_loss_t, yt, yt_pred = evaluate_rt(model_t, data_y_t, data_y_t, mask_t, loss_fn)

                Y.append(torch.cat((yr,yt),2))
                Y_pred.append(torch.cat((yr_pred, yt_pred),2))
            l1_r,l1_t,l2_r,l2_t,mean = RMSE(Y, Y_pred)
            print("Epoch ", t,"  dev_loss_all: ", str(l1_r+l1_t+l2_r+l2_t),  "  valid_l1_r: ", l1_r, "  valid_l1_t: ", l1_t,  "  valid_l2_r: ", l2_r, "  valid_l2_t: ", l2_t,"geo mean",str(mean))
            file_w.write("Epoch "+str(t)+"  dev_loss_all: "+str(l1_r+l1_t+l2_r+l2_t)+ "  valid_l1_r: "+str(l1_r)+ "  valid_l1_t: "+str(l1_t)+"  valid_l2_r: "+str(l2_r)+ "  valid_l2_t: "+str( l2_t)+"\n")
            ##test
            Y = []
            Y_pred = []
            valid_mse_r = 0
            valid_mse_t = 0
            valid_mae_r = 0
            valid_mae_t = 0
            n=0
            for b, batch in enumerate(test_iterator):
                data_z, mask  = batch
                n+=1
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
                mse_loss_r,_, yr, yr_pred,yt,yt_pred = evaluate_rt(model_t, data_z, data_z, mask, loss_fn)
                # mse_loss_t, yt, yt_pred = evaluate_rt(model_t, data_z_t, data_z_t, mask_t, loss_fn)

                
                Y.append(torch.cat((yr,yt),2))
                Y_pred.append(torch.cat((yr_pred, yt_pred),2))
            
            l1_r,l1_t,l2_r,l2_t,mean = RMSE(Y, Y_pred)
          

            print("Epoch ", t,"  test_loss_all: ", str(l1_r+l1_t+l2_r+l2_t),  "  test_l1_r: ", str( l1_r), "  test_l1_t: ", str( l1_t),  "  test_l2_r: ", str( l2_r), "  test_l2_t: ", str( l2_t),"geo mean",str(mean))
            file_w.write("Epoch "+str(t)+"  test_loss_all: "+str(l1_r+l1_t+l2_r+l2_t)+ "  test_l1_r: "+str( l1_r)+ "  test_l1_t: "+str( l1_t)+  "  test_l2_r: "+str( l2_r)+ "  test_l2_t: "+str( l2_t)+"\n")
    file_w.close()
    return model_r, model_t


def evaluate_rt(model, data_x, data_y, mask, loss_fn):  # x (batch, time_step, input_size)
    batch = data_x.shape[0]
    length = data_y.shape[1]
    start=12
    length = 24
    Pre_y, _ = model(data_x, is_train=False, gen_length=length) # b t input
    Pre_y = torch.mul(Pre_y,mask[:,:24,:])
    eval_loss_r = loss_fn(data_y[:,start:length,:47], Pre_y[:,start:length,:47])
    eval_loss_t = loss_fn(data_y[:,start:length,47:], Pre_y[:,start:length,47:])
    
    return eval_loss_r,eval_loss_t, data_y[:,start:length,:47], Pre_y[:,start:length,:47],data_y[:,start:length,47:],Pre_y[:,start:length,47:] # b t input

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
    geometric_mean = np.exp(np.log([l1_t.item(), l1_r.item(), l2_t.item(), l2_r.item()]).mean())
    return l1_r.item(), l1_t.item(), l2_r.item(), l2_t.item(),geometric_mean

def read_dataset(path):
    pkl_file = open(path, 'rb')
    return pickle.load(pkl_file)

######### set parameters ################
hidden_size = [128,256,512]
hidden_size_r = [32,64]
num_layers = [1,2]
dropout = [0.2,0.5]
batch_size = [64,128,256]
module = "LSTM_r_t"

os.environ["CUDA_VISIBLE_DEVICES"]="9"

for h in hidden_size:
    for n in num_layers:
        for d in dropout:
            for b in batch_size:
                for h_r in hidden_size_r:
                    ######## DataLoader ################
                    params = {'batch_size': b,
                              'shuffle': True,
                              'num_workers': 0}
                    dataset, mask = read_dataset('/home/mit/alternating_prediction/datasets/normalized_dataset_new.pkl')
                    train_data, train_mask, valid_data, valid_mask, test_data, test_mask = dataset_split_h(dataset, mask) # b t input
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
                        model_r = LSTM1(h, n, d, 49).cuda()
                        model_t = LSTM2(h_r, n, d, 49).cuda()
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
                    path = 'trained_models/model_lstm_rt/'+str(module)+'_'+str(h)+'_'+str(n)+'_'+str(d)+'_'+str(b)+"_"+str(h_r)
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
                    torch.save(model_r, 'model_test/'+str(module)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(dropout)+'_'+str(batch_size)+'r.model')
                    torch.save(model_t, 'model_test/'+str(module)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(dropout)+'_'+str(batch_size)+'all.model')
