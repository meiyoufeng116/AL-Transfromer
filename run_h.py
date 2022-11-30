import argparse
import os.path
import random
import shutil
import time
import sys
from matplotlib.transforms import Transform

import numpy as np
import torch
from torch.utils.data import DataLoader
from nets.informer import Informer

import tools.log
from data.data_process import dataset_split, dataset_split_h
# from nets.classifier import Classifier
# from nets.gru import GRU
# from nets.resnet_18 import ResNet18
# from nets.stacked_adrnn_new_v3 import StackedAdrnnNewV3
# from nets.stacked_adrnn_new_v4 import StackedAdrnnNewV4
from data.sepsis_dataset import (SepsisDataset, TestDataset, TrainDataset,
                                 VaildDataset)
# from nets.lstm import LSTM
# from nets.rnn import RNN
# from nets.stacked_lstm_cnn import StackedLstmCnn
from nets.transformer import StackedTransformer,Transformer
from tools.training_tools import test_informer, train_informer, train_stacked_lstm_cnn,test
from tools.simulation import simulation, simulation_transformer

# set random seed
# SEED = 123
# os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# # torch.use_deterministic_algorithms(True)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)

#     # https://pytorch.org/docs/stable/notes/randomness.html
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--model', type=str, default="transformer")
parser.add_argument('--comment', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mask', action="store_true", default=False)
args = parser.parse_args()
device = torch.device("cuda:" + str(args.gpu)
                      if torch.cuda.is_available() else "cpu")

params = {'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 0 
          }

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', args.model,
                          str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())))

if not os.path.exists(model_path):
    os.makedirs(model_path)
save_path = os.path.join(model_path, 'log.txt')

dataset, mask = torch.load('./datasets/normalized_dataset_new.pt')
train_data, train_mask, valid_data, valid_mask, test_data, test_mask = dataset_split_h(
    dataset, mask)  # b t input

train_dataset = TrainDataset(train_data, train_mask)
vaild_dataset = VaildDataset(valid_data, valid_mask)
test_dataset = TestDataset(test_data, test_mask)

train_iterator= DataLoader(train_dataset, **params)
vaild_iterator=DataLoader(vaild_dataset,**params)
test_iterator=DataLoader(test_dataset,**params)

# iterator = DataLoader(dataset, **params)

logger = tools.log.get_logger(save_path)

Test=False
if __name__ == "__main__":
    if args.comment is not None:
        logger.info(args.comment + "\n")

    model = StackedTransformer(response_size=47, treatment_size=2)
    
    # model=Informer( enc_in=49, dec_in=49, c_out=49)
    # model=Transformer()
    if Test==True:
        test(test_iterator, model, args.mask, window=12, logger=logger,
                           model_save_path=os.path.join(model_path, "model.pt"), response_size=47, treatment_size=2, device=device)    
        sys.exit()

    train_stacked_lstm_cnn(args.epoch, train_iterator,vaild_iterator,test_iterator, model, args.mask, window=12, logger=logger,
                           model_save_path=os.path.join(model_path, "model.pt"), response_size=47, treatment_size=2, device=device)

    # simulation( test_iterator,vaild_iterator, model, args.mask, window=12, logger=logger,
    #                        model_save_path=os.path.join(model_path, "model.pt"), response_size=47, treatment_size=2, device=device)

    # test_informer(test_iterator, model, args.mask, window=12, logger=logger,
    #                     model_save_path=os.path.join(model_path, "model.pt"), response_size=47, treatment_size=2, device=device)  
    # train_informer(args.epoch, train_iterator,vaild_iterator,test_iterator, model, args.mask, window=12, logger=logger,
    #                        model_save_path=os.path.join(model_path, "Informer_model.pt"), response_size=47, treatment_size=2, device=device)
    # #
    # simulation_transformer( test_iterator,vaild_iterator, model, args.mask, window=12, logger=logger,
    #                         model_save_path=os.path.join(model_path, "model.pt"), response_size=47, treatment_size=2, device=device)