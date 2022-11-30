from re import T
import matplotlib
import matplotlib.pyplot as plt
from tools.utils import EarlyStopping
from tools.training_tools import test
from tools.metrics import RMSE
from tools.evaluate_tools import evaluate, evaluate_adrnn_cnn, \
    evaluate_adrnn_cnn_multilabel, evaluate_separate_lstm, evaluate_simulate,evaluate_simu_t, evaluate_stacked_lstm, evaluate_flat_lstm,evaluate_timesetps
from tools.draw_lines import draw_lines
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import random
import string
from ast import Try
from cProfile import label
import codecs
import imp
import os
from turtle import color
from nets.encoder import Encoder, EncoderLayer, ConvLayer
from nets.decoder import Decoder, DecoderLayer
from nets.embed import DataEmbedding
from nets.attn import AttentionLayer, FullAttention
# from nets.transformer import StackedTransformer

from sklearn import metrics

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class StackedTransformer(nn.Module):
    def __init__(self, response_size=47, treatment_size=2, d_model=512, d_ff=512, activation='gelu',
                 mix=True, d_layers=1, n_heads=8, e_layers=2, distil=True, repeat_t=0, hidden_r=512, hidden_t=512,
                 hidden_cnn=256, window_sizes=[3, 5, 7], num_classes=2, window=12, dropout=0):
        super(StackedTransformer, self).__init__()
        self.response_size = response_size
        self.treatment_size = treatment_size

        self.window = window
        self.num_classes = num_classes

        self.enc_embedding = DataEmbedding(
            response_size + treatment_size, d_model, dropout)
        self.r_dec_embedding = DataEmbedding(response_size, d_model, dropout)
        self.t_dec_embedding = DataEmbedding(treatment_size, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(mask_flag=False, scale=None, attention_dropout=dropout),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.r_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True, scale=None, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=mix),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False, scale=None, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.t_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True, scale=None, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=mix),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False, scale=None, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.cnn = nn.ModuleList([nn.Sequential(
            nn.Conv1d(response_size + treatment_size, hidden_cnn,
                      kernel_size=h, stride=1, padding=1),  # 28   #1
            nn.BatchNorm1d(hidden_cnn),
            nn.ReLU(),
            nn.Conv1d(hidden_cnn, hidden_cnn, kernel_size=h,
                      stride=1, padding=1),  # 28   #1
            nn.BatchNorm1d(hidden_cnn),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1))
            for h in window_sizes])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_cnn * len(window_sizes), hidden_cnn),
            nn.BatchNorm1d(hidden_cnn),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_cnn, self.num_classes),
            nn.Sigmoid()
        )

        self.r_projection = nn.Linear(d_model, response_size, bias=True)
        self.t_projection = nn.Linear(d_model, treatment_size, bias=True)

    def replace(self, x_t_new, x_t):  # b input
        x_t_cat = torch.cat((x_t_new, x_t), 1)
        return x_t_cat[:, :x_t.shape[1]]

    def forward(self, x_r, x_t, is_train=True, gen_length=None,bias=None):
        device = x_r.device
        batch_size = x_r.shape[0]
        end_len=12
        all_len=24
        
        r_t = torch.cat([x_r, x_t], dim=-1)
        if not is_train:
            x_r = x_r[:, :end_len, :]
            x_t = x_t[:, :end_len, :]
            r_t = torch.cat([x_r, x_t], dim=-1)

            for i in range(end_len):
                enc_out = self.enc_embedding(r_t)
                enc_out, attns = self.encoder(enc_out, attn_mask=None)

                r_dec_out = self.r_dec_embedding(x_r, None)
                r_dec_out = self.r_decoder(r_dec_out, enc_out, None, None)
                r_output = self.r_projection(r_dec_out)[:,-1:,:]

                t_dec_out = self.t_dec_embedding(x_t, None)
                t_dec_out = self.t_decoder(t_dec_out, enc_out, None, None)
                t_output = self.t_projection(t_dec_out)[:,-1:,:]
                # if i>=self.window:
                
                # new_r_t = torch.cat([r_output, t_output], dim=2)
                # r_t_1 = torch.cat([r_t, new_r_t], dim=1)
                # cnn_result=[]
                # for n, sample in enumerate([r_t_1[:, i:i + self.window, :] for i in range(r_t_1.shape[1] - self.window -23,r_t_1.shape[1] - self.window + 1)]):
                # # batch_size input_size window
                #     sample = sample.permute(0, 2, 1)

                #     sample = [conv(sample) for conv in self.cnn]
                #     # print(sample.shape)

                #     sample = torch.cat(sample, dim=1).squeeze().view(batch_size, -1)

                #     sample = self.classifier(sample)
                #     cnn_result.append(sample)
                # cnn_features = torch.stack(cnn_result).permute(1, 0, 2)   
                
                cnn_input = r_t[:, -self.window:, :].permute(0, 2, 1)
                cnn_features = [conv(cnn_input) for conv in self.cnn]
                cnn_features = torch.cat(
                    cnn_features, dim=1).squeeze().view(batch_size, -1)
                cnn_features = self.classifier(cnn_features).unsqueeze(1)

                t_output = torch.mul(t_output, (cnn_features >= 0.5).float())

                x_r = torch.cat([x_r, r_output], dim=1)
                x_t = torch.cat([x_t, t_output], dim=1)
                new_r_t = torch.cat([r_output, t_output], dim=2)
                if bias!=None:
                    new_r_t[:,:,:47]=new_r_t[:,:,:47]+bias[:,i:i+1,:]
                r_t = torch.cat([r_t, new_r_t], dim=1)
            return x_r[:, :, :], x_t[:, :, :]


        # enc_out = self.enc_embedding(r_t)
        # enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # r_dec_out = self.r_dec_embedding(x_r, None)
        # r_dec_out = self.r_decoder(r_dec_out, enc_out, None, None)
        # r_output = self.r_projection(r_dec_out)

        # t_dec_out = self.t_dec_embedding(x_t, None)
        # t_dec_out = self.t_decoder(t_dec_out, enc_out, None, None)
        # t_output = self.t_projection(t_dec_out)

        for i in range(24):
            enc_out = self.enc_embedding(r_t)
            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            r_dec_out = self.r_dec_embedding(x_r, None)
            r_dec_out = self.r_decoder(r_dec_out, enc_out, None, None)
            r_output = self.r_projection(r_dec_out)[:, -1:, :]

            t_dec_out = self.t_dec_embedding(x_t, None)
            t_dec_out = self.t_decoder(t_dec_out, enc_out, None, None)
            t_output = self.t_projection(t_dec_out)[:, -1:, :]


            x_r = torch.cat([x_r, r_output], dim=1)
            x_t = torch.cat([x_t, t_output], dim=1)
            new_r_t = torch.cat([r_output, t_output], dim=2)
            r_t = torch.cat([r_t, new_r_t], dim=1)
        # return x_r[:, :, :], x_t[:, :, :]


        cnn_result = []
        # sliding window
        for n, sample in enumerate([r_t[:, i:i + self.window, :] for i in range(r_t.shape[1] - self.window + 1)]):
            # batch_size input_size window
            sample = sample.permute(0, 2, 1)

            sample = [conv(sample) for conv in self.cnn]
            # print(sample.shape)

            sample = torch.cat(sample, dim=1).squeeze().view(batch_size, -1)

            sample = self.classifier(sample)
            cnn_result.append(sample)

        cnn_result = torch.stack(cnn_result).permute(1, 0, 2)

        # 训练时不使用判别式约束
        return x_r[:, -24:, :], x_t[:, -24:, :],cnn_result[:, -24:, :]
        return r_output, t_output, cnn_result



def simulation(test_iterator, vaild_iterator, model, is_mask, window, logger, model_save_path, response_size=47, treatment_size=2,
               device=torch.device("cpu")):
    # model.load_state_dict(torch.load("trained_models/transformer/2022_08_09_13_30_10/model.pt" ,map_location={'cuda:0':'cuda:0','cuda:1': 'cuda:0','cuda:2':'cuda:0','cuda:3':'cuda:0','cuda:4':'cuda:0',
    #                                                                                                            'cuda:5':'cuda:0','cuda:6':'cuda:0','cuda:7':'cuda:0','cuda:8':'cuda:0','cuda:9':'cuda:0'}),)
    model=StackedTransformer(response_size=47, treatment_size=2)
    # model.load_state_dict(torch.load("trained_models/transformer/2022_08_18_18_45_16/model.pt",map_location='cuda:0'))
    model.load_state_dict(torch.load("trained_models/transformer/2022_11_02_14_23_52/model.pt",map_location='cuda:0'))
    model.load_state_dict(torch.load("trained_models/transformer/2022_11_15_01_30_42/model.pt",map_location='cuda:0'))  # 12hours model
    
    

    model = model.to(device)
    start = 0
    leng = 12
    end= 24
    num_batch = 0
    valid_mse_r = 0
    valid_mse_t = 0
    valid_mae_r = 0
    valid_mae_t = 0

    noise = torch.empty(1, leng, 47).to(device)

    model = model.eval()
    with torch.no_grad():

        for b, batch in enumerate(tqdm(vaild_iterator)):
            num_batch += 1
            data_x, mask = batch

            data_x_r = data_x[:, :leng, :response_size].to(device)
            data_x_t = data_x[:, :leng, -treatment_size:].to(device)
            data_x = data_x.to(device)
            mask = mask.to(device)
            data_y = data_x[:, :leng, :].to(device)

            # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
            #                                                             is_mask,
            #                                                             response_size, treatment_size)

            eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t, bias, pred_r, pred_t = evaluate_simulate(model, data_x_r, data_x_t, data_x, mask,
                                                                                                     is_mask,
                                                                                                     response_size, treatment_size)
            valid_mse_r += eval_mse_r
            valid_mse_t += eval_mse_t
            valid_mae_r += eval_mae_r
            valid_mae_t += eval_mae_t
            noise = torch.cat([noise, bias], dim=0)
        valid_mse_r /= num_batch
        valid_mse_t /= num_batch
        valid_mae_r /= num_batch
        valid_mae_t /= num_batch
        geometric_mean = np.exp(
            np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
        log_info = "geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
            format(round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                   round(valid_mae_r, 5), round(valid_mae_t, 5))

    noise = noise[1:, :, :]
    print("mean:", str(noise.mean()), "std:", str(noise.std()))
    num_batch = 0
    valid_mse_r = 0
    valid_mse_t = 0
    valid_mae_r = 0
    valid_mae_t = 0
    pr_s = torch.empty(1, leng, 47).to(device)
    pr = torch.empty(1, end, 47).to(device)
    pt = torch.empty(1, end, 2).to(device)

    model = model.eval()
    sample_N = 100
    channal = 11
    test_mse_r = []
    test_mse_t = []
    test_mae_r = []
    test_mae_t = []
    TN=0
    TP=0
    FN=0
    FP=0
    pred_t_n=[]
    t_n=[]
    # pred_rsum=[]
    # pred_tsum=[]

    with torch.no_grad():

        for b, batch in enumerate(tqdm(test_iterator)):
            pred_rsum = torch.zeros(1, leng, 47).to(device)
            pred_tsum = torch.zeros(1, leng, 2).to(device)
            for a in range(sample_N):
                index = torch.LongTensor(random.sample(
                    range(noise.size(0)), batch[0].size(0)))
                # try:
                bias = torch.index_select(noise, 0, index.cuda())
                bias= torch.normal(mean=0,std=0.4395,size=(bias.shape))
                # except:
                #     print("sample error")
                bias = bias.to(device)
                num_batch += 1
                data_x, mask = batch

                data_x_r = data_x[:, :leng, :response_size].to(device)
                # data_x_r = data_x_r+bias
                data_x_t = data_x[:, :leng, -treatment_size:].to(device)
                # data_y = data_y.to(device)
                mask = mask.to(device)
                data_x = data_x.to(device)

                # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
                #                                                             is_mask,
                #                                                             response_size, treatment_size)

                
                
                #噪音实验
                eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t, _, pred_r, pred_t = evaluate_simulate(model, data_x_r, data_x_t, data_x, mask,
                                                                                                      is_mask,
                                                                                                      response_size, treatment_size,Bias=bias)
                for i in range(batch[0].size(0)):
                    treatment=data_x[i, leng:end, -treatment_size:].nonzero()
                    treatment_pred=pred_t[i,:,:].nonzero()
                    t_n.append( 0 if treatment.shape[0]==0 else 1)
                    pred_t_n.append(0 if treatment_pred.shape[0]==0 else 1)
                    if treatment.shape[0]==0 and treatment_pred.shape[0]==0:
                        TN=TN+1
                    if treatment.shape[0]!=0 and treatment_pred.shape[0]!=0:
                        TP=TP+1  
                    if treatment.shape[0]==0 and treatment_pred.shape[0]!=0:
                        FP=FP+1   
                    if treatment.shape[0]!=0 and treatment_pred.shape[0]==0:
                        FN=FN+1                                                                
                
                # time step的实验
                # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t, pred_r, pred_t = evaluate_timesetps(model, data_x_r, data_x_t, data_x, mask,
                #                                                                                       is_mask,
                #                                                                                       response_size, treatment_size)

                pred_rsum = pred_rsum+pred_r[:,leng:,:]
                pred_tsum = pred_tsum+pred_t[:,leng:,:]

                valid_mse_r += eval_mse_r
                valid_mse_t += eval_mse_t
                valid_mae_r += eval_mae_r
                valid_mae_t += eval_mae_t
                if b == 6:
                    pr_s = torch.cat([pr_s, pred_r[11:12, leng:, :]], dim=0)
                    
                    orig = data_x[11, 0:end, channal]
                # pr=torch.cat([pr,pred_r],dim=0)
                # pt=torch.cat([pt,pred_t],dim=0)
            pred_rsum /= sample_N

            pred_tsum /= sample_N

            loss_mse = nn.MSELoss()
            loss_mae = nn.L1Loss()
            all_len = 24
            eval_mse_r = loss_mse(
                data_x[:, leng:all_len, :response_size], pred_rsum[:, :, :]).item()
            eval_mse_t = loss_mse(
                data_x[:, leng:all_len, -treatment_size:], pred_tsum[:, :, :]).item()
            eval_mae_r = loss_mae(
                data_x[:, leng:all_len, :response_size], pred_rsum[:, :, :]).item()
            eval_mae_t = loss_mae(
                data_x[:, leng:all_len, -treatment_size:], pred_tsum[:, :, :]).item()
            test_mse_r.append(eval_mse_r)
            test_mse_t.append(eval_mse_t)
            test_mae_r.append(eval_mae_r)
            test_mae_t.append(eval_mae_t)
            # fig=plt.figure()
            # matplotlib.use('TkAgg')
            # ax3 = plt.axes(projection='3d')
            # x=np.arange(0,pr.shape[1])
            # y=np.arange(0,pr.shape[2])
            # Xx, Yy = np.meshgrid(x, y)
            # Zz=data_x[1,Xx,Yy].cpu().numpy()
            # plt.xlabel("Time")
            # plt.xlabel("Channel")
            # ax3.plot_surface(Xx,Yy,Zz,rstride = 1, cstride = 1,cmap='rainbow')
            # plt.show()



        valid_mse_r /= num_batch
        valid_mse_t /= num_batch
        valid_mae_r /= num_batch
        valid_mae_t /= num_batch
        geometric_mean = np.exp(
            np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
        log_info = "test geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
            format(round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                   round(valid_mae_r, 5), round(valid_mae_t, 5))
    print("MEAN LOSS valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}".
          format(round(np.mean(test_mse_r), 5), round(np.mean(test_mse_t), 5), round(np.mean(test_mae_r), 5), round(np.mean(test_mae_t), 5)))
    print("geo_mean=", np.exp(np.log([np.mean(test_mse_r), np.mean(
        test_mse_t), np.mean(test_mae_r), np.mean(test_mae_t)]).mean()))
    print("N=", str(sample_N))
    logger.info(log_info)
    logger.info("\n")
    logger.info("best_geometric_mean {}\n".format(round(geometric_mean, 5)))
    
    
    Recall=TP/(TP+FN)
    Precision=TP/(TP+FP)
    print("recall:",str(Recall),"     precision: ",str(Precision))
    auc=metrics.roc_auc_score(t_n,pred_t_n)
    print(auc)
    pred_mean = []
    for i in range(24, pr_s.shape[1]):
        pred_mean.append(pr_s[1:101, i, channal].detach().cpu().numpy().mean())
    fig=plt.figure()
    matplotlib.use('TkAgg')
    # ax3 = plt.axes(projection='3d')
    # x=np.arange(0,pr.shape[1])
    # y=np.arange(0,pr.shape[2])
    # Xx, Yy = np.meshgrid(x, y)
    # Zz=pr[1,Xx,Yy].cpu().numpy()
    # plt.xlabel("Time")
    # plt.xlabel("Channel")
    # ax3.plot_surface(Xx,Yy,Zz,rstride = 1, cstride = 1,cmap='rainbow')
    # plt.show()

    fig, ax = plt.subplots() # 创建图实例
    for i in range(1,101):
        ax.plot(range(pr_s.shape[1]),pr_s[i,:,channal].detach().cpu().numpy(),color="lightcyan")
    ax.plot(range(pr_s.shape[1]),orig.detach().cpu().numpy(),color='r',label="Grandtruth")
    ax.plot(range(24,pr_s.shape[1]),pred_mean,color='deepskyblue',label="Mean")
    plt.axvline(24)
    plt.legend()
    plt.xlabel("Number of Time steps")
    plt.ylabel("RR value")
    plt.show()


def simulation_transformer(test_iterator, vaild_iterator, model, is_mask, window, logger, model_save_path, response_size=47, treatment_size=2,
               device=torch.device("cpu")):

    model.load_state_dict(torch.load(
        "trained_models/transformer/2022_08_15_18_51_33/Informer_model.pt"))      #24hours
    # model.load_state_dict(torch.load(
    #     "trained_models/transformer/2022_11_15_16_47_35/informer_model.pt"))     #12HOURS
    # model.load_state_dict(torch.load(
    #     "trained_models/transformer/2022_11_14_14_12_12/trans_model.pt"))       #24hours
    # model.load_state_dict(torch.load(
    #     "trained_models/transformer/2022_11_16_02_05_52/trans_model.pt"))       #12hours

    # model.load_state_dict(torch.load("trained_models/transformer/2022_07_29_03_34_19/model.pt",map_location='cuda:0'))
    model = model.to(device)

    num_batch = 0
    valid_mse_r = 0
    valid_mse_t = 0
    valid_mae_r = 0
    valid_mae_t = 0

    start = 0
    leng = 24
    end= 48


    noise = torch.empty(1, leng, 47).to(device)

    model = model.eval()
    with torch.no_grad():

        for b, batch in enumerate(tqdm(vaild_iterator)):
            num_batch += 1
            data_x, mask = batch

            data_x_r = data_x[:, :leng, :response_size].to(device)
            data_x_t = data_x[:, :leng, -treatment_size:].to(device)
            data_x = data_x.to(device)
            mask = mask.to(device)
            data_y = data_x[:, :leng, :].to(device)

            # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
            #                                                             is_mask,
            #                                                             response_size, treatment_size)

            eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t, bias, pred = evaluate_simu_t(model, data_y ,data_x, mask,
                                                                                                     
                                                                                                     response_size, treatment_size)
            valid_mse_r += eval_mse_r
            valid_mse_t += eval_mse_t
            valid_mae_r += eval_mae_r
            valid_mae_t += eval_mae_t
            noise = torch.cat([noise, bias], dim=0)
        valid_mse_r /= num_batch
        valid_mse_t /= num_batch
        valid_mae_r /= num_batch
        valid_mae_t /= num_batch
        geometric_mean = np.exp(
            np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
        log_info = "geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
            format(round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                   round(valid_mae_r, 5), round(valid_mae_t, 5))

    noise = noise[1:, :, :]
    print("mean:", str(noise.mean()), "std:", str(noise.var()))
    num_batch = 0
    valid_mse_r = 0
    valid_mse_t = 0
    valid_mae_r = 0
    valid_mae_t = 0
    pr_s = torch.empty(1, leng, 47).to(device)
    pr = torch.empty(1, end, 47).to(device)
    pt = torch.empty(1, end, 2).to(device)

    model = model.eval()
    sample_N = 100
    channal = 11
    test_mse_r = []
    test_mse_t = []
    test_mae_r = []
    test_mae_t = []
    TN=0
    TP=0
    FN=0
    FP=0
    pred_t_n=[]
    t_n=[]
    # pred_rsum=[]
    # pred_tsum=[]

    with torch.no_grad():

        for b, batch in enumerate(tqdm(test_iterator)):
            pred_rsum = torch.zeros(1, leng, 47).to(device)
            pred_tsum = torch.zeros(1, leng, 2).to(device)
            for a in range(sample_N):
                index = torch.LongTensor(random.sample(
                    range(noise.size(0)), batch[0].size(0)))
                # try:
                bias = torch.index_select(noise, 0, index.cuda())
                # bias= torch.normal(mean=0,std=0.668,size=(bias.shape))  #our bias
                # bias= torch.normal(mean=0,std=0.4528,size=(bias.shape))  #Tranformer bias
                # bias= torch.normal(mean=0,std=0.5041,size=(bias.shape))  #Informer bias
                # except:
                #     print("sample error")
                bias = bias.to(device)
                num_batch += 1
                data_x, mask = batch
                data_x=data_x.to(device)
                data_x_r = data_x[:, :leng, :response_size].to(device)
                # data_x_r = data_x_r+bias
                # data_x[:, :24, :response_size]=data_x[:, :24, :response_size]+bias
                data_x_in=data_x[:, :leng, :]
                data_x_t = data_x[:, :leng, -treatment_size:].to(device)
                # data_y = data_y.to(device)
                mask = mask.to(device)
                # data_x = data_x.to(device)

                # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
                #                                                             is_mask,
                #                                                             response_size, treatment_size)

                
                
                #噪音实验
                eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t, bias, pred = evaluate_simu_t(model, data_x_in, data_x, mask,
                                                                                                response_size, treatment_size,bias=bias)                # for i in range(batch[0].size(0)):
                    
                for i in range(batch[0].size(0)):
                    treatment=data_x[i, leng:end, -treatment_size:].nonzero()
                    pred_t=pred[i,:,-2:]>=0.05
                    treatment_pred=pred_t[:,-2:].nonzero()
                    t_n.append( 0 if treatment.shape[0]==0 else 1)
                    pred_t_n.append(0 if treatment_pred.shape[0]==0 else 1)
                    if treatment.shape[0]==0 and treatment_pred.shape[0]==0:
                        TN=TN+1
                    if treatment.shape[0]!=0 and treatment_pred.shape[0]!=0:
                        TP=TP+1  
                    if treatment.shape[0]==0 and treatment_pred.shape[0]!=0:
                        FP=FP+1   
                    if treatment.shape[0]!=0 and treatment_pred.shape[0]==0:
                        FN=FN+1                                                                
                
                # time step的实验
                # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t, pred_r, pred_t = evaluate_timesetps(model, data_x_r, data_x_t, data_x, mask,
                #                                                                                       is_mask,
                #                                                                                       response_size, treatment_size)

                pred_rsum = pred_rsum+pred[:, :, :response_size]
                pred_tsum = pred_tsum+pred[:, :, -treatment_size:]

                valid_mse_r += eval_mse_r
                valid_mse_t += eval_mse_t
                valid_mae_r += eval_mae_r
                valid_mae_t += eval_mae_t
                if b == 6:
                    pr_s = torch.cat([pr_s, pred[11:12, :leng, :47]], dim=0)
                    
                    orig = data_x[11, leng:end, channal]
                # pr=torch.cat([pr,pred_r],dim=0)
                # pt=torch.cat([pt,pred_t],dim=0)
            pred_rsum /= sample_N

            pred_tsum /= sample_N

            loss_mse = nn.MSELoss()
            loss_mae = nn.L1Loss()
            all_len = 48
            eval_mse_r = loss_mse(
                data_x[:, leng:all_len, :response_size], pred_rsum[:, :, :]).item()
            eval_mse_t = loss_mse(
                data_x[:, leng:all_len, -treatment_size:], pred_tsum[:, :, :]).item()
            eval_mae_r = loss_mae(
                data_x[:, leng:all_len, :response_size], pred_rsum[:, :, :]).item()
            eval_mae_t = loss_mae(
                data_x[:, leng:all_len, -treatment_size:], pred_tsum[:, :, :]).item()
            test_mse_r.append(eval_mse_r)
            test_mse_t.append(eval_mse_t)
            test_mae_r.append(eval_mae_r)
            test_mae_t.append(eval_mae_t)
            # fig=plt.figure()
            # matplotlib.use('TkAgg')
            # ax3 = plt.axes(projection='3d')
            # x=np.arange(0,pr.shape[1])
            # y=np.arange(0,pr.shape[2])
            # Xx, Yy = np.meshgrid(x, y)
            # Zz=data_x[1,Xx,Yy].cpu().numpy()
            # plt.xlabel("Time")
            # plt.xlabel("Channel")
            # ax3.plot_surface(Xx,Yy,Zz,rstride = 1, cstride = 1,cmap='rainbow')
            # plt.show()



        valid_mse_r /= num_batch
        valid_mse_t /= num_batch
        valid_mae_r /= num_batch
        valid_mae_t /= num_batch
        geometric_mean = np.exp(
            np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
        log_info = "test geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
            format(round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                   round(valid_mae_r, 5), round(valid_mae_t, 5))
    print("MEAN LOSS valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}".
          format(round(np.mean(test_mse_r), 5), round(np.mean(test_mse_t), 5), round(np.mean(test_mae_r), 5), round(np.mean(test_mae_t), 5)))
    print("geo_mean=", np.exp(np.log([np.mean(test_mse_r), np.mean(
        test_mse_t), np.mean(test_mae_r), np.mean(test_mae_t)]).mean()))
    print("N=", str(sample_N))
    logger.info(log_info)
    logger.info("\n")
    logger.info("best_geometric_mean {}\n".format(round(geometric_mean, 5)))
    
    
    Recall=TP/(TP+FN)
    Precision=TP/(TP+FP)
    print("recall:",str(Recall),"     precision: ",str(Precision))
    auc=metrics.roc_auc_score(t_n,pred_t_n)
    print(auc)
    pred_mean = []