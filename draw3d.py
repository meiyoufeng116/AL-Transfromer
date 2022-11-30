import argparse
import os.path
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.data_process import dataset_split_h
from data.sepsis_dataset import (SepsisDataset, TestDataset, TrainDataset,
                                 VaildDataset)
from nets.attn import AttentionLayer, FullAttention
from nets.decoder import Decoder, DecoderLayer
from nets.embed import DataEmbedding
from nets.encoder import Encoder, EncoderLayer, ConvLayer
# from nets.transformer import EncoderDecoder
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# set random seed
SEED = 1234
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, default="encoder_decoder")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--mask', action="store_true", default=False)
parser.add_argument('--data', type=str, default="sepsis")
args = parser.parse_args()
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 6 if os.name == 'posix' else 0
          }

if args.data  == 'sepsis':
    response_size = 47
    treatment_size = 2
    dataset, mask = torch.load('./datasets/normalized_dataset_new.pt')
elif args.data == 'glucose':
    response_size = 1
    treatment_size = 2
    dataset, mask = torch.load('./datasets/glucose.pt')
else:
    print('Error: Invalid model.')
    exit()

dataset, mask = torch.load('./datasets/normalized_dataset_new.pt')
train_data, train_mask, valid_data, valid_mask, test_data, test_mask = dataset_split_h(
    dataset, mask)  # b t input

train_dataset = TrainDataset(train_data, train_mask)
vaild_dataset = VaildDataset(valid_data, valid_mask)
test_dataset = TestDataset(test_data, test_mask)

train_iterator= DataLoader(train_dataset, **params)
vaild_iterator=DataLoader(vaild_dataset,**params)
test_iterator=DataLoader(test_dataset,**params)

class StackedTransformer(nn.Module):
    def __init__(self, response_size=47, treatment_size=2, d_model=512, d_ff=512, activation='gelu',
                 mix=True, d_layers=3, n_heads=8, e_layers=3, distil=True, repeat_t=0, hidden_r=512, hidden_t=512,
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

    def forward(self, x_r, x_t,thr, is_train=True, gen_length=None):
        device = x_r.device
        batch_size = x_r.shape[0]

        r_t = torch.cat([x_r, x_t], dim=-1)
        if not is_train:
            x_r = x_r[:, :24, :]
            x_t = x_t[:, :24, :]
            r_t = torch.cat([x_r, x_t], dim=-1)

            for i in range(1):
                enc_out = self.enc_embedding(r_t)
                enc_out, attns = self.encoder(enc_out, attn_mask=None)

                r_dec_out = self.r_dec_embedding(x_r, None)
                r_dec_out = self.r_decoder(r_dec_out, enc_out, None, None)
                r_output = self.r_projection(r_dec_out)

                t_dec_out = self.t_dec_embedding(x_t, None)
                t_dec_out = self.t_decoder(t_dec_out, enc_out, None, None)
                t_output = self.t_projection(t_dec_out)
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

                t_output = torch.mul(t_output, (cnn_features >= thr).float())

                x_r = torch.cat([x_r, r_output], dim=1)
                x_t = torch.cat([x_t, t_output], dim=1)
                new_r_t = torch.cat([r_output, t_output], dim=2)
                r_t = torch.cat([r_t, new_r_t], dim=1)
            return x_r[:, :, :], x_t[:, :, :]


        enc_out = self.enc_embedding(r_t)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        r_dec_out = self.r_dec_embedding(x_r, None)
        r_dec_out = self.r_decoder(r_dec_out, enc_out, None, None)
        r_output = self.r_projection(r_dec_out)

        t_dec_out = self.t_dec_embedding(x_t, None)
        t_dec_out = self.t_decoder(t_dec_out, enc_out, None, None)
        t_output = self.t_projection(t_dec_out)

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
        return r_output, t_output, cnn_result


def test(iterator, model, is_mask, response_size=47, treatment_size=2,
                       device=torch.device("cpu")):
    model = model.to(device)

    loss_mae = nn.L1Loss()

    thr1s = np.arange(0, 1.01, 0.05)
    thr2s = np.arange(0, 1.01, 0.05)
    X, Y = np.meshgrid(thr1s, thr2s)
    if not os.path.exists('threshold_mae.pt'):
        model = model.eval()
        x_tensor = torch.from_numpy(X).to(device)
        y_tensor = torch.from_numpy(Y).to(device)
        z_tensor = torch.stack((x_tensor, y_tensor)).permute(1, 2, 0)
        result = torch.zeros_like(x_tensor).to(device)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                thr = z_tensor[i, j, :]
                num_batch = 0
                mae = 0

                with torch.no_grad():
                    for b, batch in enumerate(iterator):
                        num_batch += 1
                        data_y, mask= batch

                        data_x_r = data_y[:, :24, :response_size].to(device)
                        data_x_t = data_y[:, :24, -treatment_size:].to(device)
                        data_y = data_y.to(device)
                        mask = mask.to(device)

                        length = data_y.shape[1]

                        pred_r, pred_t = model(data_x_r, data_x_t, thr, is_train=False, gen_length=length)
                        if is_mask:
                            pred_t = torch.mul(pred_t, mask[:, :48, -treatment_size:])
                        mae += loss_mae(pred_t, data_y[:, :48, -treatment_size:]).item()

                mae /= num_batch
                result[i, j] = mae
        torch.save(result, 'threshold_mae.pt')
    matplotlib.use('TkAgg')
    result = torch.load('threshold_mae.pt').cpu()
    result=result.numpy()
    thr1=result[:,6]
    thr2=result[14,:]
    plt.xlabel("threshold for treatment 1")
    plt.ylabel("mae of Treatment")
    plt.plot(thr1s,thr1,'o-m')
    plt.savefig('draw2d2.png', dpi=300)
    #plt.show()
    plt.rcParams["figure.figsize"] = [8, 8]
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(X, Y, result, rstride=1, cstride=1, cmap='rainbow')
    ax3.set(xlabel='threshold for Treatment 1', ylabel='threshold for treatment 2', zlabel='mae of Treatment')
    ax3.view_init(azim=220,elev=20)
    # plt.show()
    plt.savefig('draw3d.png', dpi=300)

if __name__ == "__main__":

    if args.model == 'encoder_decoder':

            model = StackedTransformer(response_size=response_size, treatment_size=treatment_size)
            model.load_state_dict(torch.load("trained_models/transformer/2022_07_25_10_49_40/model.pt"))

            test(iterator=test_iterator, model=model, is_mask=args.mask, response_size=response_size, treatment_size=treatment_size, device=device)


    else:
        print('Error: Invalid model.')
