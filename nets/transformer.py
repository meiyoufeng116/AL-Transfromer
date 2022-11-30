# from turtle import forward
import torch.nn as nn
import torch

from nets.encoder import Encoder, EncoderLayer, ConvLayer
from nets.decoder import Decoder, DecoderLayer
from nets.embed import DataEmbedding
from nets.attn import AttentionLayer, FullAttention


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

    def forward(self, x_r, x_t, is_train=True, gen_length=None):
        device = x_r.device
        batch_size = x_r.shape[0]
        end=24
        middle=12
        r_t = torch.cat([x_r, x_t], dim=-1)
        if not is_train:
            x_r = x_r[:, :middle, :]
            x_t = x_t[:, :middle, :]
            r_t = torch.cat([x_r, x_t], dim=-1)

            for i in range(gen_length):
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

        for i in range(gen_length):
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
        return x_r[:, -middle:, :], x_t[:, -middle:, :],cnn_result[:, -middle:, :]
        return r_output, t_output, cnn_result


class Transformer(nn.Module):
    def __init__(self, response_size=47, treatment_size=2, d_model=512, d_ff=512, activation='gelu',
                 mix=True, d_layers=2, n_heads=8, e_layers=3, distil=True, repeat_t=0, hidden_r=512, hidden_t=512,
                 hidden_cnn=256, window_sizes=[3, 5, 7], num_classes=2, window=12, dropout=0):
        super(Transformer, self).__init__()
        self.response_size = response_size
        self.treatment_size = treatment_size

        self.window = window
        self.num_classes = num_classes

        self.enc_embedding = DataEmbedding(
            response_size + treatment_size, d_model, dropout)
        # self.r_dec_embedding = DataEmbedding(response_size, d_model, dropout)
        # self.t_dec_embedding = DataEmbedding(treatment_size, d_model, dropout)
        self.dec_embedding = DataEmbedding(
            response_size + treatment_size, d_model, dropout)

        self.encoderlayer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, activation=activation)
        self.encoder = nn.TransformerEncoder(
            self.encoderlayer, num_layers=e_layers)

        self.decoderlayer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, activation=activation)
        self.decoder = nn.TransformerDecoder(
            self.decoderlayer, num_layers=d_layers)

        self.linear = nn.Linear(d_model, response_size+treatment_size)

    def forward(self, enc_in, enc_mask, dec_in, dec_mask, out_len, is_train=True,bias=None):
        if not is_train:
            dec_in = enc_in[:, :, :].permute(1,0,2)
            enc_in = enc_in[:, :, :].permute(1,0,2)
            out_list = []
            for i in range(out_len):
                enc_out = self.enc_embedding(enc_in, enc_mask)
                enc_out = self.encoder(enc_out)
                dec_out = self.dec_embedding(dec_in, dec_mask)
                dec_out = self.decoder(dec_out, enc_out)
                dec_out = self.linear(dec_out)[-1:, :, :]
                if bias!=None:
                    dec_out[-1:, :, :47]=dec_out[-1:, :, :47]+bias[:,i:i+1,:].permute(1,0,2)
                
                dec_in = torch.cat((dec_in, dec_out[-1:, :, :]), 0)
                enc_in = dec_in[-out_len:,:,:]
                out_list.append(dec_out[-1:, :, :])
            # return dec_out.permute(1,0,2)
            return torch.cat(out_list, 0).permute(1,0,2)
        
        enc_in=enc_in.permute(1,0,2)
        dec_in=dec_in.permute(1,0,2)
        enc_out = self.enc_embedding(enc_in, enc_mask)
        enc_out = self.encoder(enc_out)
        dec_out = self.dec_embedding(dec_in, dec_mask)
        dec_out = self.decoder(dec_out, enc_out)
        dec_out = self.linear(dec_out)
        dec_out = dec_out.permute(1,0,2)
        return dec_out[:, -out_len:, :]
