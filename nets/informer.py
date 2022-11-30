import torch
import torch.nn as nn

from nets.attn import FullAttention, AttentionLayer
from nets.decoder import Decoder, DecoderLayer
from nets.embed import DataEmbedding
from nets.encoder import Encoder, EncoderLayer, ConvLayer


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()

        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        # todo
        # Attn = ProbAttention if attn == 'prob' else FullAttention
        Attn = FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        # self.projection_1 = nn.Linear(d_model, c_out_1, bias=True)
        
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, out_len,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, is_train=True,bias=None):


        if not is_train:
            x_dec = x_enc[:, :, :]
            x_enc = x_enc[:, :, :]
            out_list = []
            for i in range(out_len):
                enc_out = self.enc_embedding(x_enc, x_mark_enc)
                enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
                dec_out = self.dec_embedding(x_dec, x_mark_dec)
                dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
                dec_out = self.projection(dec_out)[:, -1:, :]
                if bias!=None:
                    dec_out[:, -1:, :47] = dec_out[:, -1:, :47]+bias[:,i:i+1,:]
                x_dec = torch.cat((x_dec, dec_out[:, -1:, :]), 1)
                out_list.append(dec_out[:, -1:, :])
            return torch.cat(out_list, 1)
            return dec_out
        
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)


        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -out_len:, :], attns
        else:
            return dec_out[:, -out_len:, :]  # [B, L, D]