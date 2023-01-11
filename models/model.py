import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

# Obsismc: autoformer
from models.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.Autoformer_EncDec import series_decomp


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
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
                    d_ff,
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# author: Obsismc
class DownSampleLayer(nn.Module):
    def __init__(self, down_sample_scale, d_model):
        super(DownSampleLayer, self).__init__()
        self.localConv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=1)
        self.down_sample_norm = nn.BatchNorm1d(d_model)
        self.down_sample_activation = nn.ELU()
        self.localMax = nn.MaxPool1d(kernel_size=down_sample_scale)

    def forward(self, x: torch.Tensor):
        """

        :param x: (B,L,D)
        :return: (B,L/self.down_sample_scale,D)
        """
        x = self.localConv(x.permute(0, 2, 1))
        x = self.down_sample_norm(x)
        x = self.down_sample_activation(x)
        x = self.localMax(x)
        return x.permute(0, 2, 1)


# author: Obsismc
class UpSampleLayer(nn.Module):
    def __init__(self, down_sample_scale, d_model, padding=1, output_padding=1):
        super(UpSampleLayer, self).__init__()
        self.proj = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.upSampleNorm = nn.LayerNorm(d_model)

        kern_size = down_sample_scale + 2 * padding - output_padding  # formula of ConvTranspose1d
        self.upSample = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, padding=padding,
                                           kernel_size=kern_size, stride=down_sample_scale,
                                           output_padding=output_padding)  # need to restore the length
        self.upActivation = nn.ELU()

    def forward(self, x):
        """

        :param x: (B,L,D)
        :return: (B,self.down_sample_scale * L,D)
        """
        x = self.proj(x.permute(0, 2, 1))
        x = self.upSampleNorm(x.transpose(2, 1))
        x = self.upSample(x.transpose(2, 1))
        x = self.upActivation(x)
        return x.transpose(2, 1)


# Winformer module
class WInformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(WInformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # U-net part
        # Important: can only handle even
        # down sample step
        self.down_sample_n = 2  # depth
        self.down_sample_scale = 2
        self.downSamples = nn.ModuleList(
            [DownSampleLayer(down_sample_scale=self.down_sample_scale, d_model=d_model) for _ in
             range(self.down_sample_n)]
        )
        self.downSamples.append(nn.Identity())
        # up sample step: refer to Yformer's method
        self.upSamples = nn.ModuleList(
            [UpSampleLayer(down_sample_scale=self.down_sample_scale, d_model=d_model) for _ in
             range(self.down_sample_n)])
        self.upSamples.insert(0, nn.Identity())
        self.finalNorm = nn.LayerNorm(d_model)

        # Encoding
        # obsismc: out_channel->d_model, output dim: (B, seq_len, D)
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
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
                    d_ff,
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # encoding
        enc_down_sampled = self.enc_embedding(x_enc, x_mark_enc)
        # decoding
        dec_down_sampled = self.dec_embedding(x_dec, x_mark_dec)

        dec_outs = []
        attns = None
        for i in range(self.down_sample_n + 1):
            # get encoding attention
            enc_out_cross, _ = self.encoder(enc_down_sampled, attn_mask=enc_self_mask)

            # cross attention
            dec_out_tmp = self.decoder(dec_down_sampled, enc_out_cross, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            dec_outs.append(dec_out_tmp)

            # get down sampled embedding
            enc_down_sampled = self.downSamples[i](enc_down_sampled)
            dec_down_sampled = self.downSamples[i](dec_down_sampled)

        # up sampling step
        for i in range(self.down_sample_n, 0, -1):
            dec_outs[i - 1] += self.upSamples[i](dec_outs[i])

        dec_out = dec_outs[0]
        dec_out = self.finalNorm(dec_out)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# WAutoformer module, only with Auto-Correlation, without series decomp
class WAutoCorrelation(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(WAutoCorrelation, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # U-net part
        # Important: can only handle even
        # down sample step
        self.down_sample_n = 2  # depth
        self.down_sample_scale = 2
        self.downSamples = nn.ModuleList(
            [DownSampleLayer(down_sample_scale=self.down_sample_scale, d_model=d_model) for _ in
             range(self.down_sample_n)]
        )
        self.downSamples.append(nn.Identity())
        # up sample step: refer to Yformer's method
        self.upSamples = nn.ModuleList(
            [UpSampleLayer(down_sample_scale=self.down_sample_scale, d_model=d_model) for _ in
             range(self.down_sample_n)])
        self.upSamples.insert(0, nn.Identity())
        self.finalNorm = nn.LayerNorm(d_model)

        # Encoding
        # obsismc: out_channel->d_model, output dim: (B, seq_len, D)
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = AutoCorrelation
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                         d_model, n_heads),
                    AutoCorrelationLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False),
                                         d_model, n_heads),
                    d_model,
                    d_ff,
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_down_sampled = enc_out
        enc_outs = []
        attns = None
        for i in range(self.down_sample_n + 1):
            # get encoding attention
            enc_out_tmp, _ = self.encoder(enc_down_sampled, attn_mask=enc_self_mask)
            enc_outs.append(enc_out_tmp)

            # get down sampled embedding
            enc_down_sampled = self.downSamples[i](enc_down_sampled)

        # decoding
        # down sampling step
        dec_down_sampled = dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_outs = []
        for i in range(self.down_sample_n + 1):
            # cross attention
            dec_out_tmp = self.decoder(dec_down_sampled, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            dec_outs.append(dec_out_tmp)

            # down sampling
            dec_down_sampled = self.downSamples[i](dec_down_sampled)

        # up sampling step

        for i in range(self.down_sample_n, 0, -1):
            dec_up_sampled = self.upSamples[i](dec_outs[i])
            dec_outs[i - 1] += dec_up_sampled

        dec_out = dec_outs[0]
        dec_out = self.finalNorm(dec_out)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# Yinformer module
class Yinformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Yinformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # U-net part
        # Important: can only handle even
        # down sample step
        self.down_sample_n = 2  # depth
        self.down_sample_scale = 2
        self.downSamples = nn.ModuleList(
            [DownSampleLayer(down_sample_scale=self.down_sample_scale, d_model=d_model) for _ in
             range(self.down_sample_n)]
        )
        # up sample step: refer to Yformer's method
        self.upSamples = nn.ModuleList(
            [UpSampleLayer(down_sample_scale=self.down_sample_scale, d_model=d_model) for _ in
             range(self.down_sample_n)])
        self.finalNorm = nn.LayerNorm(d_model)

        # Encoding
        # obsismc: out_channel->d_model, output dim: (B, seq_len, D)
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # U attention
        self.bottom_attn = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads, mix=False)
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
                ) for l in range(e_layers)
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
                    d_ff,
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # encoding
        enc_down_sampled = self.enc_embedding(x_enc, x_mark_enc)
        # decoding
        dec_down_sampled = self.dec_embedding(x_dec[:, -self.pred_len:, ...], x_mark_dec[:, -self.pred_len:, ...])

        # down sampling
        enc_crosses = []
        attns = None
        for i in range(self.down_sample_n):
            # get cross
            enc_down_sampled, _ = self.encoder(enc_down_sampled, attn_mask=enc_self_mask)
            dec_down_sampled, _ = self.encoder(dec_down_sampled, attn_mask=enc_self_mask)

            enc_cross = torch.concat([enc_down_sampled, dec_down_sampled], dim=1)
            enc_crosses.append(enc_cross)

        # up sampling
        dec_up_sampled = enc_crosses[-1]
        dec_up_sampled, _ = self.bottom_attn(dec_up_sampled, dec_up_sampled, dec_up_sampled, None)
        for i in range(self.down_sample_n - 1, -1, -1):
            # cross attention
            dec_up_sampled = self.decoder(dec_up_sampled, enc_crosses[i], x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            dec_up_sampled = self.upSamples[i](dec_up_sampled)

        dec_out = self.projection(dec_up_sampled)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
