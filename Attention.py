import torch
import torch.nn as nn
from fc import FFN

## MultiHead Attention Module

class MultiHeadAttention(nn.Module):
    def __init__(self, dropout, num_hid, batchsize, nhead):
        super(MultiHeadAttention, self).__init__()

        self.proj_V = nn.Linear(num_hid, num_hid)
        self.proj_K = nn.Linear(num_hid, num_hid)
        self.proj_Q = nn.Linear(num_hid, num_hid)
        self.proj_out = nn.Linear(num_hid, num_hid)
        self.dropout = nn.Dropout(dropout)

    def attention(self, V, K, Q):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        att = F.softmax(scores, dim=-1)
        att = self.dropout(att)
        return torch.matmul(att, V)

    def forward(self, V, K, Q):
        batchsize = Q.size(0)
        V = self.proj_V(V).view(batchsize, -1, nhead, int(hidden_dim/nhead)).transpose(1, 2)
        K = self.proj_K(K).view(batchsize, -1, nhead, int(hidden_dim/nhead)).transpose(1, 2)
        Q = self.proj_Q(Q).view(batchsize, -1, nhead, int(hidden_dim/nhead)).transpose(1, 2)
        
        att = self.attention(V, K, Q)
        att = att.transpose(1, 2).contiguous().view(batchsize, -1, num_hid)
        att = self.proj_out(att)     
        return att

## Self Attention Module
class SA(nn.Module):
    def __init__(self, dropout, num_hid, batchsize, nhead):
        super(SA, self).__init__()

        self.mha_final = nn.Sequential(MultiHeadAttention(0.1, num_hid, batchsize, nhead),FFN(num_hid, num_hid, num_hid, drop_rate=dropout, use_relu=True))

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.LayerNorm(num_hid)),
            nn.Sequential(nn.Dropout(dropout), nn.LayerNorm(num_hid))
        ])

    def forward(self, sf):   
        for layer in self.mha_final:
            sf = layer(sf, sf, sf)
        for layer in self.layers:
            sf = sf + layer(sf)
        return sf

## Cross Modality Attention Module
class CA(nn.Module):
    def __init__(self, dropout, num_hid, batchsize, nhead):
        super(CA, self).__init__()

        self.mhatt = MultiHeadAttention(0.1, num_hid, batchsize, nhead)
        self.ffn = FFN(num_hid, num_hid, num_hid, drop_rate=dropout, use_relu=True)

        self.layers = nn.ModuleList([nn.Sequential(nn.Dropout(dropout), LayerNorm(num_hid)),nn.Sequential(nn.Dropout(dropout), LayerNorm(num_hid))])

    def forward(self, vf, tf, batchsize, nhead):
        vf = self.layers[0](vf + self.mhatt(tf, tf, vf))
        vf = self.layers[1](vf + self.ffn(vf))
        return vf

## Stacked of Self and Cross Attention on Dual Modlaity
class CSCA(nn.Module):
    def __init__(self, num_hid, nstack):
        super(CSCA, self).__init__()
        self.sa_features = nn.ModuleList([SA(0.1, num_hid, batchsize, nhead) for _ in range(nstack)])
        self.ca_features = nn.ModuleList([CA(0.1, num_hid, batchsize, nhead) for _ in range(nstack)])

    def forward(self, v, q, num_hid, batchsize, nhead):
        for sa, ca in zip(self.sa_features, self.ca_features):
            v = sa(v)
            q = sa(q)
            v = ca(v, q, batchsize, nhead)
            q = ca(q, v, batchsize, nhead)
        return v, q
