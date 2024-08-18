import copy
import json
import math
import re
import collections

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert (
                self.head_dim * num_heads == feature_dim
        ), "feature_dim must be divisible by num_heads"

        self.linear_query = nn.Linear(feature_dim, feature_dim)
        self.linear_key = nn.Linear(feature_dim, feature_dim)
        self.linear_value = nn.Linear(feature_dim, feature_dim)
        self.linear_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size = x.size(0)
        # print(batch_size)
        # Treat each sample in the batch as an independent sequence of length 1
        # x = x.unsqueeze(1)  # Adding sequence length dimension: [Batch Size, 1, Feature Dim]

        # Linear projections
        query = self.linear_query(x)
        key = self.linear_key(x)
        value = self.linear_value(x)

        # Reshape for multi-head attention
        query = query.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key = key.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        value = value.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)

        # Combine the attention and the values
        out = torch.matmul(attention, value)
        out = out.transpose(0, 1).contiguous().view(batch_size, self.feature_dim)

        # Final linear layer
        out = self.linear_out(out)

        return out  # Remove the sequence length dimension


class MLP(nn.Module):
    def __init__(self, n_state, cfg):
        super(MLP, self).__init__()
        self.c_fc = Conv1D(n_state, 1, n_state)
        self.c_proj = Conv1D(n_state, 1, n_state)
        self.act = ACT_FNS[cfg.SOLVER.STAGE3.AFN]
        self.dropout = nn.Dropout(cfg.SOLVER.STAGE3.RESID_PDROP)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, cfg, dim=1024, num_heads=8):
        super(Block, self).__init__()
        self.attn = Attention(feature_dim=dim, num_heads=num_heads)
        self.ln_1 = LayerNorm(dim)
        self.mlp = MLP(dim, cfg)
        self.ln_2 = LayerNorm(dim)


    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class InterImageCrossAttention(nn.Module):
    """ InterImageCrossAttention """

    def __init__(self, cfg, dim=1024, num_heads=8, num_classes=751):

        super(InterImageCrossAttention, self).__init__()
        self.num_classes = num_classes
        block = Block(cfg=cfg, dim=dim, num_heads=num_heads)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.SOLVER.STAGE3.LAYER)])

        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.layer = cfg.SOLVER.STAGE3.LAYER

    def forward(self, x):
        for block in self.h:
            x = block(x)
        return x

