from turtle import position
from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward, position_embedding
import torch
import numpy as np
from torch import nn
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(500, d_model, 0), freeze=True)

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        bs, num = input.shape[0:2]
        seq = torch.arange(1, num + 1).unsqueeze(0).expand(bs, -1).to(input.device)
        seq = seq.masked_fill(attention_mask.squeeze(1).squeeze(1), 0)
        
        outs = []
        out = input + self.pos_emb(seq)
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask


class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, without_FVE=False, d_in=2048,**kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)
        self.without_FVE= without_FVE
        if not self.without_FVE:          
            self.fc_234 = nn.Linear(d_in // 2 + d_in // 4 + d_in // 8, self.d_model)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)


    def forward(self, input, attention_weights=None):
        input_234 = torch.cat(input[:-1], -1)
        if self.without_FVE:
            out = F.relu(self.fc(input[-1]))
        else:
            out = F.relu(self.fc(input[-1])) + F.relu(self.fc_234(input_234))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return super(MemoryAugmentedEncoder, self).forward(out, attention_weights=attention_weights)
