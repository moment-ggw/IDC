from turtle import position
from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward, position_embedding
import torch
import numpy as np
from torch import nn
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table


class SelfAttLayer(nn.Module):
    def __init__(self, d_model=768, d_k=64, d_v=64, h=12, d_ff=2048, dropout=.1, mode=None):
        super(SelfAttLayer, self).__init__()
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, return_att=True, can_be_stateful=True)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.mode = mode
        
    def forward(self, q, k ,v, attention_mask=None, attention_weights=None):
        att, att_map = self.mhatt(q, k, v, attention_mask, attention_weights)
        
        ff = self.pwff(att)
        return ff, att_map
    
        if self.mode == 'slow':
            return att, att_map
        elif self.mode == 'fast':
            ff = self.pwff(att)
            return ff

class CrossAttLayer(nn.Module):
    def __init__(self, d_model=768, d_k=64, d_v=64, h=12, d_ff=2048, dropout=.1):
        super(CrossAttLayer, self).__init__()
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, return_att=True, can_be_stateful=False, residual=True)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        
    def forward(self, q, k ,v, attention_mask=None, attention_weights=None):
        att, att_map = self.mhatt(q, k, v, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff, att_map