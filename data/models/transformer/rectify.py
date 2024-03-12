from os import replace
from re import A
import torch
from torch import _cudnn_init_dropout_state, layer_norm, nn
from torch.nn import functional as F
import numpy as np

from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.beam_search import *
from models.containers import Module, ModuleList



class RectifyCaptionLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(RectifyCaptionLayer, self).__init__()
        
        self.selfattention = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, dropout=dropout, can_be_stateful=True) # True
        self.pwff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.crossatt_encoder = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, dropout=dropout, can_be_stateful=False)
        self.crossatt_decoder = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, dropout=dropout, can_be_stateful=True) # True
        
        self.fc_encoder = nn.Linear(d_model * 3, d_model)
        self.fc_decoder = nn.Linear(d_model * 3, d_model)
        self.alpha_encoder = nn.Linear(d_model * 2, d_model)
        self.alpha_decoder = nn.Linear(d_model * 2, d_model)
        
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.alpha_encoder.weight)
        nn.init.xavier_uniform_(self.alpha_decoder.weight)
        nn.init.xavier_uniform_(self.alpha_encoder.weight)
        nn.init.xavier_uniform_(self.alpha_decoder.weight)
        nn.init.constant_(self.alpha_encoder.bias, 0)
        nn.init.constant_(self.alpha_decoder.bias, 0)
        nn.init.constant_(self.alpha_encoder.bias, 0)
        nn.init.constant_(self.alpha_decoder.bias, 0)

        
    def forward(self, captions, encoder_out, decoder_out, encoder_mask, decoder_mask, mask_self_att):
        self_att = self.selfattention(captions, captions, captions, mask_self_att)
        self_att = self_att * decoder_mask
        
        encoder_out = torch.cat([encoder_out[:, i, :, :] for i in range(encoder_out.shape[1])], -1)
        encoder_out = F.relu(self.fc_encoder(encoder_out))
        enc_att = self.crossatt_encoder(self_att, encoder_out, encoder_out, encoder_mask) * decoder_mask
        
        decoder_out = torch.cat([decoder_out[:, i, :, :] for i in range(decoder_out.shape[1])], -1)
        decoder_out = F.relu(self.fc_decoder(decoder_out))
        dec_att = self.crossatt_decoder(self_att, decoder_out, decoder_out, mask_self_att)
        
        alpha_encoder = torch.sigmoid(self.alpha_encoder(torch.cat([self_att, enc_att], -1)))
        alpha_decoder = torch.sigmoid(self.alpha_decoder(torch.cat([self_att, dec_att], -1)))
        
        out = (alpha_encoder * enc_att + alpha_decoder * dec_att) / np.sqrt(2)

        out = out * decoder_mask
        out = self.pwff(out) * decoder_mask
        
        return out
    
class RectifyCaption(Module):  
    def __init__(self, N, vocab_size, max_len, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1):
        super(RectifyCaption, self).__init__()
        
        self.layers = ModuleList([RectifyCaptionLayer(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, dropout=dropout) for _ in range(N)])
        
        # 考虑softwordembedding需不需要偏置
        self.word_embedding = nn.Linear(vocab_size, d_model, bias=False)
        
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.fc_v2w = nn.Linear(d_model, vocab_size, bias=False)

        self.register_state('running_mask_self_att', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())
        
        
    def forward(self, encoder_out, encoder_mask, decoder_softout, decoder_out, decode_mask):
        '''
        images:(b_s, img_num, img_dimention)                padding = 0
        captions:(b_s, cap_len, vocal_size)                 
        labels:(b_s, img_num)                               padding = self.padding_idx
        '''
        b_s, seq_len = decoder_softout.shape[0:2]
        mask_self_att = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=decoder_softout.device), diagonal=1) # -torch.eye(seq_len)
        mask_self_att = mask_self_att.unsqueeze(0).unsqueeze(0)
        mask_self_att = mask_self_att + (decode_mask==0).squeeze(-1).unsqueeze(1).unsqueeze(1)
        mask_self_att = mask_self_att.gt(0)
        
        if self._is_stateful:
            self.running_mask_self_att = torch.cat([self.running_mask_self_att, mask_self_att], -1)
            mask_self_att = self.running_mask_self_att
        
        
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(decoder_softout.device)  # (b_s, seq_len)
        seq = seq.masked_fill(decode_mask.squeeze(-1) == 0, 0)
        
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
        
        out = self.word_embedding(decoder_softout) + self.pos_emb(seq)
        
        # encoder_out = encoder_out
        for layer in self.layers:
            out = layer(out, encoder_out, decoder_out, encoder_mask, decode_mask, mask_self_att)
        
        out = self.fc_v2w(out)
        return F.log_softmax(out, dim=-1)
    
    