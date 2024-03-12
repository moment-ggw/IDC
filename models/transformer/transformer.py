from tkinter import image_names
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
from models.transformer.utils import sinusoid_encoding_table

from models.transformer.layers import SelfAttLayer, CrossAttLayer
from models.transformer.utils import PositionWiseFeedForward
import numpy as np
from transformers import BertModel, BertForMaskedLM, BertConfig
from transformers import GPT2Tokenizer, GPT2Model

class FaseAndSlow(nn.Module):
    def __init__(self, args, d_model=768, d_k=64, d_v=64, h=12, d_ff=2048, dropout=.1, **kwargs):
        super(FaseAndSlow, self).__init__()
        self.d_model = d_model
        self.N = args.N
        self.img_selfatt = nn.ModuleList([SelfAttLayer(d_model, d_k, d_v, h, d_ff, dropout, 'fast') for _ in range(self.N[0])])
        self.text_selfatt = nn.ModuleList([SelfAttLayer(d_model, d_k, d_v, h, d_ff, dropout, 'slow') for _ in range(self.N[1])])
        self.text_crossatt = nn.ModuleList([CrossAttLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(self.N[2])])
        self.use_imgpt = args.use_imgpt
        self.use_txtpt = args.use_txtpt
        self.padding_idx = args.padding_id
        self.args = args
        self.start_gen = False
        
        if not args.use_imgpt:
            self.patchs = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
        else:
            self.img_fn = nn.Linear(args.image_dim, self.d_model)
        self.img_dropout = nn.Dropout(p=dropout)
        self.img_layernorm = nn.LayerNorm(self.d_model)
            
        if args.use_txtpt:
            try: 
                self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            except OSError:
                self.bert_model = BertModel.from_pretrained('/raid/ggw/cross-modal-retrieval/BERT_file/', return_dict=True)
        else:
            self.word_embedding = nn.Embedding(args.vocab_size, d_model, padding_idx=self.padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(args.max_len + 1, d_model, 0), freeze=True)
        self.txt_layernorm = nn.LayerNorm(self.d_model)
            
        self.cap_fn = nn.Linear(self.d_model, args.vocab_size, bias=False)
        
    def forward(self, images, texts):   
        
        images = self.img_enc(images)
        texts_fwd, texts_bwd = self.text_enc(texts)
        texts = texts_fwd + texts_bwd
        cap_hid_fwd = self.text_dec(images, texts_fwd)
        
        cap_fwd = self.cap_fn(cap_hid_fwd)
        
        return images, texts, cap_hid_fwd, cap_fwd
    
    def generate(self, images, texts, bos_id, eos_id, max_len, rerank=False):
        
        images = self.img_enc(images)
        texts_fwd, texts_bwd = self.text_enc(texts)
        
        if rerank == False:
            return images, texts_fwd + texts_bwd
        
        bs = images.shape[0]
        self.enable_gen(bs)
        cap_fwd_list = []
        for i in range(max_len):
            if i == 0:
                txt = torch.ones((bs, 1)).long().to(images.device) * bos_id
                text_finish = (txt != eos_id).long()
            texts_gen_fwd = self.text_enc(txt)
            texts_hid_fwd = self.text_dec(images, texts_gen_fwd)
            cap_fwd = self.cap_fn(texts_hid_fwd)
            cap_fwd_lgsm = F.log_softmax(cap_fwd, dim=-1)
            txt = torch.argmax(cap_fwd_lgsm, dim=-1)
            
            txt *= text_finish
            text_finish *= (txt != eos_id).long()
            
            cap_fwd_list.append(cap_fwd)
        self.disable_gen()
        
        cap_fwd = torch.cat(cap_fwd_list, dim=1)
        
        return cap_fwd
    
    def img_enc(self, images):
        
        if not self.args.use_imgpt:
            bs = images.shape[0]
            images = self.patchs(images)
            images = images.view(bs, self.d_model, -1).permute(0, 2, 1)
        else:
            images = self.img_dropout(F.relu(self.img_fn(images)))
        images = self.img_layernorm(images)
        
        for layer in self.img_selfatt:
            images, i2i = layer(images, images, images)
        return images
    
    def text_enc(self, texts):
        bs, seq_len = texts.shape[:]
        self.texts_mask = texts == self.padding_idx   
        
        if self.start_gen:
            self.running_texts_mask = torch.cat([self.running_texts_mask.to(texts.device), self.texts_mask], dim=-1)
            self.running_seq += 1
            seq = self.running_seq
        else:
            seq = torch.arange(1, seq_len + 1)
        seq = seq.view(1, -1).expand(bs, -1).to(texts.device)
        seq = seq.masked_fill(self.texts_mask, 0)
        
        
        if self.args.use_txtpt:
            attention_mask = (self.texts_mask == False).float()
            token_type_ids = torch.zeros_like(texts).long()
            texts = self.bert_model(input_ids=texts,
                                    attention_mask=attention_mask, 
                                    token_type_ids=token_type_ids)
            
            texts = texts.last_hidden_state + self.pos_emb(seq)
        else:
            texts = self.word_embedding(texts) + self.pos_emb(seq)
        texts = self.txt_layernorm(texts)
        
        if self.start_gen:
            seq_len = self.running_seq[0]
            texts_fwd_mask = self.running_texts_mask.unsqueeze(1)
        else:
            texts_fwd_mask = torch.triu(torch.ones((seq_len, seq_len), device=texts.device), diagonal=1).bool().unsqueeze(0) + self.texts_mask.unsqueeze(1)
            texts_bwd_mask = (torch.tril(torch.ones((seq_len, seq_len), device=texts.device), diagonal=-1).bool().unsqueeze(0) + self.texts_mask.unsqueeze(1)) * (self.texts_mask == False).unsqueeze(-1)
        texts_fwd, text_bwd = texts, texts
        for layer in self.text_selfatt:
            texts_fwd, t2t_fwd = layer(texts_fwd, texts_fwd, texts_fwd, attention_mask=texts_fwd_mask.unsqueeze(1))
            texts_fwd = texts_fwd * (self.texts_mask == False).unsqueeze(-1).float()
            
        if not self.start_gen:
            for layer in self.text_selfatt:
                text_bwd, t2t_bwd = layer(text_bwd, text_bwd, text_bwd, attention_mask=texts_bwd_mask.unsqueeze(1))
                text_bwd = text_bwd * (self.texts_mask == False).unsqueeze(-1).float()
            return texts_fwd, text_bwd
        
        return texts_fwd
    
    def text_dec(self, images, texts_fwd):
        text_mask = (self.texts_mask == False).unsqueeze(-1).float()
        
        for layer in self.text_crossatt:
            texts_fwd, t2i_fwd = layer(texts_fwd, images, images)
            texts_fwd = texts_fwd * text_mask
        
        return texts_fwd

    def enable_gen(self, bs):
        self.start_gen = True
        self.running_texts_mask = torch.zeros((bs, 0)).bool()
        self.running_seq = torch.zeros((1)).long()
        for layer in self.text_selfatt:
            layer.mhatt._is_stateful = True
            layer.mhatt.running_keys = layer.mhatt.running_keys.expand(bs, 0, self.d_model)
            layer.mhatt.running_values = layer.mhatt.running_values.expand(bs, 0, self.d_model)
    def disable_gen(self):
        self.start_gen = False
        self.running_texts_mask = torch.zeros((0, 1)).bool()
        self.running_seq = torch.zeros((1)).long()
        for layer in self.text_selfatt:
            layer.mhatt._is_stateful = False
            layer.mhatt.running_keys = torch.zeros((1, 0, self.d_model))
            layer.mhatt.running_values = torch.zeros((1, 0, self.d_model))
            

class ShareFAS(nn.Module):
    def __init__(self, args, d_model=768, d_k=64, d_v=64, h=12, d_ff=2048, dropout=.1, **kwargs):
        super(ShareFAS, self).__init__()
        self.d_model = d_model
        self.N = args.N
        self.img_encoder = nn.ModuleList([SelfAttLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(self.N)])
        self.txt_encoder = nn.ModuleList([SelfAttLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(self.N)])
        self.share_encoder = nn.ModuleList([SelfAttLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(self.N)])
        self.use_imgpt = args.use_imgpt
        self.use_txtpt = args.use_txtpt
        self.padding_idx = args.padding_idx
        self.args = args
        
        if not args.use_imgpt:
            self.patchs = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
            self.layer_norm = nn.LayerNorm(self.d_model)
        else:
            self.img_fn = nn.Linear(args.image_dim, self.d_model)
            
        if args.use_txtpt:
            # config = BertConfig.from_json_file('bert.json')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            # self.txt_fn = nn.Linear(self.d_model, self.d_model)
        else:
            self.word_embedding = nn.Embedding(args.vocab_size, d_model, padding_idx=self.padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(args.max_len + 1, d_model, 0), freeze=True)
    
    def forward(self, images, texts):
        images = self.img_enc(images)
        texts = self.txt_enc(texts)
        share_images, share_texts = self.share_enc(images, texts)
        return images, texts, share_images, share_texts
    
    def img_enc(self, images):
        if not self.args.use_imgpt:
            bs = images.shape[0]
            images = self.patchs(images)
            images = images.view(bs, self.d_model, -1).permute(0, 2, 1)
            images = self.layer_norm(images)
        else:
            images = self.img_fn(images)
        
        for layer in self.img_encoder:
            images, i2i = layer(images, images, images)
        
        return images
    
    def txt_enc(self, texts):
        bs, seq_len = texts.shape[:]
        self.texts_mask = texts == self.padding_idx    
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(bs, -1).to(texts.device)
        seq = seq.masked_fill(self.texts_mask, 0)
        
        if self.args.use_txtpt:
            attention_mask = torch.ones_like(texts)
            token_type_ids = torch.zeros_like(texts).long()
            texts = self.bert_model(input_ids=texts,
                                    attention_mask=attention_mask, 
                                    token_type_ids=token_type_ids)

            texts = texts.last_hidden_state + self.pos_emb(seq)
        else:
            texts = self.word_embedding(texts) + self.pos_emb(seq)
        
        texts_mask = self.texts_mask.unsqueeze(1).unsqueeze(1)
        texts_pad_mask = (self.texts_mask == False).unsqueeze(-1).float()
        for layer in self.txt_encoder:
            texts, t2t = layer(texts, texts, texts, texts_mask)
            texts = texts * texts_pad_mask
        
        return texts
    
    def share_enc(self, images, texts):
        
        for layer in self.share_encoder:
            images, i2i = layer(images, images, images)
        
        texts_mask = self.texts_mask.unsqueeze(1).unsqueeze(1)
        texts_pad_mask = (self.texts_mask == False).unsqueeze(-1).float()
        for layer in self.share_encoder:
            texts, t2t = layer(texts, texts, texts, texts_mask)
            texts = texts * texts_pad_mask
        
        return images, texts