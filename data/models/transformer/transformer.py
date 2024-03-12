from tkinter import image_names
import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, rectify):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.rectify = rectify
        self.without_rectifier = False if self.rectify is not None else True
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()


    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
        # images = self.add_global_feature(images)
        enc_output, mask_enc = self.encoder(images)
        dec_output, decoder_softout, decoder_outfeature, decoder_mask = self.decoder(seq, enc_output, mask_enc)
        if not self.without_rectifier:
            rectify_output = self.rectify(enc_output, mask_enc, decoder_softout, decoder_outfeature, decoder_mask)
            return dec_output, rectify_output
        else:
            return dec_output, dec_output
    
    def train_scst(self, images, seq_len, eos_id, beam_size, out_size):
        outs, log_probs = self.beam_search(images, seq_len, eos_id, beam_size, out_size, return_probs=False)
        
        return outs, log_probs

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output
        dec_output, decoder_softout, decoder_outfeature, decoder_mask = self.decoder(it, self.enc_output, self.mask_enc)
        if not self.without_rectifier:
            rectify_output = self.rectify(self.enc_output, self.mask_enc, decoder_softout, decoder_outfeature, decoder_mask)
            return rectify_output
        else:
            return dec_output

    def add_global_feature(self, images):
        len = images.shape[1]
        mask = (torch.sum(images, -1) != 0).float()
        num = torch.sum(mask, -1).unsqueeze(1)
        global_images = (torch.sum(images, 1) / num).unsqueeze(1)
        images = torch.cat((global_images, images), 1)
        return images[:, :len]

class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)

