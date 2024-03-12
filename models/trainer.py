import collections
from itertools import count
from operator import length_hint
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR


class Trainer():
    def __init__(self, model, device, args) -> None:
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=args.padding_id, reduction="mean")
        self.mse = torch.nn.MSELoss()
        self.optim = AdamW(self.model.parameters(), lr=args.learn_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=30)
        self.it = 0
    
    def train(self, dataloader, e):
        self.model.train()
        total_loss = 0.
        images_list = []
        texts_gt_list = []
        self.pre_itmm_it = None
        with tqdm(desc='Epoch %d - train(patience=%d)' % (e, self.args.patience), unit='it', total=len(dataloader)) as pbar:
            for it, (img_id, images, texts_gt) in enumerate(dataloader):
                images, texts = images.to(self.device), texts_gt.to(self.device)
                
                self.it += 1
                images_list.append(images)
                texts_gt_list.append(texts)

                if it == 0: continue
                elif it > 1:
                    images_list.pop(0)
                    texts_gt_list.pop(0)

                images = torch.cat(images_list, dim=0)
                texts_gt = torch.cat(texts_gt_list, dim=0)

                if self.args.use_imgpt:
                    with torch.no_grad():
                        images = self.pt_process(images)
                
                images, texts, cap_hidden, cap_fwd = self.model(images, texts_gt)
                
                images = self.mean_l2(images)
                texts = self.mean_l2(texts[:, 1:])
                cap_hidden = self.mean_l2(cap_hidden[:, :-1].contiguous()) 
                
                self.optim.zero_grad()
                loss = self.loss(images, texts, cap_hidden, cap_fwd, texts_gt)
                
                loss.backward()
                self.optim.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / (it + 1))
                
                pbar.update()
             
        return total_loss / (it + 1)

    def val(self, dataloader, e):
        self.model.eval()
        img_id_list = []
        images_list = []
        texts_list = []
        with torch.no_grad():
            with tqdm(desc='Epoch %d - val' % e, unit='it', total=len(dataloader)) as pbar:
                for it, (img_id, images, texts_gt) in enumerate(dataloader):
                    images, texts = images.to(self.device), texts_gt.to(self.device)
                    
                    if self.args.use_imgpt:
                        images = self.pt_process(images)
                    
                    images, texts = self.model.generate(images, texts, self.args.bos_id, self.args.eos_id, texts.shape[1])                
                    
                    images = self.mean_l2(images)
                    texts = self.mean_l2(texts[:, 1:])
                    
                    img_id_list.append(img_id)
                    images_list.append(images)
                    texts_list.append(texts)
                    
                    pbar.update()
            img_id = torch.cat(img_id_list, dim=0)
            images = torch.cat(images_list, dim=0)
            texts = torch.cat(texts_list, dim=0)
            
            it_scores = self.metric(img_id, images, texts)
            print("Epoch {} - Validation scores(I2T): R@1: {}; R@5: {}; R@10: {}; Rsum: {}".format(e, it_scores["R@1"], it_scores["R@5"], it_scores["R@10"], it_scores["Rsum"]))
        
        return it_scores
    
    def test(self, dataloader, e):
        self.model.eval()
        img_id_list = []
        images_list = []
        texts_list = []
        with torch.no_grad():
            with tqdm(desc='Epoch %d - test' % e, unit='it', total=len(dataloader)) as pbar:
                for it, (img_id, images, texts_gt) in enumerate(dataloader):
                    images, texts = images.to(self.device), texts_gt.to(self.device)
                
                    if self.args.use_imgpt:
                        images = self.pt_process(images)   
                    images, texts = self.model.generate(images, texts, self.args.bos_id, self.args.eos_id, texts.shape[1])
                    
                    images = self.mean_l2(images)
                    texts = self.mean_l2(texts[:, 1:])
                    
                    img_id_list.append(img_id)
                    images_list.append(images)
                    texts_list.append(texts)
                    pbar.update()
                    
            img_id = torch.cat(img_id_list, dim=0)
            images = torch.cat(images_list, dim=0)
            texts = torch.cat(texts_list, dim=0)   
            
            it_scores = self.metric(img_id, images, texts)
            print("Epoch {} - Test scores(I2T): R@1: {}; R@5: {}; R@10: {}; Rsum: {}".format(e, it_scores["R@1"], it_scores["R@5"], it_scores["R@10"], it_scores["Rsum"]))
        
        return it_scores
       
    def load_pt(self, pt):
        self.img_pt = pt.visual
        self.img_pt.forward = self.img_pt.intermediate_features
        self.img_pt.to(self.device)
        self.img_pt.eval()

    def pt_process(self, images):
        images = self.img_pt(images)
        return images
    
    def loss(self, images, texts, cap_hidden, cap_fwd, texts_gt):
        img_txt_match_loss, itmm_it = self.match_loss(images, texts, self.args.gama_it)
        cap_txt_match_loss, itmm_ct = self.match_loss(cap_hidden, texts, self.args.gama_ct)
        img_cap_match_loss, itmm_ic = self.match_loss(images, cap_hidden, self.args.gama_ic)
        cap_loss = self.caploss(cap_fwd[:, :-1].contiguous(), texts_gt[:, 1:].contiguous()) 
        
        self.args.writer.add_scalar('loss/itloss', img_txt_match_loss, self.it)
        
        # self.args.writer.add_scalar('loss/dis_loss', dis_loss, self.it)

        loss = img_txt_match_loss 
        
        if self.args.with_cap:
            loss += cap_loss
            self.args.writer.add_scalar('loss/caploss', cap_loss, self.it)
        if self.args.with_ct:
            loss += cap_txt_match_loss
            self.args.writer.add_scalar('loss/ctloss', cap_txt_match_loss, self.it)
        if self.args.with_ic:
            loss += img_cap_match_loss
            self.args.writer.add_scalar('loss/icloss', img_cap_match_loss, self.it)

        if self.pre_itmm_it is not None:
            sdloss = self.sdloss(itmm_it)
            loss = loss + sdloss
            self.args.writer.add_scalar('loss/sdloss', sdloss, self.it)
        self.pre_itmm_it = itmm_it[self.args.batch_size:, self.args.batch_size:].permute(1, 0)
        return loss

    def mean_l2(self, data):
        data = torch.mean(data, dim=1)
        return data / data.norm(dim=-1, keepdim=True)
    
    def inm_loss(self, images, texts, gama=0.04):

        i2i = torch.exp(torch.matmul(images, images.permute(1, 0)) / gama)
        i_i = torch.diag(i2i)
        i2i = torch.sum(i2i, dim=-1)
        loss1 = - torch.mean(torch.log(i_i / i2i))

        t2t = torch.exp(torch.matmul(texts, texts.permute(1, 0)) / gama)
        t_t = torch.diag(t2t)
        t2t = torch.sum(t2t, dim=-1)
        loss2 = - torch.mean(torch.log(t_t / t2t))

        self.args.writer.add_scalar('inm_loss/loss1', loss1, self.count)
        self.args.writer.add_scalar('inm_loss/loss2', loss2, self.count)
        # self.args.writer.add_scalar('inm_loss/loss3', loss3, self.count)
        self.count += 1

        return loss1 + loss2

        i_t = torch.exp(torch.sum(images * texts, dim=-1) / gama)
        i_i = torch.exp(torch.sum(images * images, dim=-1) / gama)
        i2i = torch.matmul(images, images.permute(1, 0))
        i2i = torch.sum(torch.exp(i2i/gama), dim=-1) - i_i + i_t
        loss1 = - torch.mean(torch.log(i_t / i2i))

        t_i = torch.exp(torch.sum(texts * images, dim=-1) / gama)
        t_t = torch.exp(torch.sum(texts * texts, dim=-1) / gama)
        t2t = torch.matmul(texts, texts.permute(1, 0))
        t2t = torch.sum(torch.exp(t2t/gama), dim=-1) - t_t + t_i
        loss2 = - torch.mean(torch.log(t_i / t2t))
        return loss1 + loss2
        

        ui = torch.mean(images, dim=0, keepdim=True)
        ut = torch.mean(texts, dim=0, keepdim=True)
        loss1 = - torch.log(torch.mean((torch.matmul(ui, ut.permute(1, 0)) + 1) / 2))
        loss2 = - torch.log(torch.sqrt(self.mse(images, ui.expand_as(images))) / 2)
        loss3 = - torch.log(torch.sqrt(self.mse(texts, ut.expand_as(texts))) / 2)
        return loss1 + loss2 + loss3

    def distillation(self, it, ic, ct):
        it = it.permute(1, 0)
        icct = torch.matmul(ic, ct).permute(1, 0)
        x = F.log_softmax(it, dim=-1)
        y = F.softmax(icct, dim=-1).detach()
        loss = F.kl_div(x, y, reduction="batchmean")
        return loss

    def match_loss(self, images, texts, gama):
        img_txt_match_matrix = torch.matmul(images, texts.permute(1, 0))
        itmm = img_txt_match_matrix / gama
        img_txt_match_matrix = torch.exp(itmm)
        img_txt_total_i2t = torch.sum(img_txt_match_matrix, dim=-1)
        img_txt_total_t2i = torch.sum(img_txt_match_matrix, dim=0)
        img_txt_match_pos = torch.diag(img_txt_match_matrix)
        img_txt_match_loss = - torch.mean(torch.log(img_txt_match_pos / img_txt_total_t2i)) - torch.mean(torch.log(img_txt_match_pos / img_txt_total_i2t))
        return img_txt_match_loss, itmm
    
    def caploss(self, cap_fwd, texts_gt):
        cap_fwd_loss = self.cross_entropy(cap_fwd.view(-1, self.args.vocab_size), texts_gt.view(-1))
        return cap_fwd_loss
    
    def sdloss(self, itmm):
        itmm = itmm[:self.args.batch_size, :self.args.batch_size].permute(1, 0)
        y_t2i = F.softmax(self.pre_itmm_it, dim=-1).detach()
        x_t2i = F.log_softmax(itmm, dim=-1)
        sdloss_t2i = F.kl_div(x_t2i, y_t2i, reduction="batchmean")

        # y_i2t = F.softmax(self.pre_itmm_it.permute(1, 0), dim=-1).detach()
        # x_i2t = F.log_softmax(itmm.permute(1, 0), dim=-1)
        # sdloss_i2t = F.kl_div(x_i2t, y_i2t, reduction="batchmean")

        return self.args.alpha_sd * sdloss_t2i #+ sdloss_i2t
       
    def metric(self, img_id, images, texts, o_images=None, texts_gt=None):
        label = self.label(img_id)

        if self.args.test_1k:
            images = images[:5000, :]
            texts = texts[:5000, :]
            label = label[:5000, :]
            
        len = images.shape[0]
        images = images[::5, :]
        o_images = o_images[::5, :] if o_images is not None else None
        dimension = 0 if self.args.mode == 't2i' else -1
        label = label[:, 0].unsqueeze(-1) // 5 if self.args.mode == 't2i' else label[::5, :]
        
        it_match = torch.matmul(images, texts.permute(1, 0))
        it_match = torch.exp(it_match / self.args.gama_it)
        it_total = torch.sum(it_match, dim=dimension)
        it_match = it_match / it_total
        it_rk_matrix = torch.topk(it_match, k=self.args.topk, dim=dimension, largest=True)
        it_r1, it_r5, it_r10 = self.score(it_rk_matrix[1], label, len)
        it_rsum = it_r1 + it_r5 + it_r10

        # rank_id = self.rank_k(it_rk_matrix, o_images, texts_gt)
        # rk_r1, rk_r5, rk_r10 = self.score(rank_id, label, len)
        # rk_rsum = rk_r1 + rk_r5 + rk_r10
        
        return {"R@1":it_r1, "R@5":it_r5, "R@10":it_r10, "Rsum":it_rsum}
                # {"R@1":rk_r1, "R@5":rk_r5, "R@10":rk_r10, "Rsum":rk_rsum}
        
    def rank_k(self, it_rk_matrix, o_images, texts_gt):
        seq_len = texts_gt.shape[1]
        it_picked_id = it_rk_matrix[1] if self.args.mode == 'i2t' else it_rk_matrix[1].permute(1, 0)
        it_picked_matrix = it_rk_matrix[0] if self.args.mode == 'i2t' else it_rk_matrix[0].permute(1, 0)

        
        ct_picked_matrix = torch.zeros_like(it_picked_matrix)
        with tqdm(desc='Slow', unit='it', total=it_picked_id.shape[0]) as pbar:
            for i in range(it_picked_id.shape[0]):
                images = torch.cat([o_images[id].unsqueeze(0) for id in it_picked_id[i]], dim=0)
                texts = texts_gt[i].unsqueeze(0).expand(it_picked_id[i].shape[0] , *texts_gt[i].shape[:])
                cap_fwd = self.model.generate(images, texts, self.args.bos_id, self.args.eos_id, seq_len, True)
                for j, id in enumerate(it_picked_id[i]):
                    ct_picked_matrix[i, j] = self.cross_entropy(cap_fwd[j, :-1].contiguous(), texts_gt[i, 1:].contiguous())
                pbar.update()
        picked_matrix = torch.topk(ct_picked_matrix, k=self.args.topk, dim=-1, largest=True)
        picked_id = torch.zeros_like(it_picked_id)
        for i in range(picked_matrix[1].shape[0]):
            for j, id in enumerate(picked_matrix[1][i]):
                picked_id[i, j] = it_picked_id[i, id]
        return picked_id if self.args.mode == 'i2t' else picked_id.permute(1, 0)
    
    def score(self, rk_id, label, len):
        if self.args.mode == 't2i':
            rk_id = rk_id.permute(1, 0)
        num = 0.0
        for i in range(label.shape[1]):
            num += ((rk_id - label[:, i].unsqueeze(-1)) == 0).float()
        num_1 = (torch.sum(num[:, :1], dim=-1, keepdim=True) > 0).float().sum()
        num_5 = (torch.sum(num[:, :5], dim=-1, keepdim=True) > 0).float().sum()
        num_10 = (torch.sum(num[:, :10], dim=-1, keepdim=True) > 0).float().sum()
        return num_1.item() / len, num_5.item() / len, num_10.item() / len
        
    def label(self, img_id):
        bs = img_id.shape[0]
        img_id = img_id.numpy()
        count_id = {}
        label = {}
        for i in range(bs):
            if img_id[i] not in count_id.keys():
                count_id[img_id[i]] = 0
            else:
                count_id[img_id[i]] += 1
            if '{}_{}'.format(img_id[i], count_id[img_id[i]]) not in label.keys():
                label['{}_{}'.format(img_id[i], count_id[img_id[i]])] = []

        for i in range(bs):
            for j in range(count_id[img_id[i]]+1):
                label['{}_{}'.format(img_id[i], j)].append(i)
        label = list(label.values())
        label = np.array(label)
        return torch.from_numpy(label).to(self.device)