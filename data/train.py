import os
from tkinter import image_names
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
from unittest.mock import DEFAULT
from data import ImageField, RawField, Merge
from data import COCO, DataLoader
from data import TextField, TextField_old
from weight_loss import weight_loss
from models import RectifyCaption
from models import clip
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from torch.autograd import gradcheck
from utils.utils import fine_grain_AAP

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)



def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images, captions) in enumerate(dataloader):
                images, captions = images.to(device), captions.to(device)
                images = image_model(images)
                images = fine_grain_AAP(images)
                _, rectify_out = model(images, captions)
                captions = captions[:, 1:].contiguous()
                rectify_out = rectify_out[:, :-1].contiguous()
                loss_rectify = loss_fn(rectify_out.view(-1, text_field._tokenizer.vocab_size), captions.view(-1))
                # loss_rectify = loss_fn(rectify_out.view(-1, len(text_field.vocab)), captions.view(-1))
                loss = loss_rectify 
                this_loss = loss.item()
                running_loss += this_loss
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, ((image_paths, images), caps_gt) in enumerate(dataloader):
            images = images.to(device)
            with torch.no_grad():
                images = image_model(images)
                images = fine_grain_AAP(images)
                out, _ = model.train_scst(images, 20, text_field._tokenizer.eos_idx, 5, out_size=1)
                caps_gen = text_field.decode(out)
                for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                    # gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gen['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    image_model.eval()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train(patience=%d)' % (e, patience), unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions) in enumerate(dataloader):
            images, captions = images.to(device), captions.to(device)
            
            with torch.no_grad():
                images = image_model(images)
                images = fine_grain_AAP(images)
            
            dec_out, rectify_out = model(images, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            dec_out = dec_out[:, :-1].contiguous()
            loss_dec = loss_fn(dec_out.view(-1, text_field._tokenizer.vocab_size), captions_gt.view(-1))
            # loss_dec = loss_fn(dec_out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            if not args.without_Rectifier:
                rectify_out = rectify_out[:, :-1].contiguous()
                loss_rectify = loss_fn(rectify_out.view(-1, text_field._tokenizer.vocab_size), captions_gt.view(-1))
                # loss_rectify = loss_fn(rectify_out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                loss = loss_rectify + args.lambda_ * loss_dec
            else:
                loss = loss_dec
            this_loss = loss.item()
            running_loss += this_loss
            loss.backward()
            optim.step()

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    return loss

def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0 
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train(patience=%d)' % (e, patience), unit='it', total=len(dataloader)) as pbar:
        for it, ((image_paths, images), caps_gt_o) in enumerate(dataloader):
            images = images.to(device)
            
            with torch.no_grad():
                images = image_model(images)
                images = fine_grain_AAP(images)
            
            outs, log_probs = model.train_scst(images, seq_len, text_field._tokenizer.eos_idx, beam_size, out_size=beam_size)
            # outs, log_probs = model.train_scst(images, seq_len, text_field.vocab.stoi['<eos>'], beam_size, out_size=beam_size)
            optim.zero_grad()
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt_o)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(images[-1].shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)
            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--example_name', type=str, default='CLip_RN50x16_with_FVE_with_Rectifier_0.5')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--scst_batch_size', type=int, default=20)    
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--lambda_', type=float, default=0.5)
    parser.add_argument('--clip_variant', type=str, choices=["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"], default="RN50x16")
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--without_FVE', action="store_true")
    parser.add_argument('--without_Rectifier', action="store_true")
    # parser.add_argument('--checkpoint', type=str, default='/raid/ggw/double-decoder-transformer-clip/saved_models/CLip_RN50x16_with_FVE_with_Rectifier/epoch_52.pth')
    parser.add_argument('--images_folder', type=str, default='/raid/ggw/grid-feats-vqa/datasets/coco')
    parser.add_argument('--features_path', type=str, default="/raid/ggw/datasets/image_caption/coco_grid_77_features.hdf5")
    parser.add_argument('--annotation_folder', type=str, default="/raid/ggw/datasets/image_caption/annotations")
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()
    print(args)

    # print('Meshed-Memory Transformer Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.example_name))

    # Pipeline for image regions
    clip_model, transform = clip.load(args.clip_variant, jit=False)
    image_model = clip_model.visual
    image_model.forward = image_model.intermediate_features
    image_field = ImageField(transform=transform)
    args.image_dim = image_model.embed_dim

    # Pipeline for text
    text_field = TextField()

    # Create the dataset
    dataset = COCO(image_field, text_field, args.images_folder, args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, args.without_FVE, args.image_dim, d_model=args.d_model, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(text_field._tokenizer.vocab_size, 100, 3, d_model=args.d_model)
    rectify = RectifyCaption(3, text_field._tokenizer.vocab_size, 100, d_model=args.d_model) if not args.without_Rectifier else None
    model = Transformer(text_field._tokenizer.bos_idx, encoder, decoder, rectify).to(device)
    
    dict_dataset_train = train_dataset.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField()})
    ref_caps_train = list(i.text for i in train_dataset.examples)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField()})


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=0)
    use_rl = False
    patience = 0
    start_epoch = 0
    best_cider = .0
    best_test_cider = .0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s/last.pth' % args.example_name
        else:
            fname = 'saved_models/%s/best.pth' % args.example_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    # if args.checkpoint is not None:
    #     fname = args.checkpoint
    #     if os.path.exists(fname):
    #         data = torch.load(fname)
    #         torch.set_rng_state(data['torch_rng_state'])
    #         torch.cuda.set_rng_state(data['cuda_rng_state'])
    #         np.random.set_state(data['numpy_rng_state'])
    #         random.setstate(data['random_rng_state'])
    #         model.load_state_dict(data['state_dict'], strict=False)
    #         optim.load_state_dict(data['optimizer'])
    #         scheduler.load_state_dict(data['scheduler'])
    #         start_epoch = data['epoch'] + 1
    #         best_cider = data['best_cider']
    #         patience = data['patience']
    #         use_rl = data['use_rl']
    #         print('Resuming from epoch %d, validation loss %f, best cider %f' % (
    #             data['epoch'], data['val_loss'], data['best_cider']))

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)      
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.scst_batch_size, shuffle=True,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.scst_batch_size, num_workers=args.workers)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.scst_batch_size, num_workers=args.workers)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)
        
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)
        
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/val_spice', scores['SPICE'], e)
        
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', test_cider, e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/test_spice', scores['SPICE'], e)
        
        # Prepare for next epoch
        best = False
        if test_cider >= best_cider:
                best_cider = test_cider
                patience = 0
                best = True
        else:
            patience += 1
        
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True
            
        switch_to_rl = False
        exit_train = False
        if patience == 5:  
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True
        
        if switch_to_rl and not best:
            data = torch.load('saved_models/%s/best.pth' % args.example_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))
        
        if not os.path.exists('saved_models/%s' % args.example_name):
            os.makedirs('saved_models/%s' % args.example_name)
        
        if os.path.exists('saved_models/%s/last.pth' % args.example_name):
            os.remove('saved_models/%s/last.pth' % args.example_name)
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s/last.pth' % args.example_name)

        if best and not use_rl:
            copyfile('saved_models/%s/last.pth' % args.example_name, 'saved_models/%s/best.pth' % args.example_name)
        if use_rl:    
            copyfile('saved_models/%s/last.pth' % args.example_name, 'saved_models/%s/epoch_%d.pth' % (args.example_name, e))
        
        
        if exit_train:
            writer.close()