import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
from unittest.mock import DEFAULT
from data import ImageField
from data import COCO, DataLoader, Flickr
from data import TextField
from models import clip
import numpy as np
import torch
# torch.backends.cudnn.enabled = False

from torch.utils.tensorboard import SummaryWriter
import argparse, os
from shutil import copyfile

from models import FaseAndSlow, Trainer, ShareFAS
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast And Slow Transformer')
    parser.add_argument('--exp_name', type=str, default='coco5k_RN50x16_BERT_it_cap_ic_ct_sd_loss_gama0.07')  #
    parser.add_argument('--batch_size', type=int, default=200) 
    parser.add_argument('--learn_rate', type=float, default=4e-5) 
    parser.add_argument('--workers', type=int, default=16) 
    parser.add_argument('--h', type=int, default=12) 
    parser.add_argument('--d_model', type=int, default=768) 
    parser.add_argument('--N', type=list, default=[3, 3, 3])
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=1500)
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--gama_it', type=float, default=0.07)
    parser.add_argument('--gama_ic', type=float, default=0.07)
    parser.add_argument('--gama_ct', type=float, default=0.07)
    parser.add_argument('--alpha_sd', type=float, default=20.0)

    parser.add_argument('--rerank', action='store_true', default=False)
    
    parser.add_argument('--use_imgpt', action='store_true', default=True)
    parser.add_argument('--use_txtpt', action='store_true', default=True)

    parser.add_argument('--with_cap', action='store_true', default=True)
    parser.add_argument('--with_ct', action='store_true', default=True)
    parser.add_argument('--with_ic', action='store_true', default=True)
    
    parser.add_argument('--resume_last', action='store_true', default=False)
    parser.add_argument('--resume_best', action='store_true', default=False)
    # parser.add_argument('--checkpoint', type=str, default='saved_models/RN101-BERT-finetune_ic_ct_cap_inm_loss_new1/last.pth')
    
    parser.add_argument('--mode', type=str, choices=['t2i', 'i2t'], default='t2i')
    parser.add_argument('--test_1k', action='store_true', default=False)
    
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'flickr'])
    parser.add_argument('--images_folder', type=str, default='/raid/ggw/grid-feats-vqa/datasets/coco')
    parser.add_argument('--annotation_folder', type=str, default="/raid/ggw/datasets/image_caption/annotations")
    parser.add_argument('--fimage_folder', type=str, default="/raid/ggw/datasets/flickr30k/flickr30k-images")
    parser.add_argument('--annotation_path', type=str, default="/raid/ggw/datasets/flickr30k/results_20130124.token")

    parser.add_argument('--tensorboard_logs_folder', type=str, default='saved_logs/tensorboard_logs')
    parser.add_argument('--args_logs_folder', type=str, default='saved_logs/args_logs')
    parser.add_argument('--clip_variant', type=str, choices=["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16", "RN50x64"], default="RN50x4")
    args = parser.parse_args()
    print(args)
    
    # writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_logs_folder, args.exp_name))
    writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_logs_folder, "test"))
    args.writer = writer
    # save args
    argsDict = args.__dict__
    savepath = os.path.join(args.args_logs_folder, args.exp_name) + ".txt"
    with open(savepath, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    
    if args.use_imgpt:
        clip_model, transform = clip.load(args.clip_variant, jit=False)
        args.image_dim = clip_model.visual.embed_dim
 
    image_field = ImageField(transform=transform)
    text_field = TextField()
    args.text_field = text_field

    if args.dataset == 'coco':
        dataset = COCO(image_field, text_field, args.images_folder, args.annotation_folder, args.annotation_folder)
    elif args.dataset == 'flickr':
        dataset = Flickr(args.fimage_folder, args.annotation_path, image_field, text_field)
        from torch.utils.data import DataLoader
    train_dataset, val_dataset, test_dataset = dataset.splits
    
    args.vocab_size = text_field._tokenizer.vocab_size
    args.bos_id = text_field._tokenizer.cls_token_id
    args.eos_id = text_field._tokenizer.sep_token_id
    args.padding_id = text_field._tokenizer.pad_token_id
    
    model = FaseAndSlow(args, d_model=args.d_model, h=args.h)
    trainer = Trainer(model, device=device, args=args)
    
    start_epoch = 0
    args.patience = 0
    best_val_score = 0.
    exit = False
    
    if args.resume_last or args.resume_best:
        fname = 'saved_models/%s/last.pth' % args.exp_name if args.resume_last else 'saved_models/%s/best.pth' % args.exp_name
        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            trainer.model.load_state_dict(data['state_dict'], strict=False)
            start_epoch = data['epoch'] + 1
            args.patience = data['patience']
            trainer.optim.load_state_dict(data['optimizer'])
            trainer.scheduler.load_state_dict(data['scheduler'])
            print('Resuming from epoch %d' % data['epoch'])
    
    # if args.checkpoint != None:
    #     fname = args.checkpoint
    #     if os.path.exists(fname):
    #         data = torch.load(fname)
    #         torch.set_rng_state(data['torch_rng_state'])
    #         torch.cuda.set_rng_state(data['cuda_rng_state'])
    #         np.random.set_state(data['numpy_rng_state'])
    #         random.setstate(data['random_rng_state'])
    #         trainer.model.load_state_dict(data['state_dict'], strict=False)
    #         start_epoch = data['epoch'] + 1
    #         patience = data['patience']
    #         trainer.optim.load_state_dict(data['optimizer'])
    #         print('Checkpoint: Resuming from epoch %d' % data['epoch'])

    if args.use_imgpt:
        trainer.load_pt(clip_model)
    
    for epoch in range(start_epoch, start_epoch + 50):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
        # train_loss = trainer.train(dataloader_train, epoch)
        # trainer.scheduler.step()
        # writer.add_scalar('train/train_loss', train_loss, epoch)
        
        val_it_scores = trainer.val(dataloader_val, epoch)
        writer.add_scalar('val/val_it_rsum', val_it_scores['Rsum'], epoch)
        writer.add_scalar('val/val_it_R@1', val_it_scores['R@1'], epoch)
        writer.add_scalar('val/val_it_R@5', val_it_scores['R@5'], epoch)
        writer.add_scalar('val/val_it_R@10', val_it_scores['R@10'], epoch)
        # writer.add_scalar('val/val_tc_rsum', val_tc_scores['Rsum'], epoch)
        # writer.add_scalar('val/val_tc_R@1', val_tc_scores['R@1'], epoch)
        # writer.add_scalar('val/val_tc_R@5', val_tc_scores['R@5'], epoch)
        # writer.add_scalar('val/val_tc_R@10', val_tc_scores['R@10'], epoch)

        test_it_scores = trainer.test(dataloader_test, epoch)
        writer.add_scalar('test/test_it_rsum', test_it_scores['Rsum'], epoch)
        writer.add_scalar('test/test_it_R@1', test_it_scores['R@1'], epoch)
        writer.add_scalar('test/test_it_R@5', test_it_scores['R@5'], epoch)
        writer.add_scalar('test/test_it_R@10', test_it_scores['R@10'], epoch)
        # writer.add_scalar('test/test_tc_rsum', test_tc_scores['Rsum'], epoch)
        # writer.add_scalar('test/test_tc_R@1', test_tc_scores['R@1'], epoch)
        # writer.add_scalar('test/test_tc_R@5', test_tc_scores['R@5'], epoch)
        # writer.add_scalar('test/test_tc_R@10', test_tc_scores['R@10'], epoch)
        
        best = False
        if val_it_scores['Rsum'] >= best_val_score:
            best_val_score = val_it_scores['Rsum']
            args.patience = 0
            best = True
        else:
            args.patience += 1

        if args.patience >= 5:
            exit = True

        if not os.path.exists('saved_models/%s' % args.exp_name):
            os.makedirs('saved_models/%s' % args.exp_name)  
        if os.path.exists('saved_models/%s/last.pth' % args.exp_name):
            os.remove('saved_models/%s/last.pth' % args.exp_name)
            
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': epoch,
            'state_dict': trainer.model.state_dict(),
            'optimizer': trainer.optim.state_dict(),
            'patience': args.patience,
            'scheduler': trainer.scheduler.state_dict(),
        }, 'saved_models/%s/last.pth' % args.exp_name)
        
        if best:
            copyfile('saved_models/%s/last.pth' % args.exp_name, 'saved_models/%s/best.pth' % args.exp_name)

        if exit:
            writer.close()
            break

        
    