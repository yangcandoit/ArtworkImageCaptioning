import os
import time
import random
import argparse
import datetime
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from tqdm import tqdm
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.utils import accuracy, AverageMeter
from shutil import copyfile
from config import get_config
# from models import build_model
# from data import build_loader
# from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from train import train
from test import test
from data import load_dataset
# from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from timm.data import Mixup
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn import NLLLoss

from model import *
from data import *
from utils import *
# from model.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
# from torch.utils.data import DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
import multiprocessing
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--exp_name', type=str, default='aic')
    # easy config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, help='path to dataset')
    
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp_opt_level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def evaluate_loss(model, dataloader, loss_fn, text_field,epoch):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (image, captions) in enumerate(dataloader):
                image, captions = image.to(device), captions.to(device)
                detections=model.encoder(image)
                out = model.decoder(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                # print('1111')
                # print('2222',out.view(-1, len(text_field.vocab)+1).shape)
                # print(captions.view(-1).shape)
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field,epoch):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    # print(model.decoder.mask_enc)
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                images=model.encoder(images)

                out, _ = model.decoder.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            print(caps_gt)
            print(caps_gen)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field,criterion, lr_scheduler, epoch):
    # Training with cross-entropy

    # print(model.train())
    # scheduler.step()

    running_loss = .0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions) in enumerate(dataloader):
            images, captions = images.to(device), captions.to(device)
            # print(detections.shape)
            # detections=model.encoder(images)
            # print(detections)
            out = model(images, captions)

            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            # print(len(text_field.vocab))
            # print(out.shape)
            # out.view(1,-1)
            # out.view(-1, len(text_field.vocab))
            # print(out.view(-1, len(text_field.vocab)+1).shape)
            # print(captions_gt.shape)
            # print(captions_gt.view(-1).shape)
            loss = criterion(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            lr_scheduler.step(epoch)

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field,epoch):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            detections=model.encoder(detections)
            outs, log_probs = model.decoder.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
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


def main(args,config):
    # dataset_train, dataset_val, data_loader_train, data_loader_val,mlb = load_dataset(config)


    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")    
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    
        # Pipeline for image regions
    image_field = ImageDetectionsField()
    
        # Create the dataset
    dataset = ICON(image_field, text_field, config.DATA.DATA_PATH,config)
    
    
    
    
    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        # text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name , 'rb'))
    
    # print(len(text_field.vocab.itos))
    encoder= SwinTransformer(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        window_size=config.MODEL.SWIN.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    
    mesh_encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 10})
    mesh_decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    decoder = Transformer(text_field.vocab.stoi['<bos>'], mesh_encoder, mesh_decoder).to(device)
    
    # decoder= Transformer(text_field.vocab.stoi['<bos>'],MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                    #  attention_module_kwargs={'m': 40}),MeshedDecoder(len(text_field.vocab)+1, 54, 3, text_field.vocab.stoi['<pad>']))
    
    # def a(x):
    #     for i in x.children():
    #         print('--------',hasattr(i,'enable_statefulness'))
    #         b(i)
    # def b(x):
    #     for i in x.children():
    #         print('sub',hasattr(i,'enable_statefulness'))

    # a(decoder.decoder)
    model=AIC(encoder,decoder)
    
    model.cuda()
    # print(model.decoder)
    
    # dataset_train, dataset_val, data_loader_train, data_loader_val,mlb = load_dataset(config)
    
    train_dataset, val_dataset, test_dataset = dataset.splits
    # print(encoder)
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    
    optimizer = build_optimizer(config, model)
    # if config.AMP_OPT_LEVEL != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    
     
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=config.TRAIN.EPOCHS,
        t_mul=1.,
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=config.TRAIN.WARMUP_EPOCHS,
        cycle_limit=1,
        t_in_epochs=False,
    )
    
    criterion = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    max_accuracy = 0.0
    
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.MODEL.RESUME = "./saved_models/aic_best.pth"
            
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    
    if config.MODEL.RESUME:
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")

        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        pretrained_dict=torch.load(config.MODEL.RESUME, map_location='cpu')
        model_dict=model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        msg = model.encoder.load_state_dict(model_dict, strict=False)
        logger.info(msg)
        max_accuracy = 0.0

        if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.defrost()
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
            config.freeze()
            if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
                amp.load_state_dict(checkpoint['amp'])
            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            if 'max_accuracy' in checkpoint:
                max_accuracy = checkpoint['max_accuracy']

        del checkpoint
        torch.cuda.empty_cache()
    
    
    
    
    # if args.resume_last or args.resume_best:
    #     if args.resume_last:
    #         fname = 'saved_models/%s_last.pth' % args.exp_name
    #     else:
    #         fname = 'saved_models/%s_best.pth' % args.exp_name

    #     if os.path.exists(fname):
    #         data = torch.load(fname)
    #         torch.set_rng_state(data['torch_rng_state'])
    #         torch.cuda.set_rng_state(data['cuda_rng_state'])
    #         np.random.set_state(data['numpy_rng_state'])
    #         random.setstate(data['random_rng_state'])
    #         model.load_state_dict(data['state_dict'], strict=False)
    #         optimizer.load_state_dict(data['optimizer'])
    #         scheduler.load_state_dict(data['scheduler'])
    #         start_epoch = data['epoch'] + 1
    #         best_cider = data['best_cider']
    #         patience = data['patience']
    #         use_rl = data['use_rl']
    #         print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
    #             data['epoch'], data['val_loss'], data['best_cider']))
    
    
    logger.info("Start training")
    start_time = time.time()
    
    # for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    #     data_loader_train.sampler.set_epoch(epoch)

    #     train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch,  lr_scheduler,mlb)
    #     if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
    #         save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)

    #     print(data_loader_val)
    #     acc1, loss = validate(config, data_loader_val, model)
    #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
    #     max_accuracy = max(max_accuracy, acc1)
    #     logger.info(f'Max accuracy: {max_accuracy:.2f}%')


    use_rl=False
    best_cider = .0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            
        dataloader_train = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE,  num_workers=config.DATA.NUM_WORKERS,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=config.DATA.NUM_WORKERS)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=config.DATA.BATCH_SIZE , 
                                           num_workers=config.DATA.NUM_WORKERS)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=config.DATA.BATCH_SIZE )
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=config.DATA.BATCH_SIZE )
          
            
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optimizer, text_field,criterion, lr_scheduler, epoch)
            logger.info(f"data/train_loss: {train_loss / 1e9} {epoch}")
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optimizer, cider_train, text_field,epoch)
            # logger.info('data/train_loss %d -- %d', train_loss, epoch)
            logger.info(f"data/train_loss: {train_loss / 1e9} {epoch}")
            logger.info(f"data/reward: {reward / 1e9} {epoch}")
            # logger.info('data/reward', reward, epoch)
            logger.info(f'data/reward_baseline{ reward_baseline} {epoch}')

    #     # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, criterion, text_field,epoch)
        logger.info(f'data/val_loss{val_loss}{epoch}' )

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field,epoch)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        logger.info(f'data/val_cider {val_cider}{epoch}')
        logger.info(f"data/val_bleu1 {scores['BLEU'][0]}{epoch}")
        logger.info(f"data/val_bleu4 {scores['BLEU'][3]}{epoch}")
        logger.info(f"data/val_meteor {scores['METEOR']}, {epoch}")
        logger.info(f"data/val_rouge {scores['ROUGE']} {epoch}")

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field,epoch)
        print("Test scores", scores)
        logger.info(f'data/test_cider {val_cider}{epoch}')
        logger.info(f"data/test_bleu1 {scores['BLEU'][0]}{epoch}")
        logger.info(f"data/test_bleu4 {scores['BLEU'][3]}{epoch}")
        logger.info(f"data/test_meteor {scores['METEOR']}, {epoch}")
        logger.info(f"data/test_rouge {scores['ROUGE']} {epoch}")

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1
        print(patience)
        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                # optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)

        if best:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)

        if exit_train:
            
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    args,config=parse_option()
    device = torch.device('cuda')
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    # print(torch.distributed.is_mpi_available())
    torch.cuda.set_device(config.LOCAL_RANK)
    
    # torch.distributed.init_process_group(backend='gloo', init_method='env://127.0.0.1:12345', world_size=world_size, rank=rank)
    # torch.distributed.barrier()
    
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE  / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE  / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE  / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,  name=f"{config.MODEL.NAME}")

    # if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    
    main(args,config)
    
    
    
    
    