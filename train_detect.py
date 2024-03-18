"""
Training script for a detector.
To run, example:
$ python train_detect.py config/train_yolov2_voc.py --n_worker=1
"""


import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.ops import box_convert
from torchvision.datasets import wrap_dataset_for_transforms_v2
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

from evaluator import DetEvaluator


# -----------------------------------------------------------------------------
# Default config values
# Task related
task_name = 'detect'
eval_only = False  # if True, script exits right after the first eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'backbone' or 'pretrained'
from_ckpt = ''  # only used when init_from='resume' or 'backbone' or 'pretrained'
# Data related
dataset_name = 'voc'  # 'voc' or 'nano_voc' or 'blank_voc' or 'nano_blank_voc'
img_h = 416
img_w = 416
n_class = 20
# Transform related
multiscale_min_sizes = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)  # square edge
min_wh = 1e-3  # normalized by the grid cell width,height
perspective = 0.015
crop_scale = 0.8
ratio_min = 0.5
ratio_max = 2.0
degrees = 0.5  # unit: deg
translate = 0.1
scale = 0.25
shear = 0.5  # unit: deg
brightness = 0.4
contrast = 0.4
saturation = 0.7
hue = 0.015
flip_p = 0.5
letterbox = True
imgs_mean = (0.485, 0.456, 0.406)
imgs_std = (0.229, 0.224, 0.225)
fill = (123.0, 117.0, 104.0)
# Model related
model_name = 'yolov2'
n_box_per_cell = 5  # B in the paper
anchors = ((1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52))  # w,h relative to a cell
# Loss related
lambda_noobj = 0.5
lambda_obj = 1.0
lambda_class = 1.0
lambda_coord = 5.0
# Train related
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_iters = 100000  # total number of training iterations
# Optimizer related
optimizer_type = 'adamw'  # 'adamw' or 'adam' or 'sgd'
learning_rate = 1e-3  # max learning rate
beta1 = 0.9  # beta1 for adamw, momentum for sgd
beta2 = 0.999
weight_decay = 1e-2
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'  # whether to decay the learning rate, which type of lr scheduler. False, 'cosine', 'step'
warmup_iters = 5000  # how many steps to warm up for
lr_decay_iters = 100000  # should be ~= max_iters; this is step_size if decay_lr='step'
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10; this is gamma if decay_lr='step'
use_fused = True  # whether to use fused optimizer kernel
# Eval related
eval_interval = 100  # keep frequent if we'll overfit
eval_iters = 200  # use more iterations to get good estimate
prob_thresh = 0.001  # threshold for predicted class-specific confidence score (= obj_prob * class_prob)
nms_iou_thresh = 0.5  # for NMS
use_torchmetrics = False  # whether to use cocoeval for detailed but slower metric computation
# Log related
timestamp = '00000000-000000'
out_dir = f'out/yolov2_voc/{timestamp}'
wandb_log = False  # disabled by default
wandb_project = 'voc'
wandb_run_name = f'yolov2_{timestamp}'
log_interval = 50  # don't print too often
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
# System related
device = 'cuda'  # example: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
n_worker = 0
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, tuple))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# Various inits, derived attributes, I/O setup
imgs_per_iter = gradient_accumulation_steps * batch_size
print(f"imgs_per_iter will be: {imgs_per_iter}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# Dataloader
data_dir = os.path.join('data', dataset_name.strip('nano_').strip('blank_'))
if dataset_name == 'voc' or dataset_name == 'nano_voc':
    from dataloaders.voc import VocConfig, VocCollateFn, VocTrainDataLoader, VocValDataLoader
    dataloader_args = dict(
        img_h=img_h,
        img_w=img_w,
        multiscale_min_sizes=multiscale_min_sizes,
        n_box_per_cell=n_box_per_cell,
        min_wh=min_wh,
        perspective=perspective,
        crop_scale=crop_scale,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        flip_p=flip_p,
        letterbox=letterbox,
        imgs_mean=imgs_mean,
        imgs_std=imgs_std,
        fill=fill,
    )
    dataloader_config = VocConfig(**dataloader_args)
    collate_fn_train = VocCollateFn(multiscale=True, config=dataloader_config)
    collate_fn_val = VocCollateFn(multiscale=False, config=dataloader_config)
    dataloader_train = VocTrainDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                          collate_fn=collate_fn_train, nano=dataset_name.startswith('nano_'))
    dataloader_val = VocValDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                      collate_fn=collate_fn_val, shuffle=True,  # shuffle=True since only eval partial data
                                      nano=dataset_name.startswith('nano_'))
elif dataset_name == 'blank_voc' or dataset_name == 'nano_blank_voc':
    from dataloaders.voc import BlankVocConfig, VocCollateFn, BlankVocTrainDataLoader, BlankVocValDataLoader
    dataloader_args = dict(
        img_h=img_h,
        img_w=img_w,
        multiscale_min_sizes=multiscale_min_sizes,
        n_box_per_cell=n_box_per_cell,
        min_wh=min_wh,
        letterbox=letterbox,
        imgs_mean=imgs_mean,
        imgs_std=imgs_std,
        fill=fill,
    )
    dataloader_config = BlankVocConfig(**dataloader_args)
    collate_fn_train = VocCollateFn(multiscale=True, config=dataloader_config)
    collate_fn_val = VocCollateFn(multiscale=False, config=dataloader_config)
    dataloader_train = BlankVocTrainDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                               collate_fn=collate_fn_train, nano=dataset_name.startswith('nano_'))
    dataloader_val = BlankVocValDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                           collate_fn=collate_fn_val, shuffle=True,  # shuffle=True since only eval partial data
                                           nano=dataset_name.startswith('nano_'))
else:
    raise ValueError(f"dataset_name: {dataset_name} not supported")
print(f"train dataset: {len(dataloader_train.dataset)} samples, {len(dataloader_train)} batches")
print(f"val dataset: {len(dataloader_val.dataset)} samples, {len(dataloader_val)} batches")

class BatchGetter:  # for looping through dataloaders is still a bit faster and less gpu memory than this
    assert len(dataloader_train) >= eval_iters, f"Not enough batches in train loader for eval."
    assert len(dataloader_val) >= eval_iters, f"Not enough batches in val loader for eval."
    dataiter = {'train': iter(dataloader_train), 'val': iter(dataloader_val)}

    @classmethod
    def get_batch(cls, split):
        try:
            X, Y, Y_supp = next(cls.dataiter[split])
        except StopIteration:
            cls.dataiter[split] = iter(dataloader_train) if split == 'train' else iter(dataloader_val)
            X, Y, Y_supp = next(cls.dataiter[split])

        if device_type == 'cuda':
            # X, Y is pinned in dataloader, which allows us to move them to GPU asynchronously (non_blocking=True)
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)

        return X, Y, Y_supp


# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


# Model init
if init_from == 'scratch':
    # Init a new model from scratch
    print(f"Initializing a new {model_name} model from scratch")
elif init_from == 'resume':
    print(f"Resuming training {model_name} from {from_ckpt}")
    # Resume training from a checkpoint
    checkpoint = torch.load(from_ckpt, map_location='cpu')  # load to CPU first to avoid GPU OOM
    torch.set_rng_state(checkpoint['rng_state'].to('cpu'))
    checkpoint_model_args = checkpoint['model_args']
    assert model_name == checkpoint['config']['model_name'], "model_name mismatch"
    assert dataset_name == checkpoint['config']['dataset_name'], "dataset_name mismatch"
elif init_from == 'backbone':
    print(f"Initializing a {model_name} model with pretrained backbone weights: {from_ckpt}")
    # Init a new model with pretrained backbone weights
    checkpoint = torch.load(from_ckpt, map_location='cpu')
elif init_from == 'pretrained':
    print(f"Initializing a {model_name} model with entire pretrained weights: {from_ckpt}")
    # Init a new model with entire pretrained weights
    checkpoint = torch.load(from_ckpt, map_location='cpu')
else:
    raise ValueError(f"Invalid init_from: {init_from}")

if model_name == 'yolov2':
    from models.yolov2 import Yolov2Config, Yolov2
    model_args = dict(
        img_h=img_h,
        img_w=img_w,
        n_class=n_class,
        multiscale_min_sizes=multiscale_min_sizes,
        anchors=anchors,
        n_box_per_cell=n_box_per_cell,
        lambda_noobj=lambda_noobj,
        lambda_obj=lambda_obj,
        lambda_class=lambda_class,
        lambda_coord=lambda_coord,
        prob_thresh=prob_thresh,
        nms_iou_thresh=nms_iou_thresh,
    )  # start with model_args from command line
    if init_from == 'resume':
        # Force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['img_h', 'img_w', 'n_class', 'n_box_per_cell']:
            model_args[k] = checkpoint_model_args[k]
    # Create the model
    model_config = Yolov2Config(**model_args)
    model = Yolov2(model_config)
else:
    raise ValueError(f"model_name: {model_name} not supported")

if init_from == 'resume':
    state_dict = checkpoint['model']
    # Fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from == 'backbone':
    state_dict = checkpoint['model']
    wanted_prefix = 'backbone.'
    for k,v in list(state_dict.items()):
        if not k.startswith(wanted_prefix):
            state_dict.pop(k)
        else:
            state_dict[k[len(wanted_prefix):]] = state_dict.pop(k)
    model.backbone.load_state_dict(state_dict)
elif init_from == 'pretrained':
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
elif init_from == 'scratch':
    pass
else:
    raise ValueError(f"Invalid init_from: {init_from}")

model = model.to(device)


# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


# Optimizer
optimizer = model.configure_optimizers(optimizer_type, learning_rate, (beta1, beta2), weight_decay, device_type, use_fused)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory


# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


# Helps estimate an arbitrarily accurate loss over either split using many batches
# Not accurate since losses were averaged at each iteration (exact would have been a sum),
# then averaged altogether again at the end, but the metrics are accurate.
@torch.inference_mode()
def estimate_loss():
    out_losses, out_map50 = {}, {}
    out_losses_noobj, out_losses_obj, out_losses_class, out_losses_xy, out_losses_wh = {}, {}, {}, {}, {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(int(eval_iters * gradient_accumulation_steps))
        losses_noobj = torch.zeros(int(eval_iters * gradient_accumulation_steps))
        losses_obj = torch.zeros(int(eval_iters * gradient_accumulation_steps))
        losses_class = torch.zeros(int(eval_iters * gradient_accumulation_steps))
        losses_xy = torch.zeros(int(eval_iters * gradient_accumulation_steps))
        losses_wh = torch.zeros(int(eval_iters * gradient_accumulation_steps))
        if use_torchmetrics:
            metric = MeanAveragePrecision(iou_type='bbox')
            metric.warn_on_many_detections = False
        else:
            metric = DetEvaluator()
        for k in range(int(eval_iters * gradient_accumulation_steps)):
            X, Y, Y_supp = BatchGetter.get_batch(split)
            with ctx:
                logits, loss, loss_noobj, loss_obj, loss_class, loss_xy, loss_wh = model(X, Y)
            losses[k] = loss.item()
            losses_noobj[k] = loss_noobj.item()
            losses_obj[k] = loss_obj.item()
            losses_class[k] = loss_class.item()
            losses_xy[k] = loss_xy.item()
            losses_wh[k] = loss_wh.item()
            preds_for_eval, targets_for_eval = model.postprocess_for_eval(logits, Y_supp)
            metric.update(preds_for_eval, targets_for_eval)
        map50 = metric.compute()['map_50'] * 100
        map50 = map50.item() if isinstance(map50, torch.Tensor) else map50
        out_losses[split] = losses.mean()
        out_losses_noobj[split] = losses_noobj.mean()
        out_losses_obj[split] = losses_obj.mean()
        out_losses_class[split] = losses_class.mean()
        out_losses_xy[split] = losses_xy.mean()
        out_losses_wh[split] = losses_wh.mean()
        out_map50[split] = map50
    model.train()
    return out_losses, out_losses_noobj, out_losses_obj, out_losses_class, out_losses_xy, out_losses_wh, out_map50


# Learning rate decay scheduler (cosine with warmup)
if decay_lr == 'cosine':
    def get_lr(it):
        # 1) Linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
elif decay_lr == 'step':
    def get_lr(it):
        # 1) Linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) After warmup, use step decay with min_lr as gamma
        return learning_rate * (min_lr ** ((it - warmup_iters) // lr_decay_iters))
elif decay_lr == False:
    pass
else:
    raise ValueError(f"Invalid decay_lr: {decay_lr}")


# Logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# Training loop
X, Y, Y_supp = BatchGetter.get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
pbar = tqdm(total=max_iters, initial=iter_num, dynamic_ncols=True)

while True:

    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses, losses_noobj, losses_obj, losses_class, losses_xy, losses_wh, map50 = estimate_loss()
        tqdm.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val map50 {map50['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "train/loss_noobj": losses_noobj['train'],
                "train/loss_obj": losses_obj['train'],
                "train/loss_class": losses_class['train'],
                "train/loss_xy": losses_xy['train'],
                "train/loss_wh": losses_wh['train'],
                "train/map50": map50['train'],
                "val/loss": losses['val'],
                "val/loss_noobj": losses_noobj['val'],
                "val/loss_obj": losses_obj['val'],
                "val/loss_class": losses_class['val'],
                "val/loss_xy": losses_xy['val'],
                "val/loss_wh": losses_wh['val'],
                "val/map50": map50['val'],
                "lr": lr
            })

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
            'rng_state': torch.get_rng_state()
        }
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint['best_val_loss'] = best_val_loss
            if iter_num > 0:
                tqdm.write(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))  # TODO: save top k checkpoints
        if iter_num + eval_interval > max_iters:  # last eval
            checkpoint['best_val_loss'] = losses['val']
            tqdm.write(f"saving last checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt_last.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss, _, _, _, _, _ = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, Y_supp = BatchGetter.get_batch('train')

        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # Clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # Flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # Get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        tqdm.write(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1
    pbar.update(1)

    # Termination conditions
    if iter_num > max_iters:
        break
