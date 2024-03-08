# Config for finetuning Darknet19 to 448x448 model on ImageNet2012 dataset for image classification as the backbone for YOLOv2
# Simple augmentation as https://github.com/pytorch/vision/blob/main/references/classification/presets.py
import time

# Task related
task_name = 'classify'
init_from = 'pretrained'
from_ckpt = 'out/darknet19_imagenet2012/20240301-062829/ckpt_last.pt'

# Data related
dataset_name = 'imagenet2012'
img_h = 448
img_w = 448
n_class = 1000

# Transform related
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
scale_min = 0.08  # torchvision default
ratio_min = 3.0 / 4.0
ratio_max = 4.0 / 3.0
flip_p = 0.5

# Model related
model_name = 'darknet19'

# Train related
# the number of examples per iter:
# 128 batch_size * 4 grad_accum = 512 imgs/iter
# imagenet2012 train set has 1,281,167 imgs, so 1 epoch ~= 2,503 iters
gradient_accumulation_steps = 4
batch_size = 128  # filled up the gpu memory on my machine
max_iters = 50060  # 20 epochs, 2 times of the paper

# Optimizer related
optimizer_type = 'sgd'
learning_rate = 1e-3  # as paper
beta1 = 0.9
beta2 = 0.999
weight_decay = 5e-4  # as paper
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'  # whether to decay the learning rate
warmup_iters = 0  # no warmup
lr_decay_iters = 50060  # should be ~= max_iters
min_lr = 0  # minimum learning rate, should be ~= learning_rate/10, but set to 0 as pytorch default
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# imagenet2012 val set has 50,000 imgs, so 1 epoch ~= 98 iters
eval_interval = 2503  # keep frequent if we'll overfit
eval_iters = 98  # use entire val to get good estimate

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/darknet19-448_imagenet2012/{timestamp}'
wandb_log = True
wandb_project = 'imagenet2012'
wandb_run_name = f'darknet19-448_{timestamp}'
log_interval = 200  # don't print too often
always_save_checkpoint = False  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 8
