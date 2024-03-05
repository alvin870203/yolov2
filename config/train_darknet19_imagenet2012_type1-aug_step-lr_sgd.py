# Config for finetuning Darknet19 model on ImageNet2012 dataset for image classification as the backbone for YOLOv2
import time

# Task related
task_name = 'classify'
init_from = 'pretrained'
from_ckpt = 'out/darknet19_imagenet2012/20240301-062829/ckpt_last.pt'

# Data related
dataset_name = 'imagenet2012'
img_h = 224
img_w = 224
n_class = 1000

# Transform related
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
brightness = 0.75
contrast = 0.75
saturation = 0.75
hue = 0.5
degrees = 7
scale_min = 0.5
scale_max = 2.0
ratio_min = 3.0 / 4.0
ratio_max = 4.0 / 3.0
flip_p = 0.5

# Model related
model_name = 'darknet19'

# Train related
# the number of examples per iter:
# 512 batch_size * 1 grad_accum = 512 imgs/iter
# imagenet2012 train set has 1,281,167 imgs, so 1 epoch ~= 2,503 iters
gradient_accumulation_steps = 1
batch_size = 512  # filled up the gpu memory on my machine
max_iters = 250300  # finish in ~25.5 hr on my machine

# Optimizer related
optimizer_type = 'sgd'
learning_rate = 0.1
beta1 = 0.9  # momentum
#beta2 = 0.999  # not used in sgd
weight_decay = 1e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'step'  # whether to decay the learning rate
warmup_iters = 0  # no warmup
lr_decay_iters = 75090  # decay every 30 epochs
min_lr = 0.1  # decay 1/10 every lr_decay_iters
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# imagenet2012 val set has 50,000 imgs, so 1 epoch ~= 98 iters
eval_interval = 2503  # keep frequent if we'll overfit
eval_iters = 98  # use entire val to get good estimate

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/darknet19_imagenet2012/{timestamp}'
wandb_log = True
wandb_project = 'imagenet2012'
wandb_run_name = f'darknet19_{timestamp}'
log_interval = 200  # don't print too often
always_save_checkpoint = False  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 8
