# Config for training Darknet19 model on no-aug ImageNet2012 dataset for image classification
# as the upper-bound training accuracy (overfit no-aug data) baseline of the backbone for YOLOv2
# with advanced cosine learning scheduler
import time

# Task related
task_name = 'classify'
init_from = 'scratch'

# Data related
dataset_name = 'imagenet2012'
img_h = 224
img_w = 224
n_class = 1000

# Transform related
# No augmentation
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
brightness = 0.0
contrast = 0.0
saturation = 0.0
hue = 0.0
degrees = 0.0
scale_min = 1.0
scale_max = 1.0
ratio_min = 1.0
ratio_max = 1.0
flip_p = 0.0

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
optimizer_type = 'adam'
learning_rate = 3e-4  # smaller lr to overfit stably
beta1 = 0.9
beta2 = 0.999
weight_decay = 5e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'  # whether to decay the learning rate
warmup_iters = 12515  # warmup 5 epochs
lr_decay_iters = 250300  # should be ~= max_iters
min_lr = 1e-6  # minimum learning rate, should be ~= learning_rate/10, but set to a bit smaller, since imagenet usually decay 0.1 times per 30 epochs
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
