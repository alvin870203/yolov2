# Config for training YOLOv2 model on nano Pascal VOC 2007&2012 Detection dataset for object detection debug
# Train on nano VOC 2007 trainval and 2012 trainval, and evaluate on nano VOC 2007 test
import time

# Task related
task_name = 'detect'
init_from = 'backbone'
from_ckpt = 'saved/darknet19-448_imagenet2012/20240303-055517/ckpt_last.pt'

# Data related
dataset_name = 'nano_voc'
img_h = 416
img_w = 416
n_class = 20

# Transform related
multiscale_min_sizes = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)
min_wh = 1e-3
perspective = 0.015
crop_scale = 0.8
ratio_min = 0.5
ratio_max = 2.0
degrees = 0.5
translate = 0.1
scale = 0.25
shear = 0.5
brightness = 0.4
contrast = 0.4
saturation = 0.7
hue = 0.015
flip_p = 0.5
letterbox = True
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
fill = (123.0, 117.0, 104.0)

# Model related
model_name = 'yolov2'
n_box_per_cell = 5
anchors = ((1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52))

# Loss related
lambda_noobj = 0.5
lambda_obj = 1.0
lambda_class = 1.0
lambda_coord = 5.0
lambda_burnin = 0.01
anchors_burnin_n_seen_img = 20
noobj_iou_thresh = 0.6
match_by_anchors = True
rescore = False

# Train related
# the number of examples per iter:
# 2 batch_size * 1 grad_accum = 2 imgs/iter
# nano voc train set has 2 imgs, so 1 epoch ~= 1 iters
gradient_accumulation_steps = 1
batch_size = 2  # entire dataset
max_iters = 16000

# Optimizer related
optimizer_type = 'adamw'
learning_rate = 1e-5  # nano lr to overfit nano dataset stably
beta1 = 0.9
beta2 = 0.999
weight_decay = 5e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = False  # whether to decay the learning rate
#warmup_iters = 5  # warmup 5 epochs
#lr_decay_iters = 16000  # should be ~= max_iters
#min_lr = 0  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# voc val set has 2 imgs, so 1 epoch ~= 1 iters
eval_interval = 20  # keep frequent if we'll overfit, there are 10 multiscale sizes
eval_iters = 1  # use entire val set to get good estimate
prob_thresh = 0.001
nms_iou_thresh = 0.5

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/yolov2_nano-voc/{timestamp}'
wandb_log = True
wandb_project = 'nano-voc'
wandb_run_name = f'yolov2_{timestamp}'
log_interval = 1  # don't print too often
always_save_checkpoint = False  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 4
