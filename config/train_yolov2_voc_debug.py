# Config for training YOLOv2 model on Pascal VOC 2007&2012 Detection dataset for object detection
# Train on VOC 2007 trainval and 2012 trainval, and evaluate on VOC 2007 test
import time

# Task related
task_name = 'detect'
init_from = 'backbone'
from_ckpt = 'saved/darknet19-448_imagenet2012/20240303-055517/ckpt_last.pt'

# Data related
dataset_name = 'voc'
img_h = 416
img_w = 416
n_class = 20

# Transform related
# multiscale_min_sizes = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)
multiscale_min_sizes = (416,)  # TODO: enable multiscale
min_wh = 1e-3
# TODO: enable augmentation
# perspective = 0.015
# crop_scale = 0.8
# ratio_min = 0.5
# ratio_max = 2.0
# degrees = 0.5
# translate = 0.1
# scale = 0.25
# shear = 0.5
# brightness = 0.4
# contrast = 0.4
# saturation = 0.7
# hue = 0.015
# flip_p = 0.5
perspective = 0.0
crop_scale = 1.0
ratio_min = 1.0
ratio_max = 1.0
degrees = 0.0
translate = 0.0
scale = 0.0
shear = 0.0
brightness = 0.0
contrast = 0.0
saturation = 0.0
hue = 0.0
flip_p = 0.0
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
anchors_burnin_n_seen_img = 12800
noobj_iou_thresh = 0.6

# Train related
gradient_accumulation_steps = 1
batch_size = 128  # filled up the gpu memory on my machine
max_iters = 160000  # TODO: adjust

# Optimizer related
optimizer_type = 'adamw'
learning_rate = 1e-5  # TODO: adjust
beta1 = 0.9
beta2 = 0.999
weight_decay = 5e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'  # whether to decay the learning rate
warmup_iters = 5  # warmup 5 epochs
lr_decay_iters = 160000  # should be ~= max_iters
min_lr = 0  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
eval_interval = 650  # keep frequent if we'll overfit, there are 10 multiscale sizes
eval_iters = 16  # TODO: adjust
prob_thresh = 0.001
nms_iou_thresh = 0.5

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/yolov2_voc/{timestamp}'
wandb_log = True
wandb_project = 'voc'
wandb_run_name = f'yolov2_{timestamp}'
log_interval = 1  # don't print too often
always_save_checkpoint = True  # only save when val improves if we expect overfit  # TODO: adjust

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 4
