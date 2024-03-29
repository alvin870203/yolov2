# Config for training YOLOv2 model on Pascal VOC 2007&2012 Detection dataset for object detection
# Train on VOC 2007 trainval and 2012 trainval, and evaluate on VOC 2007 test
import time

# Task related
task_name = 'detect'
init_from = 'backbone'
from_ckpt = 'saved/darknet19-448_imagenet2012/20240303-055517/ckpt_last.pt'

# Data related
dataset_name = 'voc'
img_h = 608
img_w = 608
n_class = 20

# Transform related
# TODO: enable multiscale
multiscale_min_sizes = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)
# multiscale_min_sizes = (608,)
min_wh = 1e-3
# TODO: stronger augmentation
perspective = 0.1
crop_scale = 0.3
ratio_min = 0.5
ratio_max = 2.0
degrees = 0.5
translate = 0.25
scale = 0.5
shear = 0.5
brightness = 0.7
contrast = 0.7
saturation = 0.7
hue = 0.1
flip_p = 0.5
# TODO: default augmentation
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
# TODO: no augmentation
# perspective = 0.0
# crop_scale = 1.0
# ratio_min = 1.0
# ratio_max = 1.0
# degrees = 0.0
# translate = 0.0
# scale = 0.0
# shear = 0.0
# brightness = 0.0
# contrast = 0.0
# saturation = 0.0
# hue = 0.0
# flip_p = 0.0
letterbox = True  # TODO: try letterbox=False
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
fill = (123.0, 117.0, 104.0)

# Model related
model_name = 'yolov2'
n_box_per_cell = 5
# TODO: see which anchors are better, paper's or AlexeyAB's
# anchors = ((1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52))  # paper's
anchors = ((1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071))  # AlexeyAB's darknet

# Loss related
lambda_noobj = 0.5
lambda_obj = 1.0
lambda_class = 1.0
lambda_coord = 5.0
lambda_burnin = 0.01
anchors_burnin_n_seen_img = 166400  # 10 epochs, larger than AlexeyAB's 12800 seen imgs  # TODO: adjust with batch_size
noobj_iou_thresh = 0.6
match_by_anchors = True
rescore = True

# Train related
# the number of examples per iter:
# 512 batch_size * 2 grad_accum = 128 imgs/iter
# voc train set has 16,551 imgs, so 1 epoch ~= 130 iters
gradient_accumulation_steps = 2
batch_size = 64  # filled up the gpu memory on my machine
max_iters = 78000  # 600 epochs, about a day on my machine  # TODO: adjust with batch_size

# Optimizer related
optimizer_type = 'sgd'  # TODO: sgd causes nan loss when no warmup
learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.999
weight_decay = 5e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'  # whether to decay the learning rate
warmup_iters = 390  # warmup 3 epochs, sgd causes nan loss when no warmup  # TODO: adjust with batch_size
lr_decay_iters = 78000  # should be ~= max_iters  # TODO: adjust with batch_size
min_lr = 5e-5  # should be ~= learning_rate/10
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# voc val set has 4,952 imgs, so 1 epoch ~= 39 iters
eval_interval = 650  # keep frequent if we'll overfit, there are 10 multiscale sizes
eval_iters = 39  # use entire val set to get good estimate  # TODO: adjust to speed up
prob_thresh = 0.001
nms_iou_thresh = 0.5

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/yolov2_voc/{timestamp}'
wandb_log = True
wandb_project = 'voc'
wandb_run_name = f'yolov2_{timestamp}'
log_interval = 10  # don't print too often
always_save_checkpoint = True  # only save when val improves if we expect overfit  # TODO: adjust

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 8  # TODO: adjust
