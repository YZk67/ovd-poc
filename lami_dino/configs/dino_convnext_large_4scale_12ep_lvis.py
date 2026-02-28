from detrex.config import get_config
from .models.dino_convnextl import model

model.vlm_query_path = "dataset/metadata/ovdcoco_confuse_convnextl.npy"
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes = 'dataset/metadata/ovcoco_seen_classes.json'
model.all_classes = 'dataset/metadata/ovcoco_all_classes.json'
model.vlm_temperature = 100.0 # keep same with f-vlm
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0
model.unknown_match_mode = "soft"
model.unknown_match_cost = 1.5
model.unknown_iou_thr_start = 0.2
model.unknown_iou_thr = 0.4
model.unknown_warmup_iters = 4000
model.pseudo_score_thr = 0.9

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep_warmup
train = get_config("common/train.py").train
train.ddp.find_unused_parameters = True


# modify training config
# train.init_checkpoint = "clip_convnext_large_trans.pth"
train.init_checkpoint = "./pretrained_models/clip_convnext_large_trans.pth"
train.output_dir = "./output/dino_convnext_large_ovcoco65_baseline"

# max training iterations
train.max_iter = 85200# TODO

# run evaluation every 5000 iters
train.eval_period = 7100

# log training infomation every 20 iters
train.log_period = 200

# save checkpoint every 5000 iters
train.checkpointer.period = 7100

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

model.num_classes = 65
model.query_path = "dataset/metadata/ovdcoco_visual_desc_convnextl.npy"
model.eval_query_path = "dataset/metadata/ovdcoco_visual_desc_convnextl.npy"

model.use_fed_loss = True
model.cluster_fed_loss = True
model.cluster_label_path = 'dataset/cluster/ovdcoco_cluster4_16.npy'
model.cat_freq_path = "dataset/coco/ovd_ins_train2017_all_cat_info.json"
model.fed_loss_num_cat=32
model.select_box_nums_for_evaluation = 2000

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.dataset.names = "ovd_coco_train_with_pseudo"
dataloader.train.num_workers = 4

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
dataloader.test.dataset.names = "ovdcoco65_2017_val_all"
