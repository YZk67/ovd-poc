from detrex.config import get_config

from .models.dino_convnextl import model


# D3 Level-0: treat every D3 sent_id as one fixed detection class.
# The default split is d3_intra_full, matching the D3/DOD Full AP reported in
# most paper main tables. Use the *_inter_full config for the harder stress test.
# Generate this file with tools/prepare_d3_metadata.py + tools/generate_text_embeddings.py.
d3_query_path = "dataset/metadata/d3_clip_convnextl_sentences.npy"

model.num_classes = 422
model.query_path = d3_query_path
model.eval_query_path = d3_query_path
model.vlm_query_path = None
model.score_ensemble = False
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes = None
model.all_classes = None
model.unseen_classes = None
model.select_box_nums_for_evaluation = 300

# Use static text weights for the zero-shot transfer smoke test. This keeps the
# config compatible with released LaMI-DETR checkpoints without training TPA.
model.classifier.use_tpa = False
model.classifier.text_embed_path = d3_query_path
model.classifier.eval_text_embed_path = d3_query_path
model.transformer.use_rpsa = False

dataloader = get_config("common/data/d3_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/lvis_schedule.py").lr_multiplier_12ep_warmup
train = get_config("common/train.py").train

train.init_checkpoint = "./pretrained_models/lami_convnext_large_12ep_lvis/model_final.pth"
train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_full"
train.max_iter = 1
train.eval_period = 1
train.log_period = 20
train.checkpointer.period = 1
train.device = "cuda"
model.device = train.device
train.amp.enabled = True

dataloader.test.dataset.names = "d3_intra_full"
dataloader.test.num_workers = 4
dataloader.evaluator.output_dir = train.output_dir
