cat > lami_dino/configs/eval_val_t.py <<'PY'
from detrex.config import get_config
from .dino_convnext_large_4scale_12ep_lvis import *  # 复用你训练时的全部设置
from copy import deepcopy

_base_dl = get_config("common/data/coco_detr.py").dataloader

tt = deepcopy(_base_dl.test)
tt.dataset.names = "ovcoco_2017_val_t"
tt.num_workers = 0

ev = deepcopy(_base_dl.evaluator)
ev.dataset_name = "ovcoco_2017_val_t"

dataloader = {}
dataloader["test"] = tt
dataloader["evaluator"] = ev

train.eval_period = 99999999
PY