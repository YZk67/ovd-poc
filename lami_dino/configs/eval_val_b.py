cat > lami_dino/configs/eval_val_b.py <<'PY'
from detrex.config import get_config
from .dino_convnext_large_4scale_12ep_lvis import *  # 复用你训练时的全部设置
from copy import deepcopy

# 单独构造一个 test + evaluator，避免列表与插值
_base_dl = get_config("common/data/coco_detr.py").dataloader

tb = deepcopy(_base_dl.test)
tb.dataset.names = "ovcoco_2017_val_b"
tb.num_workers = 0

ev = deepcopy(_base_dl.evaluator)
ev.dataset_name = "ovcoco_2017_val_b"

# 覆盖成单一 test/evaluator
dataloader = {}
dataloader["test"] = tb
dataloader["evaluator"] = ev

# 训练期评测关掉（以防万一）
train.eval_period = 99999999
PY