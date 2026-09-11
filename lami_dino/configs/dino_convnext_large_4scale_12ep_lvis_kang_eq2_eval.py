"""Same-checkpoint Eq. (2) counterfactual for the released Kang model.

This config reconstructs the historical inference settings around the collapsed
``model_final_ovd_lvis_kang.pth`` checkpoint.  The runner evaluates the same
weights twice and changes only ``tpa_eval_legacy_logsumexp``:

* False: duplicate-invariant calibrated log-mean-exp.
* True: historical uncalibrated log-sum-exp.

Do not use this config for training.  Its purpose is causal attribution of the
legacy Eq. (2), not a new performance protocol.
"""

from .dino_convnext_large_4scale_12ep_lvis import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Historical score-ensemble protocol used for the reported Kang checkpoint.
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0
model.select_box_nums_for_evaluation = 300
model.inference_query_class_topk = 0

# Historical TPA/query path.  The old implementation divided attention logits
# by sqrt(d_h) and then tau=0.07, so 0.07 (not the paper-aligned 0.004375) is
# required to reconstruct the checkpoint's prototype bank.
model.classifier.tpa_tau = 0.07
model.classifier.tpa_cls_tau = 0.07
model.classifier.tpa_slot_prior_strength = 0.0
model.classifier.tpa_prototype_mode_strength = 0.0
model.classifier.tpa_identity_value_init = False
model.classifier.tpa_eval_logit_bias = 0.0
model.classifier.tpa_eval_legacy_logsumexp = False
model.tpa_eval_mode_scale = 1.0

# The historical query initializer committed to the encoder's top-1 category,
# then performed soft routing only among that category's prototypes.
model.use_soft_attention = True
model.soft_attention_tau = 0.15
model.soft_category_topk = 1

# Eval-only loading always restores the full model.  Keeping this explicit also
# prevents accidental backbone-only use if the config is inspected elsewhere.
train.init_checkpoint_scope = "full"
train.output_dir = "./output/kang_eq2_counterfactual"
dataloader.evaluator.output_dir = train.output_dir
