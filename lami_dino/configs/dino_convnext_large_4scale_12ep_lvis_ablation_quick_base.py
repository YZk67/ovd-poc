from .dino_convnext_large_4scale_12ep_lvis import model, dataloader, optimizer, lr_multiplier, train


_EMBED_ROOT = "dataset/metadata/ablation"
_LLM_PROMPT_BANK = "dataset/metadata/lvis_claude_prompts_convnextl.npy"
_QUICK_ITERS = 14200
_QUICK_EVAL_PERIOD = 7100


train.max_iter = _QUICK_ITERS
train.eval_period = _QUICK_EVAL_PERIOD
train.checkpointer.period = _QUICK_EVAL_PERIOD
train.log_period = 200
train.output_dir = "/root/prototype_ablation_quick/base"
dataloader.evaluator.output_dir = train.output_dir

# Keep quick ablations focused on the text prototype pathway rather than the
# separate VLM score ensemble branch.
model.score_ensemble = False
model.backbone.score_ensemble = model.score_ensemble
model.vlm_query_path = None
model.alpha = 0.0
model.beta = 0.0
model.novel_scale = 1.0


def _set_output_dir(name):
    train.output_dir = f"/root/prototype_ablation_quick/{name}"
    dataloader.evaluator.output_dir = train.output_dir


def _set_static_embedding(path, name):
    model.query_path = path
    model.eval_query_path = path
    model.classifier.use_tpa = False
    model.classifier.text_embed_path = path
    model.classifier.eval_text_embed_path = path
    model.use_soft_attention = False
    model.transformer.use_rpsa = False
    model.criterion.weight_dict["loss_apr"] = 0.0
    model.criterion.weight_dict["loss_rpsa"] = 0.0
    _set_output_dir(name)


def _set_tpa(name, apr_weight, use_rpsa):
    model.query_path = _LLM_PROMPT_BANK
    model.eval_query_path = _LLM_PROMPT_BANK
    model.classifier.use_tpa = True
    model.classifier.text_embed_path = _LLM_PROMPT_BANK
    model.classifier.eval_text_embed_path = _LLM_PROMPT_BANK
    model.classifier.tpa_num_prototypes = 5
    model.use_soft_attention = True
    model.criterion.weight_dict["loss_apr"] = apr_weight
    model.transformer.use_rpsa = use_rpsa
    model.criterion.weight_dict["loss_rpsa"] = 0.05 if use_rpsa else 0.0
    if use_rpsa:
        model.transformer.rpsa_warmup_start = 0
        model.transformer.rpsa_warmup_iters = 1500
    _set_output_dir(name)
