import json
import os
import torch


_FED_DEBUG_PRINTED = False


def _safe_multinomial_probs(probs, candidate_mask=None):
    # --- safety for multinomial probs ---
    probs = probs.clone()

    # remove NaN/Inf and negatives
    probs[~torch.isfinite(probs)] = 0
    probs = torch.clamp(probs, min=0)

    if candidate_mask is not None:
        cand = candidate_mask.clone()
        if cand.dtype != torch.bool:
            cand = cand.bool()
        probs[~cand] = 0

    s = probs.sum()
    if s <= 0:
        # fallback: uniform over allowed candidates if you have a mask,
        # otherwise uniform over all classes
        if candidate_mask is not None:
            cand = candidate_mask.clone()
            if cand.dtype != torch.bool:
                cand = cand.bool()
            if cand.any():
                probs = cand.float()
            else:
                probs = torch.ones_like(probs)
        else:
            probs = torch.ones_like(probs)
        probs = probs / probs.sum()
    else:
        probs = probs / s
    # --- end safety ---
    return probs


def _maybe_debug_bad_probs(probs, *, appeared=None, candidate_mask=None):
    global _FED_DEBUG_PRINTED
    if _FED_DEBUG_PRINTED or os.environ.get("FED_SAMPLING_DEBUG", "0") != "1":
        return
    bad = (not torch.isfinite(probs).all()) or probs.sum() <= 0
    if not bad:
        return
    p = torch.nan_to_num(probs)
    print(
        "[FedSampling DEBUG] sum=",
        float(p.sum()),
        "nonzero=",
        int((p > 0).sum()),
        "min=",
        float(p.min()),
        "max=",
        float(p.max()),
    )
    if appeared is not None:
        print("[FedSampling DEBUG] appeared=", int(torch.unique(appeared).numel()))
    if candidate_mask is not None:
        cm = candidate_mask.bool() if candidate_mask.dtype != torch.bool else candidate_mask
        print("[FedSampling DEBUG] candidate_mask.sum()=", int(cm.sum()))
    _FED_DEBUG_PRINTED = True

def load_class_freq(
    path='datasets/metadata/lvis_v1_train_cat_info.json', freq_weight=0.5):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    cat_info = cat_info.float()
    freq_weight = cat_info ** freq_weight
    return freq_weight

def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        candidate_mask = prob > 0
        candidate_mask[-1] = False  # never sample background index
        _maybe_debug_bad_probs(prob, appeared=appeared, candidate_mask=candidate_mask)
        probs = _safe_multinomial_probs(prob, candidate_mask=candidate_mask)
        nonzero = int((probs > 0).sum().item())
        need = num_sample_cats - len(appeared)
        replacement = nonzero < need
        more_appeared = torch.multinomial(
            probs, need,
            replacement=replacement)
        appeared = torch.cat([appeared, more_appeared])
    return appeared

def get_cluster_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None, cluster_label=None):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if cluster_label is not None:
        cluster_label = torch.tensor(cluster_label)# [8212]
        gt_classes_cluster = cluster_label[appeared]# [58]
        appeared_cluster = torch.unique(gt_classes_cluster)# [38]
        same_cluster_class = torch.nonzero(torch.isin(cluster_label, appeared_cluster)).squeeze()
        other_cluster_class = torch.nonzero(~torch.isin(cluster_label, appeared_cluster)).squeeze()
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        if cluster_label is not None:
            prob[same_cluster_class] = 0# 只取不同cluster
        candidate_mask = prob > 0
        candidate_mask[-1] = False  # never sample background index
        _maybe_debug_bad_probs(prob, appeared=appeared, candidate_mask=candidate_mask)
        probs = _safe_multinomial_probs(prob, candidate_mask=candidate_mask)
        nonzero = int((probs > 0).sum().item())
        need = num_sample_cats - len(appeared)
        replacement = nonzero < need
        more_appeared = torch.multinomial(
            probs, need,
            replacement=replacement)
        appeared = torch.cat([appeared, more_appeared])
    return appeared
