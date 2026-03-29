import json
import torch

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
    prob = appeared.new_ones(C).float()
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight[:C].float().clone()
        prob[appeared[appeared < C]] = 0
        num_remaining = max(0, num_sample_cats - len(appeared))
        if prob.sum() > 0 and num_remaining > 0:
            num_remaining = min(num_remaining, (prob > 0).sum().item())
            more_appeared = torch.multinomial(
                prob, num_remaining,
                replacement=False)
            appeared = torch.cat([appeared, more_appeared])
    return appeared

def get_cluster_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None, cluster_label=None):
    appeared = torch.unique(gt_classes) # C'
    # Only use indices 0..C-1 for sampling (index C is a dummy slot, never sampled)
    prob = appeared.new_ones(C).float()
    if cluster_label is not None:
        cluster_label = torch.tensor(cluster_label[:C])  # ensure cluster_label matches C
        gt_classes_valid = appeared[appeared < C]
        if len(gt_classes_valid) > 0:
            gt_classes_cluster = cluster_label[gt_classes_valid]
            appeared_cluster = torch.unique(gt_classes_cluster)
            same_cluster_class = torch.nonzero(torch.isin(cluster_label, appeared_cluster)).squeeze()
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight[:C].float().clone()
        prob[appeared[appeared < C]] = 0
        if cluster_label is not None and len(gt_classes_valid) > 0:
            prob[same_cluster_class] = 0
        num_remaining = max(0, num_sample_cats - len(appeared))
        if prob.sum() > 0 and num_remaining > 0:
            num_remaining = min(num_remaining, (prob > 0).sum().item())
            more_appeared = torch.multinomial(
                prob, num_remaining,
                replacement=False)
            appeared = torch.cat([appeared, more_appeared])
    return appeared