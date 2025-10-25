# Copyright (c) 2025
# Post-training visualization and analysis for LaMI-DETR

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from pathlib import Path

@torch.no_grad()
def analyze_prototypes(model, save_dir="analysis_outputs"):
    """
    Compute prototype orthogonality, entropy, and visualize embedding layout.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    tpa = model.transformer.decoder.class_embed[0].tpa
    P = F.normalize(tpa.last_prototypes.detach().cpu(), dim=-1)   # [C,K,D]
    C, K, D = P.shape

    # 1. Orthogonality matrix heatmap
    G = torch.einsum("ckd,cmd->ckm", P, P)  # [C,K,K]
    mean_orth = float((G - torch.eye(K)).abs().mean())
    plt.figure(figsize=(6,5))
    sns.heatmap(G[0], cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Prototype correlation (class 0) mean|offdiag|={mean_orth:.3f}")
    plt.savefig(Path(save_dir)/"prototype_corr_heatmap.png"); plt.close()

    # 2. Usage entropy
    if hasattr(tpa, "last_attention"):
        A = tpa.last_attention.detach().cpu()
        usage = A.mean(dim=-1).mean().item()
        np.save(Path(save_dir)/"prototype_usage_entropy.npy", np.array([usage]))

    # 3. TSNE projection of prototypes
    proto_2d = TSNE(n_components=2, init="random", learning_rate="auto").fit_transform(P.view(C*K, D))
    plt.figure(figsize=(6,6))
    plt.scatter(proto_2d[:,0], proto_2d[:,1], s=8, c=np.repeat(np.arange(C), K), cmap="tab20")
    plt.title("t-SNE of all prototypes")
    plt.savefig(Path(save_dir)/"prototype_tsne.png"); plt.close()

    print(f"[Analyzer] Prototype mean orthogonality={mean_orth:.4f}, saved to {save_dir}")

@torch.no_grad()
def visualize_region_alignment(model, image_tensor, save_path="analysis_outputs/region_proto_heatmap.png"):
    """
    Visualize region-prototype similarity map for one image.
    image_tensor: [1,3,H,W] normalized input
    """
    out = model.backbone(image_tensor)
    region_feats = out.flatten(2).permute(0,2,1)  # [1,N,256]
    tpa = model.transformer.decoder.class_embed[0].tpa
    proto = F.normalize(tpa.last_prototypes.mean(1), dim=-1)      # [C,256]
    sim = torch.matmul(F.normalize(region_feats[0],dim=-1), proto.T)  # [N,C]
    sim_map = sim[:,0].view(int(image_tensor.shape[2]/16), int(image_tensor.shape[3]/16))
    plt.figure(figsize=(5,5))
    sns.heatmap(sim_map.cpu(), cmap="magma")
    plt.title("Region-Prototype similarity (class 0)")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"[Analyzer] Saved region-prototype heatmap → {save_path}")


# Example usage
# from analysis.post_eval_analyzer import analyze_prototypes, visualize_region_alignment
# ckpt = "output/checkpoint_best.pth"
# model = instantiate(cfg.model)
# DetectionCheckpointer(model).load(ckpt)
# model.eval().to("cuda")


# analyze_prototypes(model, save_dir="analysis_outputs")
# visualize_region_alignment(model, sample_image_tensor)
