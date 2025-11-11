#!/usr/bin/env python3
"""
分析TPA prototypes的问题
诊断为什么prototypes没有学习到不同的表示
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from examples.tpa_visualization_classes import SELECTED_CLASSES

def analyze_prototype_diversity(
    checkpoint_path: str,
    config_path: str,
    prompts_json_path: str,
):
    """分析prototypes的多样性问题"""
    
    print("=" * 70)
    print("TPA Prototype Diversity Analysis")
    print("=" * 70)
    
    # 加载模型
    print("\n[1] Loading model...")
    cfg = LazyConfig.load(config_path)
    model = instantiate(cfg.model)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    model.eval()
    
    # 获取TPA
    text_classifier = model.transformer.decoder.class_embed[0]
    tpa = text_classifier.tpa
    text_feats = text_classifier.train_text_feats
    
    device = next(model.parameters()).device
    text_feats = text_feats.to(device)
    
    print(f"[Info] Text features shape: {text_feats.shape}")
    print(f"[Info] TPA config:")
    print(f"  - num_prototypes: {tpa.num_prototypes}")
    print(f"  - lambda_orth: {tpa.lambda_orth_base}")
    print(f"  - lambda_div: {tpa.lambda_div_base}")
    print(f"  - tau: {tpa.tau}")
    
    # 运行TPA
    print("\n[2] Running TPA forward pass...")
    with torch.no_grad():
        prototypes, _ = tpa(text_feats, with_loss=False)
        attention_logits = tpa._last_logits
        attention_weights = F.softmax(attention_logits / tpa.tau, dim=-1)
    
    # 加载类别名称
    with open(prompts_json_path, 'r') as f:
        prompts_dict = json.load(f)
    
    all_class_names = list(prompts_dict.keys())
    selected_indices = [all_class_names.index(c) for c in SELECTED_CLASSES if c in all_class_names]
    
    print(f"\n[3] Analyzing {len(selected_indices)} selected classes...")
    
    # 分析每个类别
    print("\n" + "=" * 70)
    print("Per-Class Analysis")
    print("=" * 70)
    
    for cls_idx, cls_name in enumerate(SELECTED_CLASSES[:len(selected_indices)]):
        if cls_name not in all_class_names:
            continue
        
        idx = all_class_names.index(cls_name)
        attn = attention_weights[idx].detach().cpu().numpy()  # [K, N]
        proto = prototypes[idx].detach().cpu()  # [K, D]
        
        print(f"\n【{cls_name}】")
        print(f"  Attention weights shape: {attn.shape}")
        
        # 1. 计算prototypes之间的相似度
        proto_norm = F.normalize(proto, dim=-1)
        proto_sim = torch.mm(proto_norm, proto_norm.t()).numpy()
        print(f"  Prototype similarity matrix (mean off-diag): {proto_sim[~np.eye(proto_sim.shape[0], dtype=bool)].mean():.4f}")
        
        # 2. 计算attention模式的相似度（每个prototype的attention分布）
        # 使用KL散度或余弦相似度
        attn_similarities = []
        for i in range(attn.shape[0]):
            for j in range(i+1, attn.shape[0]):
                # 余弦相似度
                cos_sim = np.dot(attn[i], attn[j]) / (np.linalg.norm(attn[i]) * np.linalg.norm(attn[j]))
                attn_similarities.append(cos_sim)
        
        mean_attn_sim = np.mean(attn_similarities)
        print(f"  Attention pattern similarity (mean): {mean_attn_sim:.4f}")
        print(f"    (1.0 = identical, 0.0 = completely different)")
        
        # 3. 分析每个prototype最关注的prompt
        print(f"  Top prompts for each prototype:")
        for k in range(attn.shape[0]):
            top_prompt_idx = np.argmax(attn[k])
            top_prompt_weight = attn[k, top_prompt_idx]
            print(f"    Proto{k+1}: P{top_prompt_idx+1} (weight={top_prompt_weight:.3f})")
        
        # 4. 检查是否所有prototypes关注相同的prompts
        top_prompts = [np.argmax(attn[k]) for k in range(attn.shape[0])]
        unique_top_prompts = len(set(top_prompts))
        print(f"  Unique top prompts: {unique_top_prompts}/{attn.shape[0]}")
        if unique_top_prompts < attn.shape[0] * 0.6:
            print(f"    ⚠️  WARNING: Most prototypes focus on the same prompts!")
        
        # 5. 计算attention分布的熵（越高越均匀，越低越集中）
        entropies = []
        for k in range(attn.shape[0]):
            entropy = -np.sum(attn[k] * np.log(attn[k] + 1e-8))
            entropies.append(entropy)
        mean_entropy = np.mean(entropies)
        max_entropy = np.log(attn.shape[1])  # 最大熵（均匀分布）
        print(f"  Attention entropy: {mean_entropy:.4f} (max={max_entropy:.4f}, {mean_entropy/max_entropy*100:.1f}% of max)")
    
    # 全局分析
    print("\n" + "=" * 70)
    print("Global Analysis")
    print("=" * 70)
    
    # 计算所有prototypes的正交性
    P = F.normalize(prototypes, dim=-1)  # [C, K, D]
    C, K, D = P.shape
    
    # 每个类别内的prototypes相似度
    intra_class_sims = []
    for c in range(C):
        proto_c = P[c]  # [K, D]
        sim_matrix = torch.mm(proto_c, proto_c.t())  # [K, K]
        off_diag = sim_matrix[~torch.eye(K, dtype=bool, device=sim_matrix.device)]
        intra_class_sims.append(off_diag.mean().item())
    
    print(f"\n[4] Prototype Orthogonality:")
    print(f"  Mean intra-class similarity: {np.mean(intra_class_sims):.4f}")
    print(f"  (Lower is better, 0.0 = perfectly orthogonal)")
    if np.mean(intra_class_sims) > 0.3:
        print(f"    ⚠️  WARNING: Prototypes are too similar within classes!")
    
    # 分析attention多样性
    all_attn = attention_weights.detach().cpu().numpy()  # [C, K, N]
    attn_diversities = []
    for c in range(C):
        attn_c = all_attn[c]  # [K, N]
        # 计算每个prototype的attention分布差异
        similarities = []
        for i in range(K):
            for j in range(i+1, K):
                cos_sim = np.dot(attn_c[i], attn_c[j]) / (np.linalg.norm(attn_c[i]) * np.linalg.norm(attn_c[j]))
                similarities.append(cos_sim)
        attn_diversities.append(1.0 - np.mean(similarities))  # 1 - similarity = diversity
    
    print(f"\n[5] Attention Pattern Diversity:")
    print(f"  Mean diversity: {np.mean(attn_diversities):.4f}")
    print(f"  (Higher is better, 1.0 = completely different patterns)")
    if np.mean(attn_diversities) < 0.3:
        print(f"    ⚠️  WARNING: Attention patterns are too similar!")
    
    # 建议
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)
    
    if np.mean(intra_class_sims) > 0.3:
        print("1. Increase lambda_orth (orthogonality loss weight)")
        print("   Current: lambda_orth =", tpa.lambda_orth_base)
        print("   Suggested: lambda_orth =", tpa.lambda_orth_base * 2)
    
    if np.mean(attn_diversities) < 0.3:
        print("\n2. Increase lambda_div (diversity loss weight)")
        print("   Current: lambda_div =", tpa.lambda_div_base)
        print("   Suggested: lambda_div =", tpa.lambda_div_base * 3)
    
    if np.mean(attn_diversities) < 0.3:
        print("\n3. Consider increasing tau (temperature)")
        print("   Current: tau =", tpa.tau)
        print("   Suggested: tau =", tpa.tau * 1.5)
        print("   (Higher tau = softer attention = more exploration)")
    
    print("\n4. Check training logs for loss_orth and loss_div")
    print("   - loss_orth should decrease (prototypes become more orthogonal)")
    print("   - loss_div should decrease (attention patterns become more diverse)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TPA prototype diversity")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    parser.add_argument("--prompts-json", type=str, default="dataset2/metadata/lvis_prompts_claude.json")
    
    args = parser.parse_args()
    analyze_prototype_diversity(args.checkpoint, args.config, args.prompts_json)

