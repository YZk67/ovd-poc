#!/usr/bin/env python3
"""
检查TPA prototype_queries的梯度是否在更新
"""

import sys
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer


def check_tpa_gradients(checkpoint_path: str, config_path: str):
    """检查TPA prototype_queries的梯度"""
    print("=" * 70)
    print("检查TPA prototype_queries的梯度")
    print("=" * 70)
    
    # 加载配置
    cfg = LazyConfig.load(config_path)
    cfg = LazyConfig.apply_overrides(cfg, [])
    
    # 创建模型
    model = instantiate(cfg.model)
    model.eval()  # 先设置为eval模式
    
    # 加载checkpoint
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    
    # 获取TPA模块
    text_classifier = model.transformer.decoder.class_embed[0]
    if not hasattr(text_classifier, 'tpa'):
        print("❌ 找不到TPA模块！")
        return
    
    tpa = text_classifier.tpa
    print(f"\n【TPA模块信息】")
    print(f"  num_prototypes: {tpa.num_prototypes}")
    print(f"  hidden_dim: {tpa.hidden_dim}")
    print(f"  lambda_orth_base: {tpa.lambda_orth_base}")
    print(f"  lambda_div_base: {tpa.lambda_div_base}")
    print(f"  warmup_steps: {tpa.warmup_steps}")
    
    # 检查_step
    current_step = tpa._step.item() if isinstance(tpa._step, torch.Tensor) else tpa._step
    print(f"  current_step: {current_step}")
    
    # 检查prototype_queries
    prototype_queries = tpa.prototype_queries
    print(f"\n【prototype_queries参数】")
    print(f"  shape: {prototype_queries.shape}")
    print(f"  requires_grad: {prototype_queries.requires_grad}")
    print(f"  dtype: {prototype_queries.dtype}")
    print(f"  device: {prototype_queries.device}")
    
    # 检查参数值
    print(f"\n【prototype_queries的值】")
    print(f"  mean: {prototype_queries.data.mean().item():.6f}")
    print(f"  std: {prototype_queries.data.std().item():.6f}")
    print(f"  min: {prototype_queries.data.min().item():.6f}")
    print(f"  max: {prototype_queries.data.max().item():.6f}")
    
    # 检查prototypes之间的相似度
    with torch.no_grad():
        normalized = torch.nn.functional.normalize(prototype_queries.data, p=2, dim=1)
        similarity_matrix = torch.mm(normalized, normalized.t())
        off_diag_mask = ~torch.eye(similarity_matrix.size(0), dtype=bool, device=similarity_matrix.device)
        off_diag_similarities = similarity_matrix[off_diag_mask]
        
        print(f"\n【prototype_queries之间的相似度】")
        print(f"  mean similarity: {off_diag_similarities.mean().item():.6f}")
        print(f"  max similarity: {off_diag_similarities.max().item():.6f}")
        print(f"  min similarity: {off_diag_similarities.min().item():.6f}")
        
        if off_diag_similarities.mean().item() > 0.99:
            print(f"  ⚠️  所有prototypes几乎完全相同！")
        elif off_diag_similarities.mean().item() > 0.9:
            print(f"  ⚠️  Prototypes非常相似")
        else:
            print(f"  ✅ Prototypes有差异")
    
    # 检查梯度（需要forward一次）
    print(f"\n【检查梯度】")
    model.train()  # 设置为训练模式
    
    # 创建dummy输入
    device = next(model.parameters()).device
    C = 1203  # num_classes
    N = 8     # num_prompts
    D = 256   # feat_dim
    
    # 获取text_feats（从TPA中获取）
    if hasattr(text_classifier, 'train_text_feats'):
        text_feats = text_classifier.train_text_feats
        if text_feats.device != device:
            text_feats = text_feats.to(device)
    else:
        # 创建dummy text_feats
        text_feats = torch.randn(C, N, D, device=device)
    
    # Forward pass
    tpa.zero_grad()
    prototypes, apr_loss = tpa(text_feats, with_loss=True)
    
    # 检查梯度
    if prototype_queries.grad is not None:
        grad = prototype_queries.grad
        print(f"  ✅ 梯度存在")
        print(f"  gradient shape: {grad.shape}")
        print(f"  gradient mean: {grad.abs().mean().item():.6f}")
        print(f"  gradient std: {grad.std().item():.6f}")
        print(f"  gradient max: {grad.abs().max().item():.6f}")
        print(f"  gradient min: {grad.abs().min().item():.6f}")
        
        # 检查梯度是否为零
        if grad.abs().mean().item() < 1e-8:
            print(f"  ❌ 梯度几乎为零！可能没有正确反向传播")
        else:
            print(f"  ✅ 梯度非零，说明在更新")
        
        # 检查每个prototype的梯度
        print(f"\n【每个prototype的梯度大小】")
        for i in range(tpa.num_prototypes):
            grad_norm = grad[i].norm().item()
            print(f"  prototype {i}: {grad_norm:.6f}")
    else:
        print(f"  ❌ 梯度不存在！")
        print(f"  可能的原因：")
        print(f"    1. 还没有进行反向传播")
        print(f"    2. requires_grad=False")
        print(f"    3. 梯度被detach了")
    
    # 检查loss
    print(f"\n【APR Loss】")
    print(f"  apr_loss: {apr_loss.item():.6f}")
    if hasattr(tpa, 'last_loss_terms'):
        loss_terms = tpa.last_loss_terms
        print(f"  loss_orth: {loss_terms.get('loss_orth', 'N/A')}")
        print(f"  loss_div: {loss_terms.get('loss_div', 'N/A')}")
        print(f"  lambda_orth: {loss_terms.get('lambda_orth', 'N/A')}")
        print(f"  lambda_div: {loss_terms.get('lambda_div', 'N/A')}")
    
    # 检查effective lambda
    lam_orth, lam_div = tpa._effective_lambdas()
    print(f"\n【Effective Lambda】")
    print(f"  effective lambda_orth: {lam_orth:.6f}")
    print(f"  effective lambda_div: {lam_div:.6f}")
    
    if current_step < tpa.warmup_steps:
        progress = (current_step + 1) / float(tpa.warmup_steps)
        print(f"  ⚠️  还在warmup阶段，progress: {progress:.2%}")
    else:
        print(f"  ✅ 已过warmup阶段")
    
    # 反向传播测试
    print(f"\n【反向传播测试】")
    apr_loss.backward()
    
    if prototype_queries.grad is not None:
        grad_after = prototype_queries.grad
        print(f"  ✅ 反向传播后梯度存在")
        print(f"  gradient mean: {grad_after.abs().mean().item():.6f}")
        
        # 检查梯度是否来自diversity loss
        print(f"\n【梯度来源分析】")
        print(f"  如果梯度很小，可能是：")
        print(f"    1. diversity loss的梯度被其他loss覆盖")
        print(f"    2. lambda_div太小")
        print(f"    3. 梯度被裁剪了")
    else:
        print(f"  ❌ 反向传播后梯度仍然不存在！")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    check_tpa_gradients(args.checkpoint, args.config)

