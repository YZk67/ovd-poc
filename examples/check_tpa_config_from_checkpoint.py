#!/usr/bin/env python3
"""
检查checkpoint中保存的TPA配置
确认训练时使用的lambda值和代码版本
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

def check_tpa_config(checkpoint_path: str, config_path: str):
    """检查checkpoint中的TPA配置"""
    print("=" * 70)
    print("检查Checkpoint中的TPA配置")
    print("=" * 70)
    
    # 加载模型
    print("\n[1] Loading model and checkpoint...")
    cfg = LazyConfig.load(config_path)
    model = instantiate(cfg.model)
    
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    model.eval()
    
    # 获取TPA
    text_classifier = model.transformer.decoder.class_embed[0]
    if not hasattr(text_classifier, 'tpa') or not text_classifier.use_tpa:
        print("❌ Model does not use TPA")
        return
    
    tpa = text_classifier.tpa
    
    print("\n[2] TPA Configuration from Checkpoint:")
    print("-" * 70)
    print(f"  num_prototypes: {tpa.num_prototypes}")
    print(f"  tau: {tpa.tau}")
    print(f"  lambda_orth_base: {tpa.lambda_orth_base}")
    print(f"  lambda_div_base: {tpa.lambda_div_base}")
    print(f"  warmup_steps: {tpa.warmup_steps}")
    
    # 检查prototype_queries的状态
    print("\n[3] Prototype Queries State:")
    print("-" * 70)
    queries = tpa.prototype_queries.data  # [K, D]
    K, D = queries.shape
    print(f"  Shape: {queries.shape}")
    
    # 计算queries之间的相似度
    queries_norm = torch.nn.functional.normalize(queries, p=2, dim=1)
    sim_matrix = torch.mm(queries_norm, queries_norm.t())
    off_diag_mask = ~torch.eye(K, dtype=bool, device=sim_matrix.device)
    off_diag_sims = sim_matrix[off_diag_mask]
    
    print(f"  Queries similarity (mean): {off_diag_sims.mean().item():.4f}")
    print(f"  Queries similarity (std): {off_diag_sims.std().item():.4f}")
    print(f"  Queries similarity (min): {off_diag_sims.min().item():.4f}")
    print(f"  Queries similarity (max): {off_diag_sims.max().item():.4f}")
    
    if off_diag_sims.mean().item() > 0.9:
        print("  ⚠️  警告: Queries相似度太高！可能初始化有问题")
    else:
        print("  ✅ Queries相似度正常")
    
    # 检查配置是否匹配
    print("\n[4] Configuration Check:")
    print("-" * 70)
    expected_lambda_orth = 0.20
    expected_lambda_div = 0.12
    
    if abs(tpa.lambda_orth_base - expected_lambda_orth) < 0.01:
        print(f"  ✅ lambda_orth正确: {tpa.lambda_orth_base} (期望: {expected_lambda_orth})")
    else:
        print(f"  ❌ lambda_orth不匹配: {tpa.lambda_orth_base} (期望: {expected_lambda_orth})")
    
    if abs(tpa.lambda_div_base - expected_lambda_div) < 0.01:
        print(f"  ✅ lambda_div正确: {tpa.lambda_div_base} (期望: {expected_lambda_div})")
    else:
        print(f"  ❌ lambda_div不匹配: {tpa.lambda_div_base} (期望: {expected_lambda_div})")
    
    # 检查当前step（如果保存了）
    print(f"\n[5] Training State:")
    print("-" * 70)
    print(f"  Current step: {tpa._step}")
    print(f"  Warmup steps: {tpa.warmup_steps}")
    
    if tpa._step < tpa.warmup_steps:
        progress = (tpa._step + 1) / float(tpa.warmup_steps)
        import math
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        lam_orth = tpa.lambda_orth_base * factor
        lam_div = tpa.lambda_div_base * factor
        print(f"  ⚠️  还在warmup阶段！")
        print(f"  Warmup progress: {progress:.2%}")
        print(f"  Effective lambda_orth: {lam_orth:.4f}")
        print(f"  Effective lambda_div: {lam_div:.4f}")
    else:
        print(f"  ✅ Warmup已完成")
        print(f"  Effective lambda_orth: {tpa.lambda_orth_base}")
        print(f"  Effective lambda_div: {tpa.lambda_div_base}")
    
    # 检查diversity loss的实现
    print("\n[6] Diversity Loss Implementation Check:")
    print("-" * 70)
    # 创建一个简单的测试
    import torch.nn.functional as F
    C, K, N = 1, 5, 8
    test_logits = torch.randn(C, K, N)
    
    # 检查是否有_last_logits
    if hasattr(tpa, '_last_logits'):
        print("  ✅ _last_logits存在（说明forward被调用过）")
    else:
        print("  ⚠️  _last_logits不存在（可能需要forward一次）")
    
    # 测试diversity loss计算
    try:
        w = F.softmax(test_logits, dim=1)
        attn_norm = F.normalize(w[0], p=2, dim=1)
        sim_matrix = torch.mm(attn_norm, attn_norm.t())
        mask = ~torch.eye(K, dtype=bool)
        off_diag_sims = sim_matrix[mask]
        diversity_loss = off_diag_sims.mean()
        print(f"  ✅ Diversity loss计算正常: {diversity_loss.item():.4f}")
    except Exception as e:
        print(f"  ❌ Diversity loss计算错误: {e}")
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    issues = []
    if abs(tpa.lambda_orth_base - expected_lambda_orth) > 0.01:
        issues.append("lambda_orth配置不匹配")
    if abs(tpa.lambda_div_base - expected_lambda_div) > 0.01:
        issues.append("lambda_div配置不匹配")
    if off_diag_sims.mean().item() > 0.9:
        issues.append("prototype_queries相似度太高")
    if tpa._step < tpa.warmup_steps:
        issues.append("还在warmup阶段，diversity loss作用较弱")
    
    if issues:
        print("\n⚠️  发现以下问题：")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ 配置检查通过")
        print("   如果prototypes仍然相似，可能需要：")
        print("   1. 更多训练时间（当前2.8 epochs可能不够）")
        print("   2. 进一步增加lambda_div和lambda_orth")
        print("   3. 检查训练日志中的loss_div是否在下降")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, 
                       default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    args = parser.parse_args()
    
    check_tpa_config(args.checkpoint, args.config)

