#!/usr/bin/env python3
"""
验证RPSA计算逻辑的脚本
"""
import torch
import numpy as np

def test_rpsa_computation():
    """测试RPSA计算逻辑"""
    print("="*60)
    print("RPSA Computation Verification")
    print("="*60)
    
    # 模拟数据
    B, K, D = 2, 4, 256  # batch=2, clusters=4, dim=256
    C, Kp = 80, 5  # classes=80, prototypes per class=5
    
    # 创建模拟数据
    mu = torch.randn(B, K, D)
    text_protos = torch.randn(C, Kp, D)
    pi = torch.rand(B, K, C).abs()  # 随机权重
    
    print(f"\nInput shapes:")
    print(f"  mu: {mu.shape}")
    print(f"  text_protos: {text_protos.shape}")
    print(f"  pi: {pi.shape}")
    
    # 测试相似度矩阵计算
    print(f"\n1. Testing similarity matrix computation...")
    mu_n = mu / (mu.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    P = text_protos.view(C * Kp, D)
    P = P / (P.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    
    S_flat = torch.einsum('bkd,md->bkm', mu_n, P)  # [B, K, C*Kp]
    S = S_flat.view(B, K, C, Kp)  # [B, K, C, Kp]
    
    print(f"  mu_n: {mu_n.shape}")
    print(f"  P: {P.shape}")
    print(f"  S_flat: {S_flat.shape}")
    print(f"  S: {S.shape}")
    print(f"  S range: [{S.min().item():.3f}, {S.max().item():.3f}]")
    
    # 测试InfoNCE计算
    print(f"\n2. Testing InfoNCE computation...")
    tau = 0.07
    
    pos = torch.logsumexp(S / tau, dim=-1)  # [B, K, C] over Kp
    all_ = torch.logsumexp(S.view(B, K, -1) / tau, dim=-1, keepdim=True)  # [B, K, 1]
    
    print(f"  pos: {pos.shape}, range: [{pos.min().item():.3f}, {pos.max().item():.3f}]")
    print(f"  all_: {all_.shape}, range: [{all_.min().item():.3f}, {all_.max().item():.3f}]")
    print(f"  pos - all_: {pos.shape}, range: [{(pos - all_).min().item():.3f}, {(pos - all_).max().item():.3f}]")
    
    # 测试权重计算
    print(f"\n3. Testing weight computation...")
    pi_clamped = pi.clamp_min(0.0)
    pi_max_raw = pi_clamped.max(dim=-1).values
    bg_mask = (pi_max_raw < 0.1)
    pi_tilde = (pi_clamped ** 1.0)
    pi_tilde = pi_tilde / (pi_tilde.sum(dim=-1, keepdim=True).clamp_min(1e-6))
    
    print(f"  pi_max_raw: {pi_max_raw.shape}, range: [{pi_max_raw.min().item():.3f}, {pi_max_raw.max().item():.3f}]")
    print(f"  bg_mask: {bg_mask.shape}, background clusters: {bg_mask.sum().item()}/{bg_mask.numel()}")
    print(f"  pi_tilde: {pi_tilde.shape}, sum per cluster: {pi_tilde.sum(dim=-1)}")
    
    # 测试损失计算
    print(f"\n4. Testing loss computation...")
    loss_k = -(pi_tilde * (pos - all_)).sum(dim=-1)  # [B, K]
    loss_k = loss_k.masked_fill(bg_mask, 0.0)
    
    valid = (~bg_mask).float().sum().clamp_min(1.0)
    loss = loss_k.sum() / valid
    
    print(f"  loss_k: {loss_k.shape}, range: [{loss_k.min().item():.6f}, {loss_k.max().item():.6f}]")
    print(f"  valid clusters: {valid.item()}")
    print(f"  final loss: {loss.item():.6f}")
    
    # 验证数值稳定性
    print(f"\n5. Testing numerical stability...")
    print(f"  pos contains NaN: {torch.isnan(pos).any().item()}")
    print(f"  all_ contains NaN: {torch.isnan(all_).any().item()}")
    print(f"  loss_k contains NaN: {torch.isnan(loss_k).any().item()}")
    print(f"  loss is NaN: {torch.isnan(loss).item()}")
    print(f"  loss is Inf: {torch.isinf(loss).item()}")
    
    # 验证InfoNCE数学正确性
    print(f"\n6. Verifying InfoNCE mathematical correctness...")
    # 对于单个样本，验证 log(exp(pos)/exp(all_)) = pos - all_
    test_pos = pos[0, 0, 0].item()
    test_all = all_[0, 0, 0].item()
    ratio = test_pos - test_all
    print(f"  For cluster 0, class 0:")
    print(f"    pos = {test_pos:.6f}")
    print(f"    all_ = {test_all:.6f}")
    print(f"    pos - all_ = {ratio:.6f}")
    print(f"    This represents log(P(positive) / P(all))")
    
    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)

if __name__ == "__main__":
    test_rpsa_computation()

