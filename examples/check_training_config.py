#!/usr/bin/env python3
"""
检查训练配置文件是否正确
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detectron2.config import LazyConfig

def check_config(config_path: str):
    """检查配置文件"""
    print("=" * 70)
    print("训练配置文件检查")
    print("=" * 70)
    
    cfg = LazyConfig.load(config_path)
    
    print("\n【TPA配置检查】")
    print("-" * 70)
    
    classifier = cfg.model.classifier
    
    # 检查TPA是否启用
    use_tpa = getattr(classifier, 'use_tpa', False)
    print(f"  use_tpa: {use_tpa}")
    if not use_tpa:
        print("  ❌ TPA未启用！")
        return
    else:
        print("  ✅ TPA已启用")
    
    # 检查关键参数
    checks = [
        ("tpa_lambda_orth", 0.20, "Orthogonality loss weight"),
        ("tpa_lambda_div", 0.12, "Diversity loss weight"),
        ("tpa_tau", 0.10, "Attention temperature"),
        ("tpa_warmup_steps", 4615, "Warmup steps"),
        ("tpa_num_prototypes", 5, "Number of prototypes"),
    ]
    
    all_ok = True
    for param_name, expected_value, description in checks:
        value = getattr(classifier, param_name, None)
        if value is None:
            print(f"  ❌ {param_name}: 未设置")
            all_ok = False
        elif abs(value - expected_value) < 0.01:
            print(f"  ✅ {param_name}: {value} ({description})")
        else:
            print(f"  ⚠️  {param_name}: {value} (期望: {expected_value})")
            all_ok = False
    
    print("\n【训练配置检查】")
    print("-" * 70)
    
    train = cfg.train
    print(f"  max_iter: {train.max_iter}")
    print(f"  batch_size: {cfg.dataloader.train.total_batch_size}")
    print(f"  output_dir: {train.output_dir}")
    
    # 计算warmup比例
    warmup_steps = getattr(classifier, 'tpa_warmup_steps', 0)
    warmup_ratio = warmup_steps / train.max_iter if train.max_iter > 0 else 0
    print(f"  warmup_ratio: {warmup_ratio:.2%} ({warmup_steps}/{train.max_iter})")
    
    if warmup_ratio < 0.04 or warmup_ratio > 0.06:
        print("  ⚠️  Warmup比例不在5%左右")
    else:
        print("  ✅ Warmup比例正常")
    
    print("\n【代码检查】")
    print("-" * 70)
    
    # 检查代码是否已修复
    try:
        from lami_dino.models.text_prototype_aggregator import TextPrototypeAggregator
        import inspect
        
        # 检查_step是否注册为buffer
        source = inspect.getsource(TextPrototypeAggregator.__init__)
        if "register_buffer('_step'" in source or 'register_buffer("_step"' in source:
            print("  ✅ _step已注册为buffer（代码已修复）")
        else:
            print("  ❌ _step未注册为buffer（需要修复）")
            all_ok = False
        
        # 检查diversity loss实现
        source = inspect.getsource(TextPrototypeAggregator._diversity_term)
        if "similarity_matrix" in source and "off_diag" in source:
            print("  ✅ Diversity loss实现已修复（使用相似度矩阵）")
        else:
            print("  ⚠️  Diversity loss实现可能未修复")
        
    except Exception as e:
        print(f"  ⚠️  无法检查代码: {e}")
    
    print("\n【总结】")
    print("=" * 70)
    
    if all_ok:
        print("✅ 配置检查通过！可以开始训练")
        print("\n【训练建议】")
        print("  1. 训练时观察loss_div是否在下降")
        print("  2. 在1.5 epochs (iter 9,230)检查prototype多样性")
        print("  3. 在2.2 epochs (iter 13,845)再次检查")
    else:
        print("⚠️  发现配置问题，请修复后再训练")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    check_config(args.config)

