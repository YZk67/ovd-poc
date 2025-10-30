#!/usr/bin/env python3
"""
诊断脚本：检查RPSA模块的状态和配置
"""
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_rpsa_status():
    """检查RPSA模块的状态"""
    print("="*60)
    print("RPSA Status Diagnostic Script")
    print("="*60)
    
    try:
        # 1. 检查配置文件
        print("\n1. Checking configuration file...")
        from detectron2.config import LazyConfig
        cfg = LazyConfig.load("lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
        
        print(f"   model.transformer.use_rpsa: {cfg.model.transformer.use_rpsa}")
        if hasattr(cfg.model.transformer, 'rpsa_module'):
            print(f"   model.transformer.rpsa_module exists: True")
            rpsa_cfg = cfg.model.transformer.rpsa_module
            if hasattr(rpsa_cfg, '_target_'):
                print(f"   rpsa_module type: {rpsa_cfg._target_}")
            if hasattr(rpsa_cfg, 'K'):
                print(f"   rpsa_module.K: {rpsa_cfg.K}")
        else:
            print(f"   model.transformer.rpsa_module: NOT FOUND")
        
        if hasattr(cfg.model.criterion, 'weight_dict'):
            wd = cfg.model.criterion.weight_dict
            if 'loss_rpsa' in wd:
                print(f"   criterion.weight_dict['loss_rpsa']: {wd['loss_rpsa']}")
            else:
                print(f"   criterion.weight_dict['loss_rpsa']: NOT FOUND")
        
        # 2. 尝试实例化模型（仅初始化部分）
        print("\n2. Checking model initialization...")
        from detectron2.config import instantiate
        
        # 检查transformer配置
        transformer_cfg = cfg.model.transformer
        print(f"   use_rpsa in config: {transformer_cfg.use_rpsa}")
        print(f"   rpsa_module in config: {hasattr(transformer_cfg, 'rpsa_module')}")
        
        if hasattr(transformer_cfg, 'rpsa_module'):
            rpsa_cfg = transformer_cfg.rpsa_module
            print(f"   rpsa_module config type: {type(rpsa_cfg)}")
            if hasattr(rpsa_cfg, '_target_'):
                print(f"   rpsa_module._target_: {rpsa_cfg._target_}")
        
        # 3. 检查实际模型（如果可能）
        print("\n3. Checking actual model (if available)...")
        print("   Note: This requires a trained model checkpoint")
        
    except Exception as e:
        print(f"\nERROR during configuration check: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Diagnostic complete")
    print("="*60)

if __name__ == "__main__":
    check_rpsa_status()

