#!/usr/bin/env python3
"""
在训练过程中监控TPA的梯度、loss_div、prototypes相似度等指标
可以作为hook添加到训练代码中，或者单独运行来检查checkpoint
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def monitor_tpa_metrics(model, iteration: int, log_interval: int = 100) -> Optional[Dict]:
    """
    监控TPA的关键指标
    
    Args:
        model: 训练中的模型
        iteration: 当前iteration
        log_interval: 每N次iteration输出一次
    
    Returns:
        包含监控指标的字典，如果不需要输出则返回None
    """
    if iteration % log_interval != 0:
        return None
    
    # 获取TPA模块
    try:
        text_classifier = model.transformer.decoder.class_embed[0]
        if not hasattr(text_classifier, 'tpa'):
            return None
        
        tpa = text_classifier.tpa
        prototype_queries = tpa.prototype_queries
        
        metrics = {
            'iter': iteration,
        }
        
        # 1. 检查_step
        current_step = tpa._step.item() if isinstance(tpa._step, torch.Tensor) else tpa._step
        metrics['step'] = current_step
        
        # 2. 检查effective lambda
        lam_orth, lam_div = tpa._effective_lambdas()
        metrics['lambda_orth'] = lam_orth
        metrics['lambda_div'] = lam_div
        
        # 3. 检查prototype_queries的梯度
        if prototype_queries.grad is not None:
            grad = prototype_queries.grad
            metrics['grad_mean'] = grad.abs().mean().item()
            metrics['grad_max'] = grad.abs().max().item()
            metrics['grad_norm'] = grad.norm().item()
        else:
            metrics['grad_mean'] = 0.0
            metrics['grad_max'] = 0.0
            metrics['grad_norm'] = 0.0
        
        # 4. 检查prototypes之间的相似度
        with torch.no_grad():
            queries_norm = F.normalize(prototype_queries.data, p=2, dim=1)
            similarity_matrix = torch.mm(queries_norm, queries_norm.t())
            mask = ~torch.eye(tpa.num_prototypes, dtype=bool, device=similarity_matrix.device)
            off_diag_similarities = similarity_matrix[mask]
            metrics['prototype_similarity_mean'] = off_diag_similarities.mean().item()
            metrics['prototype_similarity_max'] = off_diag_similarities.max().item()
        
        # 5. 检查loss值
        if hasattr(tpa, 'last_loss_terms'):
            loss_terms = tpa.last_loss_terms
            metrics['loss_apr'] = loss_terms.get('loss_apr', 0.0)
            metrics['loss_orth'] = loss_terms.get('loss_orth', 0.0)
            metrics['loss_div'] = loss_terms.get('loss_div', 0.0)
        
        return metrics
    
    except Exception as e:
        print(f"[TPA Monitor] Error: {e}")
        return None


def print_tpa_metrics(metrics: Dict):
    """打印TPA监控指标"""
    if metrics is None:
        return
    
    print(f"\n[TPA Monitor] iter={metrics['iter']:06d} step={metrics['step']}")
    print(f"  Effective lambda: orth={metrics['lambda_orth']:.4f}, div={metrics['lambda_div']:.4f}")
    print(f"  Gradient: mean={metrics['grad_mean']:.6f}, max={metrics['grad_max']:.6f}, norm={metrics['grad_norm']:.6f}")
    print(f"  Prototype similarity: mean={metrics['prototype_similarity_mean']:.4f}, max={metrics['prototype_similarity_max']:.4f}")
    print(f"  Loss: apr={metrics['loss_apr']:.4f}, orth={metrics['loss_orth']:.4f}, div={metrics['loss_div']:.4f}")


def create_tpa_monitor_hook(log_interval: int = 100):
    """
    创建一个hook函数，可以在训练代码中使用
    
    Usage:
        from examples.monitor_tpa_during_training import create_tpa_monitor_hook
        monitor_hook = create_tpa_monitor_hook(log_interval=100)
        
        # 在run_step()中，backward()之后调用
        metrics = monitor_hook(self.model, self.iter)
        if metrics:
            print_tpa_metrics(metrics)
    """
    def hook(model, iteration: int):
        return monitor_tpa_metrics(model, iteration, log_interval)
    
    return hook


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from detectron2.config import LazyConfig, instantiate
    from detectron2.checkpoint import DetectionCheckpointer
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    cfg = LazyConfig.load(args.config)
    model = instantiate(cfg.model)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.checkpoint)
    
    model.train()
    
    # 模拟一次forward和backward
    text_classifier = model.transformer.decoder.class_embed[0]
    if hasattr(text_classifier, 'train_text_feats'):
        text_feats = text_classifier.train_text_feats
        prototypes, apr_loss = text_classifier.tpa(text_feats, with_loss=True)
        apr_loss.backward()
    
    # 检查指标
    metrics = monitor_tpa_metrics(model, iteration=100, log_interval=1)
    if metrics:
        print_tpa_metrics(metrics)

