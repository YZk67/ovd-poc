#!/usr/bin/env python3
"""
诊断工具：检查TPA prototypes是否学到了不同的内容

主要检查：
1. Attention权重分布 - 不同prototype是否关注不同的prompts
2. Prototype相似度 - prototypes之间是否足够不同
3. Prompt使用情况 - 每个prompt被哪些prototype使用
4. Diversity指标 - 定量评估多样性
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

from examples.tpa_visualization_classes import SELECTED_CLASSES, CLASS_INDICES

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class PrototypeDiversityDiagnostic:
    """诊断TPA prototypes的多样性"""
    
    def __init__(self, output_dir: str = "prototype_diversity_diagnosis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_and_extract_data(
        self,
        checkpoint_path: str,
        config_path: str,
        text_embed_path: Optional[str] = None,
    ) -> Dict:
        """加载模型并提取TPA数据"""
        print("[Info] Loading model...")
        cfg = LazyConfig.load(config_path)
        model = instantiate(cfg.model)
        
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(checkpoint_path)
        model.eval()
        
        # 获取TPA
        text_classifier = model.transformer.decoder.class_embed[0]
        if not hasattr(text_classifier, 'tpa') or not text_classifier.use_tpa:
            raise ValueError("Model does not use TPA.")
        
        tpa = text_classifier.tpa
        
        # 获取text embeddings
        if text_embed_path and Path(text_embed_path).exists():
            print(f"[Info] Loading text embeddings from: {text_embed_path}")
            text_feats_raw = np.load(text_embed_path)
            if text_feats_raw.ndim == 2:
                text_feats_raw = text_feats_raw[:, None, :]
            text_feats = torch.from_numpy(text_feats_raw).to(dtype=torch.float32)
        else:
            text_feats = text_classifier.train_text_feats
        
        device = next(model.parameters()).device
        if text_feats.device != device:
            text_feats = text_feats.to(device)
        
        # Forward pass获取prototypes和attention
        with torch.no_grad():
            prototypes, _ = tpa(text_feats, with_loss=False)
            # 获取attention权重
            if hasattr(tpa, '_last_logits'):
                logits = tpa._last_logits  # [C, K, N]
                attention_weights = F.softmax(logits / tpa.tau, dim=-1)  # [C, K, N]
            else:
                # 重新计算
                keys = tpa.key_proj(text_feats)
                logits = torch.einsum("kh,cnh->ckn", tpa.prototype_queries, keys)
                logits = logits / np.sqrt(tpa.key_proj.out_features)
                attention_weights = F.softmax(logits / tpa.tau, dim=-1)
        
        return {
            'prototypes': prototypes.detach().cpu(),  # [C, K, D]
            'attention_weights': attention_weights.detach().cpu(),  # [C, K, N]
            'original_prompts': text_feats.detach().cpu(),  # [C, N, D]
            'class_names': SELECTED_CLASSES,
        }
    
    def compute_diversity_metrics(
        self,
        attention_weights: torch.Tensor,  # [C, K, N]
        prototypes: torch.Tensor,  # [C, K, D]
    ) -> Dict[str, float]:
        """计算多样性指标"""
        C, K, N = attention_weights.shape
        
        metrics = {}
        
        # 1. Attention Pattern Similarity (不同prototype的attention分布相似度)
        attention_similarities = []
        for c in range(C):
            attn_c = attention_weights[c]  # [K, N]
            # 计算prototype之间的attention分布相似度
            attn_norm = F.normalize(attn_c, p=2, dim=1)  # [K, N]
            sim_matrix = torch.mm(attn_norm, attn_norm.t())  # [K, K]
            # 只取上三角（不包括对角线）
            mask = torch.triu(torch.ones(K, K), diagonal=1).bool()
            off_diag_sims = sim_matrix[mask]
            attention_similarities.append(off_diag_sims.mean().item())
        
        metrics['attention_pattern_similarity'] = np.mean(attention_similarities)
        metrics['attention_pattern_similarity_std'] = np.std(attention_similarities)
        
        # 2. Prototype Similarity (prototype embedding之间的相似度)
        prototype_similarities = []
        for c in range(C):
            proto_c = prototypes[c]  # [K, D]
            proto_norm = F.normalize(proto_c, p=2, dim=1)  # [K, D]
            sim_matrix = torch.mm(proto_norm, proto_norm.t())  # [K, K]
            mask = torch.triu(torch.ones(K, K), diagonal=1).bool()
            off_diag_sims = sim_matrix[mask]
            prototype_similarities.append(off_diag_sims.mean().item())
        
        metrics['prototype_similarity'] = np.mean(prototype_similarities)
        metrics['prototype_similarity_std'] = np.std(prototype_similarities)
        
        # 3. Attention Entropy (每个prototype的attention分布熵)
        entropies = []
        for c in range(C):
            for k in range(K):
                attn_dist = attention_weights[c, k]  # [N]
                # 避免log(0)
                attn_dist = attn_dist + 1e-8
                attn_dist = attn_dist / attn_dist.sum()
                entropy = -(attn_dist * torch.log(attn_dist)).sum().item()
                entropies.append(entropy)
        
        metrics['attention_entropy_mean'] = np.mean(entropies)
        metrics['attention_entropy_std'] = np.std(entropies)
        metrics['attention_entropy_max'] = np.log(N)  # 最大熵（均匀分布）
        
        # 4. Unique Top Prompts (每个prototype的top-1 prompt是否不同)
        unique_top_prompts = []
        for c in range(C):
            attn_c = attention_weights[c]  # [K, N]
            top_prompts = attn_c.argmax(dim=1).tolist()  # [K]
            unique_count = len(set(top_prompts))
            unique_top_prompts.append(unique_count)
        
        metrics['unique_top_prompts_mean'] = np.mean(unique_top_prompts)
        metrics['unique_top_prompts_ratio'] = np.mean(unique_top_prompts) / K
        
        return metrics
    
    def visualize_attention_patterns(
        self,
        attention_weights: torch.Tensor,  # [C, K, N]
        class_names: List[str],
        save_name: str = "attention_patterns.png"
    ):
        """可视化attention模式 - 检查不同prototype是否关注不同prompts"""
        C, K, N = attention_weights.shape
        
        # 选择几个代表性类别
        selected_indices = [SELECTED_CLASSES.index(name) for name in class_names 
                          if name in SELECTED_CLASSES][:5]
        
        n_classes = len(selected_indices)
        fig, axes = plt.subplots(n_classes, 1, figsize=(12, 3 * n_classes))
        if n_classes == 1:
            axes = [axes]
        
        for idx, class_idx in enumerate(selected_indices):
            ax = axes[idx]
            attn = attention_weights[class_idx].numpy()  # [K, N]
            
            # 绘制heatmap
            im = ax.imshow(attn, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
            
            ax.set_title(f'{class_names[class_idx]} - Attention Patterns', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Prompt Index', fontsize=10)
            ax.set_ylabel('Prototype Index', fontsize=10)
            ax.set_yticks(range(K))
            ax.set_yticklabels([f'Proto{i+1}' for i in range(K)])
            ax.set_xticks(range(N))
            ax.set_xticklabels([f'P{i+1}' for i in range(N)])
            
            plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Diagnostic] Saved attention patterns → {save_path}")
        plt.close()
    
    def visualize_prototype_similarity(
        self,
        prototypes: torch.Tensor,  # [C, K, D]
        class_names: List[str],
        save_name: str = "prototype_similarity.png"
    ):
        """可视化prototype之间的相似度矩阵"""
        C, K, D = prototypes.shape
        
        # 选择几个代表性类别
        selected_indices = [SELECTED_CLASSES.index(name) for name in class_names 
                          if name in SELECTED_CLASSES][:5]
        
        n_classes = len(selected_indices)
        fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
        if n_classes == 1:
            axes = [axes]
        
        for idx, class_idx in enumerate(selected_indices):
            ax = axes[idx]
            proto = prototypes[class_idx]  # [K, D]
            proto_norm = F.normalize(proto, p=2, dim=1)
            sim_matrix = torch.mm(proto_norm, proto_norm.t()).numpy()  # [K, K]
            
            im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(f'{class_names[class_idx]}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Prototype Index', fontsize=10)
            ax.set_ylabel('Prototype Index', fontsize=10)
            ax.set_xticks(range(K))
            ax.set_yticks(range(K))
            ax.set_xticklabels([f'P{i+1}' for i in range(K)])
            ax.set_yticklabels([f'P{i+1}' for i in range(K)])
            
            # 添加数值标注
            for i in range(K):
                for j in range(K):
                    text = ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                                 ha="center", va="center",
                                 color="white" if abs(sim_matrix[i, j]) > 0.5 else "black",
                                 fontsize=8)
            
            plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Diagnostic] Saved prototype similarity → {save_path}")
        plt.close()
    
    def visualize_top_prompts_per_prototype(
        self,
        attention_weights: torch.Tensor,  # [C, K, N]
        class_names: List[str],
        top_k: int = 3,
        save_name: str = "top_prompts_per_prototype.png"
    ):
        """可视化每个prototype的top-k prompts"""
        C, K, N = attention_weights.shape
        
        # 选择几个代表性类别
        selected_indices = [SELECTED_CLASSES.index(name) for name in class_names 
                          if name in SELECTED_CLASSES][:5]
        
        fig, axes = plt.subplots(len(selected_indices), 1, 
                                figsize=(10, 3 * len(selected_indices)))
        if len(selected_indices) == 1:
            axes = [axes]
        
        for idx, class_idx in enumerate(selected_indices):
            ax = axes[idx]
            attn = attention_weights[class_idx]  # [K, N]
            
            # 对每个prototype，找到top-k prompts
            top_prompts_data = []
            for k in range(K):
                top_k_vals, top_k_indices = attn[k].topk(top_k)
                for i, (val, prompt_idx) in enumerate(zip(top_k_vals, top_k_indices)):
                    top_prompts_data.append({
                        'prototype': k,
                        'prompt': prompt_idx.item(),
                        'attention': val.item(),
                        'rank': i
                    })
            
            # 绘制bar chart
            x_pos = np.arange(K)
            width = 0.8 / top_k
            
            for rank in range(top_k):
                data_for_rank = [d for d in top_prompts_data if d['rank'] == rank]
                prompts = [d['prompt'] for d in data_for_rank]
                attentions = [d['attention'] for d in data_for_rank]
                
                ax.bar(x_pos + rank * width, attentions, width,
                      label=f'Top-{rank+1} Prompt', alpha=0.8)
            
            ax.set_xlabel('Prototype Index', fontsize=10)
            ax.set_ylabel('Attention Weight', fontsize=10)
            ax.set_title(f'{class_names[class_idx]} - Top-{top_k} Prompts per Prototype',
                         fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos + width * (top_k - 1) / 2)
            ax.set_xticklabels([f'Proto{i+1}' for i in range(K)])
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Diagnostic] Saved top prompts per prototype → {save_path}")
        plt.close()
    
    def print_diversity_report(
        self,
        metrics: Dict[str, float],
        attention_weights: torch.Tensor,
    ):
        """打印多样性诊断报告"""
        print("\n" + "=" * 70)
        print("Prototype Diversity Diagnostic Report")
        print("=" * 70)
        
        print(f"\n【Attention Pattern Similarity】")
        print(f"  平均相似度: {metrics['attention_pattern_similarity']:.4f}")
        print(f"  标准差: {metrics['attention_pattern_similarity_std']:.4f}")
        if metrics['attention_pattern_similarity'] > 0.8:
            print(f"  ⚠️  警告: 相似度太高！不同prototype的attention模式几乎相同")
            print(f"      → 说明diversity loss可能没有生效")
        elif metrics['attention_pattern_similarity'] > 0.6:
            print(f"  ⚠️  注意: 相似度较高，可能需要增强diversity loss")
        else:
            print(f"  ✅ 良好: 不同prototype学到了不同的attention模式")
        
        print(f"\n【Prototype Similarity】")
        print(f"  平均相似度: {metrics['prototype_similarity']:.4f}")
        print(f"  标准差: {metrics['prototype_similarity_std']:.4f}")
        if metrics['prototype_similarity'] > 0.5:
            print(f"  ⚠️  警告: prototype embedding相似度太高！")
            print(f"      → 说明orthogonality loss可能不够强")
        elif metrics['prototype_similarity'] > 0.3:
            print(f"  ⚠️  注意: 相似度较高，可能需要增强orthogonality loss")
        else:
            print(f"  ✅ 良好: prototypes在embedding空间中足够不同")
        
        print(f"\n【Attention Entropy】")
        print(f"  平均熵: {metrics['attention_entropy_mean']:.4f}")
        print(f"  最大熵: {metrics['attention_entropy_max']:.4f}")
        entropy_ratio = metrics['attention_entropy_mean'] / metrics['attention_entropy_max']
        print(f"  熵比率: {entropy_ratio:.4f} (1.0 = 完全均匀分布)")
        if entropy_ratio < 0.5:
            print(f"  ⚠️  警告: 熵太低！每个prototype只关注少数prompts")
        elif entropy_ratio < 0.7:
            print(f"  ⚠️  注意: 熵较低，attention分布较集中")
        else:
            print(f"  ✅ 良好: attention分布较均匀")
        
        print(f"\n【Unique Top Prompts】")
        print(f"  平均唯一top prompts数: {metrics['unique_top_prompts_mean']:.2f} / {attention_weights.shape[1]}")
        print(f"  唯一性比率: {metrics['unique_top_prompts_ratio']:.4f}")
        if metrics['unique_top_prompts_ratio'] < 0.6:
            print(f"  ⚠️  警告: 多个prototype关注相同的top prompt！")
            print(f"      → 说明prototypes没有学到不同的关注点")
        elif metrics['unique_top_prompts_ratio'] < 0.8:
            print(f"  ⚠️  注意: 唯一性较低，部分prototype关注相同prompts")
        else:
            print(f"  ✅ 良好: 每个prototype关注不同的prompts")
        
        print("\n" + "=" * 70)
        print("诊断建议：")
        print("=" * 70)
        
        issues = []
        if metrics['attention_pattern_similarity'] > 0.6:
            issues.append("1. 增加 lambda_div (diversity loss权重)")
        if metrics['prototype_similarity'] > 0.3:
            issues.append("2. 增加 lambda_orth (orthogonality loss权重)")
        if metrics['unique_top_prompts_ratio'] < 0.8:
            issues.append("3. 检查diversity loss实现是否正确")
        
        if issues:
            print("\n⚠️  发现以下问题，建议修复：")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ 所有指标正常，TPA工作良好！")
        
        print("\n" + "=" * 70)
    
    def run_full_diagnosis(
        self,
        checkpoint_path: str,
        config_path: str,
        text_embed_path: Optional[str] = None,
    ):
        """运行完整诊断"""
        print("=" * 70)
        print("TPA Prototype Diversity Diagnostic")
        print("=" * 70)
        
        # 加载数据
        data = self.load_model_and_extract_data(
            checkpoint_path, config_path, text_embed_path
        )
        
        # 计算指标
        print("\n[1/4] Computing diversity metrics...")
        metrics = self.compute_diversity_metrics(
            data['attention_weights'],
            data['prototypes']
        )
        
        # 打印报告
        print("\n[2/4] Generating diagnostic report...")
        self.print_diversity_report(metrics, data['attention_weights'])
        
        # 可视化
        print("\n[3/4] Creating visualizations...")
        self.visualize_attention_patterns(
            data['attention_weights'],
            data['class_names'],
            save_name="attention_patterns.png"
        )
        
        self.visualize_prototype_similarity(
            data['prototypes'],
            data['class_names'],
            save_name="prototype_similarity.png"
        )
        
        self.visualize_top_prompts_per_prototype(
            data['attention_weights'],
            data['class_names'],
            top_k=3,
            save_name="top_prompts_per_prototype.png"
        )
        
        # 保存指标
        print("\n[4/4] Saving metrics...")
        import json
        metrics_save = {k: float(v) for k, v in metrics.items()}
        with open(self.output_dir / "diversity_metrics.json", 'w') as f:
            json.dump(metrics_save, f, indent=2)
        
        print(f"\n✅ All results saved to: {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Diagnose TPA Prototype Diversity")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py",
        help="Path to model config file"
    )
    parser.add_argument(
        "--text-embed",
        type=str,
        default=None,
        help="Path to text embeddings npy file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prototype_diversity_diagnosis",
        help="Output directory for diagnosis results"
    )
    
    args = parser.parse_args()
    
    diagnostic = PrototypeDiversityDiagnostic(output_dir=args.output_dir)
    diagnostic.run_full_diagnosis(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        text_embed_path=args.text_embed,
    )


if __name__ == "__main__":
    main()

