"""
TPA Training Process Visualizer
可视化TPA在训练过程中的效果和作用

功能：
1. 训练曲线可视化（从CSV日志）
2. Prototype相似度矩阵热力图
3. Attention权重分布
4. Prototype在embedding space中的分布（t-SNE）
5. 多个checkpoint的对比分析
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TPATrainingVisualizer:
    """TPA训练过程可视化工具"""
    
    def __init__(self, output_dir: str = "tpa_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        
    def visualize_training_curves(
        self, 
        csv_path: str,
        metrics: Optional[List[str]] = None,
        save_name: str = "training_curves.png"
    ):
        """
        从训练日志CSV绘制训练曲线
        
        Args:
            csv_path: training_log.csv路径
            metrics: 要可视化的指标列表，如果为None则自动选择TPA相关指标
            save_name: 保存文件名
        """
        df = pd.read_csv(csv_path)
        
        if metrics is None:
            # 自动选择TPA相关指标
            tpa_metrics = [
                'loss_apr', 'loss_orth', 'loss_div',
                'orth_off_mse', 'diag_mse', 'usage_entropy',
                'lambda_orth', 'lambda_div',
                'orthogonality'  # 如果存在
            ]
            metrics = [m for m in tpa_metrics if m in df.columns]
        
        n_metrics = len(metrics)
        if n_metrics == 0:
            print(f"[Warning] No TPA metrics found in {csv_path}")
            return
        
        # 创建子图
        n_cols = 3
        n_rows = math.ceil(n_metrics / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            if metric in df.columns:
                ax.plot(df['iter'], df[metric], linewidth=2, label=metric)
                ax.set_xlabel('Iteration', fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(f'TPA {metric} over Training', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'{metric} not found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[TPA-Vis] Saved training curves → {save_path}")
        plt.close()
        
    def visualize_prototype_similarity(
        self,
        prototypes: torch.Tensor,
        class_indices: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        save_name: str = "prototype_similarity.png"
    ):
        """
        可视化prototype之间的相似度矩阵
        
        Args:
            prototypes: [C, K, D] tensor，C个类，每类K个prototypes，维度D
            class_indices: 要可视化的类别索引列表，如果为None则选择前几个
            class_names: 类别名称列表
            save_name: 保存文件名
        """
        P = F.normalize(prototypes, dim=-1)  # [C, K, D]
        C, K, D = P.shape
        
        if class_indices is None:
            class_indices = list(range(min(9, C)))  # 默认显示前9个类
        
        n_classes = len(class_indices)
        n_cols = 3
        n_rows = math.ceil(n_classes / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_classes > 1 else [axes]
        
        for idx, c in enumerate(class_indices):
            ax = axes[idx]
            # 计算该类内prototypes的相似度矩阵
            G = torch.einsum('kd,md->km', P[c], P[c])  # [K, K]
            G_np = G.detach().cpu().numpy()
            
            # 绘制热力图
            im = ax.imshow(G_np, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            ax.set_title(f'Class {c}' + (f': {class_names[c]}' if class_names else ''), 
                        fontsize=12)
            ax.set_xlabel('Prototype Index', fontsize=10)
            ax.set_ylabel('Prototype Index', fontsize=10)
            
            # 添加数值标注
            for i in range(K):
                for j in range(K):
                    text = ax.text(j, i, f'{G_np[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax)
        
        # 隐藏多余的子图
        for idx in range(n_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[TPA-Vis] Saved prototype similarity → {save_path}")
        plt.close()
        
    def visualize_attention_distribution(
        self,
        logits: torch.Tensor,
        class_indices: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        save_name: str = "attention_distribution.png"
    ):
        """
        可视化attention权重分布
        
        Args:
            logits: [C, K, N] tensor，attention logits
            class_indices: 要可视化的类别索引列表
            class_names: 类别名称列表
            save_name: 保存文件名
        """
        C, K, N = logits.shape
        attn = F.softmax(logits, dim=1)  # [C, K, N]
        
        if class_indices is None:
            class_indices = list(range(min(9, C)))
        
        n_classes = len(class_indices)
        n_cols = 3
        n_rows = math.ceil(n_classes / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_classes > 1 else [axes]
        
        for idx, c in enumerate(class_indices):
            ax = axes[idx]
            # 计算每个prototype的平均attention权重
            attn_mean = attn[c].mean(dim=-1).detach().cpu().numpy()  # [K]
            
            # 绘制柱状图
            bars = ax.bar(range(K), attn_mean, color='steelblue', alpha=0.7)
            ax.set_title(f'Class {c}' + (f': {class_names[c]}' if class_names else ''),
                        fontsize=12)
            ax.set_xlabel('Prototype Index', fontsize=10)
            ax.set_ylabel('Average Attention Weight', fontsize=10)
            ax.set_xticks(range(K))
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标注
            for i, (bar, val) in enumerate(zip(bars, attn_mean)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 隐藏多余的子图
        for idx in range(n_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[TPA-Vis] Saved attention distribution → {save_path}")
        plt.close()
        
    def visualize_prototype_embedding_space(
        self,
        prototypes: torch.Tensor,
        class_names: Optional[List[str]] = None,
        method: str = 'tsne',
        save_name: str = "prototype_embedding_space.png"
    ):
        """
        可视化prototypes在embedding space中的分布
        
        Args:
            prototypes: [C, K, D] tensor
            class_names: 类别名称列表
            method: 'tsne' 或 'pca'
            save_name: 保存文件名
        """
        P = F.normalize(prototypes, dim=-1)
        C, K, D = P.shape
        
        # 展平为 [C*K, D]
        P_flat = P.view(C * K, D).detach().cpu().numpy()
        
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            print("[TPA-Vis] Computing t-SNE...")
        else:
            reducer = PCA(n_components=2, random_state=42)
            print("[TPA-Vis] Computing PCA...")
        
        P_2d = reducer.fit_transform(P_flat)
        
        # 绘制
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 为每个类使用不同颜色
        colors = plt.cm.tab20(np.linspace(0, 1, C))
        
        for c in range(C):
            start_idx = c * K
            end_idx = (c + 1) * K
            label = class_names[c] if class_names and c < len(class_names) else f'Class {c}'
            ax.scatter(P_2d[start_idx:end_idx, 0], 
                      P_2d[start_idx:end_idx, 1],
                      c=[colors[c]], label=label, s=50, alpha=0.6)
        
        ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
        ax.set_title(f'Prototype Distribution in Embedding Space ({method.upper()})', 
                    fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[TPA-Vis] Saved embedding space visualization → {save_path}")
        plt.close()
        
    def visualize_prototype_evolution(
        self,
        checkpoint_paths: List[str],
        class_idx: int = 0,
        class_name: Optional[str] = None,
        save_name: str = "prototype_evolution.png"
    ):
        """
        可视化prototypes在训练过程中的演化（需要多个checkpoint）
        
        Args:
            checkpoint_paths: checkpoint路径列表，按训练顺序
            class_idx: 要可视化的类别索引
            class_name: 类别名称
            save_name: 保存文件名
        """
        # 注意：这个功能需要能够加载checkpoint并提取TPA
        # 这里提供一个框架，实际使用时需要根据模型结构调整
        print(f"[TPA-Vis] Prototype evolution visualization requires model loading")
        print(f"[TPA-Vis] This feature needs to be implemented based on your model structure")
        
    def visualize_orthogonality_heatmap(
        self,
        prototypes: torch.Tensor,
        save_name: str = "orthogonality_heatmap.png"
    ):
        """
        可视化所有prototypes的正交性矩阵
        
        Args:
            prototypes: [C, K, D] tensor
            save_name: 保存文件名
        """
        P = F.normalize(prototypes, dim=-1)
        C, K, D = P.shape
        
        # 计算所有prototypes之间的相似度
        # 展平为 [C*K, D]
        P_flat = P.view(C * K, D)
        G_all = torch.einsum('id,jd->ij', P_flat, P_flat)  # [C*K, C*K]
        G_np = G_all.detach().cpu().numpy()
        
        # 绘制大热力图
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(G_np, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # 添加网格线分隔不同类别
        for c in range(1, C):
            ax.axhline(y=c * K - 0.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
            ax.axvline(x=c * K - 0.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
        
        ax.set_title('All Prototypes Similarity Matrix', fontsize=14)
        ax.set_xlabel('Prototype Index (Class*K + Prototype)', fontsize=12)
        ax.set_ylabel('Prototype Index (Class*K + Prototype)', fontsize=12)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[TPA-Vis] Saved orthogonality heatmap → {save_path}")
        plt.close()
        
    def create_comprehensive_report(
        self,
        model,
        csv_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        prefix: str = "tpa_report"
    ):
        """
        创建综合可视化报告
        
        Args:
            model: 加载的模型（需要包含TPA）
            csv_path: 训练日志CSV路径（可选）
            class_names: 类别名称列表（可选）
            prefix: 输出文件前缀
        """
        print("[TPA-Vis] Creating comprehensive TPA visualization report...")
        
        # 提取TPA
        try:
            tpa = model.transformer.decoder.class_embed[0].tpa
        except AttributeError:
            print("[Error] Could not find TPA in model. Please check model structure.")
            return
        
        # 获取prototypes和logits
        if hasattr(tpa, '_last_prototypes') and tpa._last_prototypes is not None:
            prototypes = tpa._last_prototypes
            logits = tpa._last_logits if hasattr(tpa, '_last_logits') else None
        else:
            print("[Warning] No cached prototypes found. Running forward pass...")
            # 需要运行一次forward来生成prototypes
            # 这里假设有text_feats可用
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'text_proto_bank'):
                text_feats = model.transformer.text_proto_bank.text_feats
                prototypes, _ = model.transformer.text_proto_bank.aggregator(text_feats, with_loss=False)
                logits = model.transformer.text_proto_bank.aggregator._last_logits
            else:
                print("[Error] Cannot generate prototypes. Model structure not compatible.")
                return
        
        # 1. Prototype相似度矩阵
        self.visualize_prototype_similarity(
            prototypes, 
            class_names=class_names,
            save_name=f"{prefix}_similarity.png"
        )
        
        # 2. Attention分布
        if logits is not None:
            self.visualize_attention_distribution(
                logits,
                class_names=class_names,
                save_name=f"{prefix}_attention.png"
            )
        
        # 3. Embedding space分布
        self.visualize_prototype_embedding_space(
            prototypes,
            class_names=class_names,
            method='tsne',
            save_name=f"{prefix}_embedding_tsne.png"
        )
        
        self.visualize_prototype_embedding_space(
            prototypes,
            class_names=class_names,
            method='pca',
            save_name=f"{prefix}_embedding_pca.png"
        )
        
        # 4. 正交性热力图
        self.visualize_orthogonality_heatmap(
            prototypes,
            save_name=f"{prefix}_orthogonality.png"
        )
        
        # 5. 训练曲线（如果有CSV）
        if csv_path and Path(csv_path).exists():
            self.visualize_training_curves(
                csv_path,
                save_name=f"{prefix}_curves.png"
            )
        
        print(f"[TPA-Vis] Comprehensive report saved to {self.output_dir}/")


def load_model_from_checkpoint(checkpoint_path: str, config_path: str):
    """
    从checkpoint加载模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        config_path: 配置文件路径
    
    Returns:
        model: 加载的模型
    """
    from detectron2.config import LazyConfig, instantiate
    from detectron2.checkpoint import DetectionCheckpointer
    
    # 加载配置
    cfg = LazyConfig.load(config_path)
    
    # 创建模型
    model = instantiate(cfg.model)
    
    # 加载checkpoint
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    
    model.eval()
    return model


def load_class_names(json_path: Optional[str] = None) -> Optional[List[str]]:
    """
    从JSON文件加载类别名称
    
    Args:
        json_path: LVIS类别JSON文件路径
    
    Returns:
        类别名称列表
    """
    if json_path is None:
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item.get('name', f'class_{i}') for i, item in enumerate(data)]
            elif isinstance(data, dict) and 'categories' in data:
                return [cat.get('name', f'class_{i}') for i, cat in enumerate(data['categories'])]
    except Exception as e:
        print(f"[Warning] Could not load class names from {json_path}: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(description="TPA Training Process Visualizer")
    parser.add_argument("--mode", type=str, default="comprehensive",
                       choices=["comprehensive", "curves", "prototypes", "attention", "embedding"],
                       help="Visualization mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None,
                       help="Model config path")
    parser.add_argument("--csv", type=str, default=None,
                       help="Training log CSV path")
    parser.add_argument("--class-names", type=str, default=None,
                       help="Path to class names JSON file")
    parser.add_argument("--output-dir", type=str, default="tpa_visualizations",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    visualizer = TPATrainingVisualizer(output_dir=args.output_dir)
    
    if args.mode == "comprehensive":
        if args.checkpoint is None or args.config is None:
            print("[Error] --checkpoint and --config are required for comprehensive mode")
            return
        
        model = load_model_from_checkpoint(args.checkpoint, args.config)
        class_names = load_class_names(args.class_names)
        
        visualizer.create_comprehensive_report(
            model,
            csv_path=args.csv,
            class_names=class_names
        )
    
    elif args.mode == "curves":
        if args.csv is None:
            print("[Error] --csv is required for curves mode")
            return
        visualizer.visualize_training_curves(args.csv)
    
    else:
        print(f"[Error] Mode {args.mode} requires checkpoint and config")
        print("Please use --mode comprehensive for full visualization")


if __name__ == "__main__":
    main()

