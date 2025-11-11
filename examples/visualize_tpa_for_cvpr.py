#!/usr/bin/env python3
"""
TPA可视化工具 - 专门用于CVPR论文
针对选定的10个代表性类别进行可视化

主要功能：
1. Prototype Attention热力图 - 展示TPA如何选择/融合原始prompts
2. Embedding Space可视化 - 展示prototypes vs 原始prompts的位置
3. Prototypes vs 简单平均的对比
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from examples.tpa_visualization_classes import (
    SELECTED_CLASSES,
    CLASS_INDICES,
    VISUALIZATION_SUGGESTIONS
)

# 设置matplotlib参数
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class TPAVisualizerForCVPR:
    """专门用于CVPR论文的TPA可视化工具"""
    
    def __init__(self, output_dir: str = "tpa_cvpr_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_and_extract_data(
        self,
        checkpoint_path: str,
        config_path: str,
        prompts_json_path: str,
        text_embed_path: Optional[str] = None,
    ) -> Dict:
        """
        加载模型并提取TPA相关数据
        
        Returns:
            dict包含:
                - prototypes: [C, K, D] TPA生成的prototypes
                - original_prompts: [C, N, D] 原始prompts embeddings
                - attention_weights: [C, K, N] attention权重
                - simple_mean: [C, D] 简单平均的embeddings
                - class_names: 类别名称列表
                - selected_indices: 选定的10个类别的索引
        """
        from detectron2.config import LazyConfig, instantiate
        from detectron2.checkpoint import DetectionCheckpointer
        
        print("[Info] Loading model...")
        cfg = LazyConfig.load(config_path)
        model = instantiate(cfg.model)
        
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(checkpoint_path)
        model.eval()
        
        # 获取TPA - 从class_embed获取
        text_classifier = model.transformer.decoder.class_embed[0]
        if not hasattr(text_classifier, 'tpa') or not text_classifier.use_tpa:
            raise ValueError("Model does not use TPA. Please check if use_tpa=True in config.")
        
        tpa = text_classifier.tpa
        
        # 获取原始text embeddings
        # 优先使用指定的npy文件，否则使用模型已加载的
        if text_embed_path and Path(text_embed_path).exists():
            print(f"[Info] Loading text embeddings from: {text_embed_path}")
            text_feats_raw = np.load(text_embed_path)
            if text_feats_raw.ndim == 2:
                text_feats_raw = text_feats_raw[:, None, :]  # [C, 1, D] -> [C, N, D]
            text_feats = torch.from_numpy(text_feats_raw).to(dtype=torch.float32)
            print(f"[Info] Loaded text embeddings shape: {text_feats.shape}")
        else:
            # 从模型获取（模型初始化时已从npy文件加载）
            text_feats = text_classifier.train_text_feats  # [C, N, D]
            if text_embed_path:
                print(f"[Warning] Text embed path {text_embed_path} not found, using model's loaded embeddings")
            else:
                print(f"[Info] Using text embeddings from model (loaded from config)")
        
        # 确保text_feats在正确的设备上
        device = next(model.parameters()).device
        if text_feats.device != device:
            text_feats = text_feats.to(device)
        
        print(f"[Info] Text features shape: {text_feats.shape}")
        
        # 运行TPA获取prototypes和attention
        print("[Info] Running TPA forward pass...")
        with torch.no_grad():
            prototypes, _ = tpa(text_feats, with_loss=False)
            attention_logits = tpa._last_logits  # [C, K, N]
            attention_weights = F.softmax(attention_logits / tpa.tau, dim=-1)  # [C, K, N]
        
        print(f"[Info] Prototypes shape: {prototypes.shape}")
        print(f"[Info] Attention weights shape: {attention_weights.shape}")
        
        # 计算简单平均
        simple_mean = text_feats.mean(dim=1)  # [C, D]
        
        # 加载类别名称和prompts文本
        with open(prompts_json_path, 'r') as f:
            prompts_dict = json.load(f)
        
        # 获取选定的类别索引
        all_class_names = list(prompts_dict.keys())
        selected_indices = []
        selected_class_names = []
        selected_prompts = []
        
        for cls_name in SELECTED_CLASSES:
            if cls_name in all_class_names:
                idx = all_class_names.index(cls_name)
                selected_indices.append(idx)
                selected_class_names.append(cls_name)
                selected_prompts.append(prompts_dict[cls_name])
            else:
                print(f"[Warning] Class {cls_name} not found in prompts JSON")
        
        print(f"[Info] Selected {len(selected_indices)} classes for visualization")
        
        # 提取选定类别的数据
        selected_prototypes = prototypes[selected_indices]  # [10, K, D]
        selected_original = text_feats[selected_indices]  # [10, N, D]
        selected_attention = attention_weights[selected_indices]  # [10, K, N]
        selected_simple_mean = simple_mean[selected_indices]  # [10, D]
        
        return {
            'prototypes': selected_prototypes,
            'original_prompts': selected_original,
            'attention_weights': selected_attention,
            'simple_mean': selected_simple_mean,
            'class_names': selected_class_names,
            'prompts_text': selected_prompts,
            'selected_indices': selected_indices,
        }
    
    def visualize_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        class_names: List[str],
        prompts_text: List[List[str]],
        save_name: str = "attention_heatmap.png"
    ):
        """
        可视化Prototype Attention热力图
        
        Args:
            attention_weights: [C, K, N] attention权重
            class_names: 类别名称列表
            prompts_text: 每个类别的prompts文本列表
        """
        C, K, N = attention_weights.shape
        
        # 选择5个代表性类别
        selected_classes = VISUALIZATION_SUGGESTIONS["attention_heatmap"]
        selected_indices = [SELECTED_CLASSES.index(c) for c in selected_classes if c in SELECTED_CLASSES]
        
        if len(selected_indices) == 0:
            selected_indices = list(range(min(5, C)))
        
        n_classes = len(selected_indices)
        fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 5))
        
        if n_classes == 1:
            axes = [axes]
        
        for idx, class_idx in enumerate(selected_indices):
            ax = axes[idx]
            cls_name = class_names[class_idx]
            attn = attention_weights[class_idx].detach().cpu().numpy()  # [K, N]
            
            # 绘制热力图
            im = ax.imshow(attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'{cls_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Original Prompt Index', fontsize=10)
            ax.set_ylabel('Prototype Index', fontsize=10)
            ax.set_xticks(range(N))
            ax.set_yticks(range(K))
            ax.set_xticklabels([f'P{i+1}' for i in range(N)], fontsize=8)
            ax.set_yticklabels([f'Proto{i+1}' for i in range(K)], fontsize=8)
            
            # 添加数值标注（只显示较大的值）
            for i in range(K):
                for j in range(N):
                    val = attn[i, j]
                    if val > 0.1:  # 只显示大于0.1的值
                        text = ax.text(j, i, f'{val:.2f}',
                                     ha="center", va="center",
                                     color="white" if val > 0.5 else "black",
                                     fontsize=7, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[TPA-Vis] Saved attention heatmap → {save_path}")
        plt.close()
    
    def visualize_embedding_space(
        self,
        prototypes: torch.Tensor,
        original_prompts: torch.Tensor,
        simple_mean: torch.Tensor,
        class_names: List[str],
        method: str = 'tsne',
        save_name: str = "embedding_space.png"
    ):
        """
        可视化embedding space中的分布
        
        Args:
            prototypes: [C, K, D] TPA prototypes
            original_prompts: [C, N, D] 原始prompts
            simple_mean: [C, D] 简单平均
            class_names: 类别名称列表
            method: 'tsne' 或 'pca'
        """
        C, K, D = prototypes.shape
        _, N, _ = original_prompts.shape
        
        # 选择相似类别对进行可视化
        similar_pairs = VISUALIZATION_SUGGESTIONS["embedding_space"]
        
        n_pairs = len(similar_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, 6))
        
        if n_pairs == 1:
            axes = [axes]
        
        for pair_idx, (cls1, cls2) in enumerate(similar_pairs):
            if cls1 not in SELECTED_CLASSES or cls2 not in SELECTED_CLASSES:
                continue
            
            ax = axes[pair_idx]
            idx1 = SELECTED_CLASSES.index(cls1)
            idx2 = SELECTED_CLASSES.index(cls2)
            
            # 准备数据：每个类别的prototypes, prompts, 和simple mean
            data_list = []
            labels_list = []
            colors_list = []
            
            for idx, cls_name in [(idx1, cls1), (idx2, cls2)]:
                # Prototypes
                proto_data = prototypes[idx].detach().cpu().numpy()  # [K, D]
                data_list.append(proto_data)
                labels_list.extend([f'{cls_name}_Proto{i+1}' for i in range(K)])
                colors_list.extend([f'{cls_name}_proto'] * K)
                
                # Original prompts
                prompt_data = original_prompts[idx].detach().cpu().numpy()  # [N, D]
                data_list.append(prompt_data)
                labels_list.extend([f'{cls_name}_Prompt{i+1}' for i in range(N)])
                colors_list.extend([f'{cls_name}_prompt'] * N)
                
                # Simple mean
                mean_data = simple_mean[idx].detach().cpu().numpy().reshape(1, -1)  # [1, D]
                data_list.append(mean_data)
                labels_list.append(f'{cls_name}_Mean')
                colors_list.append(f'{cls_name}_mean')
            
            # 合并所有数据
            all_data = np.vstack(data_list)  # [total_points, D]
            
            # 降维
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data)-1))
            else:
                reducer = PCA(n_components=2, random_state=42)
            
            print(f"[Info] Computing {method.upper()} for {cls1} vs {cls2}...")
            data_2d = reducer.fit_transform(all_data)
            
            # 绘制
            color_map = {
                f'{cls1}_proto': 'red',
                f'{cls1}_prompt': 'lightcoral',
                f'{cls1}_mean': 'darkred',
                f'{cls2}_proto': 'blue',
                f'{cls2}_prompt': 'lightblue',
                f'{cls2}_mean': 'darkblue',
            }
            
            marker_map = {
                f'{cls1}_proto': 'o',
                f'{cls1}_prompt': 's',
                f'{cls1}_mean': '^',
                f'{cls2}_proto': 'o',
                f'{cls2}_prompt': 's',
                f'{cls2}_mean': '^',
            }
            
            size_map = {
                f'{cls1}_proto': 100,
                f'{cls1}_prompt': 50,
                f'{cls1}_mean': 200,
                f'{cls2}_proto': 100,
                f'{cls2}_prompt': 50,
                f'{cls2}_mean': 200,
            }
            
            # 分别绘制每个类别
            for cls_name in [cls1, cls2]:
                for data_type in ['proto', 'prompt', 'mean']:
                    key = f'{cls_name}_{data_type}'
                    mask = np.array([c == key for c in colors_list])
                    if mask.sum() > 0:
                        ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                                 c=color_map[key], marker=marker_map[key],
                                 s=size_map[key], alpha=0.7, label=key.replace('_', ' '))
            
            ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=11)
            ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=11)
            ax.set_title(f'{cls1} vs {cls2}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[TPA-Vis] Saved embedding space visualization → {save_path}")
        plt.close()
    
    def visualize_prototype_vs_mean_comparison(
        self,
        prototypes: torch.Tensor,
        simple_mean: torch.Tensor,
        class_names: List[str],
        save_name: str = "prototype_vs_mean.png"
    ):
        """
        可视化prototypes与简单平均的相似度对比
        """
        C, K, D = prototypes.shape
        
        # 计算每个prototype与simple mean的相似度
        prototypes_norm = F.normalize(prototypes, dim=-1)  # [C, K, D]
        simple_mean_norm = F.normalize(simple_mean, dim=-1)  # [C, D]
        
        # 计算相似度
        similarities = torch.einsum('ckd,cd->ck', prototypes_norm, simple_mean_norm)  # [C, K]
        similarities = similarities.detach().cpu().numpy()
        
        # 绘制
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(C)
        width = 0.8 / K
        
        colors = plt.cm.Set3(np.linspace(0, 1, K))
        
        for k in range(K):
            ax.bar(x_pos + k * width, similarities[:, k], width,
                  label=f'Prototype {k+1}', color=colors[k], alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=11)
        ax.set_ylabel('Cosine Similarity with Simple Mean', fontsize=11)
        ax.set_title('Prototypes vs Simple Mean Similarity', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos + width * (K - 1) / 2)
        ax.set_xticklabels([name.replace('_', ' ') for name in class_names],
                          rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[TPA-Vis] Saved prototype vs mean comparison → {save_path}")
        plt.close()
    
    def create_all_visualizations(
        self,
        checkpoint_path: str,
        config_path: str,
        prompts_json_path: str,
        text_embed_path: Optional[str] = None,
    ):
        """创建所有可视化"""
        print("=" * 70)
        print("TPA Visualization for CVPR Paper")
        print("=" * 70)
        
        # 加载数据
        data = self.load_model_and_extract_data(
            checkpoint_path, config_path, prompts_json_path, text_embed_path
        )
        
        # 1. Attention热力图
        print("\n[1/3] Creating attention heatmap...")
        self.visualize_attention_heatmap(
            data['attention_weights'],
            data['class_names'],
            data['prompts_text'],
            save_name="attention_heatmap.png"
        )
        
        # 2. Embedding space可视化
        print("\n[2/3] Creating embedding space visualization...")
        self.visualize_embedding_space(
            data['prototypes'],
            data['original_prompts'],
            data['simple_mean'],
            data['class_names'],
            method='tsne',
            save_name="embedding_space_tsne.png"
        )
        
        self.visualize_embedding_space(
            data['prototypes'],
            data['original_prompts'],
            data['simple_mean'],
            data['class_names'],
            method='pca',
            save_name="embedding_space_pca.png"
        )
        
        # 3. Prototypes vs Mean对比
        print("\n[3/3] Creating prototype vs mean comparison...")
        self.visualize_prototype_vs_mean_comparison(
            data['prototypes'],
            data['simple_mean'],
            data['class_names'],
            save_name="prototype_vs_mean.png"
        )
        
        print("\n" + "=" * 70)
        print("All visualizations saved to:", self.output_dir)
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="TPA Visualization for CVPR Paper")
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
        "--prompts-json",
        type=str,
        default="dataset2/metadata/lvis_prompts_claude.json",
        help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--text-embed",
        type=str,
        default=None,
        help="Path to text embeddings npy file (e.g., dataset2/metadata/lvis_claude_prompts_convnextl.npy). "
             "If not provided, will use embeddings loaded by the model."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tpa_cvpr_visualizations",
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    visualizer = TPAVisualizerForCVPR(output_dir=args.output_dir)
    visualizer.create_all_visualizations(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        prompts_json_path=args.prompts_json,
        text_embed_path=args.text_embed,
    )


if __name__ == "__main__":
    main()

