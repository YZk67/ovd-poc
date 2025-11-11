#!/usr/bin/env python3
"""
TPA训练过程可视化示例脚本

使用方法：
1. 从训练日志CSV可视化训练曲线：
   python examples/visualize_tpa_training.py \
       --mode curves \
       --csv output/training_log.csv \
       --output-dir tpa_visualizations

2. 从checkpoint可视化prototypes：
   python examples/visualize_tpa_training.py \
       --mode comprehensive \
       --checkpoint output/model_final.pth \
       --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
       --class-names dataset2/lvis/lvis_v1_all_classes.json \
       --csv output/training_log.csv \
       --output-dir tpa_visualizations
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lami_dino.visuliazation.tpa_training_visualizer import (
    TPATrainingVisualizer,
    load_model_from_checkpoint,
    load_class_names
)
import json


def load_lvis_class_names(json_path: str) -> list:
    """
    从LVIS JSON文件加载类别名称
    
    Args:
        json_path: LVIS类别JSON文件路径（如lvis_v1_all_classes.json）
    
    Returns:
        类别名称列表
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 如果是简单的类别名称列表
    if isinstance(data, list):
        return data
    
    # 如果是包含categories的字典
    if isinstance(data, dict):
        if 'categories' in data:
            categories = sorted(data['categories'], key=lambda x: x.get('id', 0))
            return [cat.get('name', f'class_{i}') for i, cat in enumerate(categories)]
        elif 'thing_classes' in data:
            return data['thing_classes']
    
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TPA Training Process Visualizer Example")
    parser.add_argument("--mode", type=str, default="comprehensive",
                       choices=["comprehensive", "curves", "prototypes"],
                       help="Visualization mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint path")
    parser.add_argument("--config", type=str, 
                       default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py",
                       help="Model config path")
    parser.add_argument("--csv", type=str, default=None,
                       help="Training log CSV path (e.g., output/training_log.csv)")
    parser.add_argument("--class-names", type=str, default=None,
                       help="Path to class names JSON file (e.g., dataset2/lvis/lvis_v1_all_classes.json)")
    parser.add_argument("--output-dir", type=str, default="tpa_visualizations",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    visualizer = TPATrainingVisualizer(output_dir=args.output_dir)
    
    if args.mode == "comprehensive":
        if args.checkpoint is None:
            print("[Error] --checkpoint is required for comprehensive mode")
            print("\nExample usage:")
            print("  python examples/visualize_tpa_training.py \\")
            print("      --mode comprehensive \\")
            print("      --checkpoint output/model_final.pth \\")
            print("      --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \\")
            print("      --class-names dataset2/lvis/lvis_v1_all_classes.json \\")
            print("      --csv output/training_log.csv")
            return
        
        print(f"[Info] Loading model from {args.checkpoint}...")
        model = load_model_from_checkpoint(args.checkpoint, args.config)
        
        # 加载类别名称
        class_names = None
        if args.class_names:
            print(f"[Info] Loading class names from {args.class_names}...")
            class_names = load_lvis_class_names(args.class_names)
            if class_names:
                print(f"[Info] Loaded {len(class_names)} class names")
            else:
                print("[Warning] Could not load class names, continuing without them...")
        
        # 创建综合报告
        visualizer.create_comprehensive_report(
            model,
            csv_path=args.csv,
            class_names=class_names,
            prefix="tpa_report"
        )
        
        print(f"\n[Success] Visualizations saved to {args.output_dir}/")
        print("\nGenerated files:")
        print("  - tpa_report_curves.png: Training curves")
        print("  - tpa_report_similarity.png: Prototype similarity matrices")
        print("  - tpa_report_attention.png: Attention weight distributions")
        print("  - tpa_report_embedding_tsne.png: t-SNE visualization")
        print("  - tpa_report_embedding_pca.png: PCA visualization")
        print("  - tpa_report_orthogonality.png: Orthogonality heatmap")
    
    elif args.mode == "curves":
        if args.csv is None:
            print("[Error] --csv is required for curves mode")
            print("\nExample usage:")
            print("  python examples/visualize_tpa_training.py \\")
            print("      --mode curves \\")
            print("      --csv output/training_log.csv")
            return
        
        print(f"[Info] Visualizing training curves from {args.csv}...")
        visualizer.visualize_training_curves(args.csv)
        print(f"\n[Success] Training curves saved to {args.output_dir}/training_curves.png")
    
    elif args.mode == "prototypes":
        if args.checkpoint is None:
            print("[Error] --checkpoint is required for prototypes mode")
            return
        
        print(f"[Info] Loading model from {args.checkpoint}...")
        model = load_model_from_checkpoint(args.checkpoint, args.config)
        
        class_names = None
        if args.class_names:
            class_names = load_lvis_class_names(args.class_names)
        
        # 提取TPA
        try:
            tpa = model.transformer.decoder.class_embed[0].tpa
        except AttributeError:
            print("[Error] Could not find TPA in model")
            return
        
        # 获取prototypes
        if hasattr(tpa, '_last_prototypes') and tpa._last_prototypes is not None:
            prototypes = tpa._last_prototypes
        else:
            print("[Info] Running forward pass to generate prototypes...")
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'text_proto_bank'):
                text_feats = model.transformer.text_proto_bank.text_feats
                prototypes, _ = model.transformer.text_proto_bank.aggregator(text_feats, with_loss=False)
            else:
                print("[Error] Cannot generate prototypes")
                return
        
        # 可视化prototypes
        visualizer.visualize_prototype_similarity(
            prototypes,
            class_names=class_names,
            save_name="prototypes_similarity.png"
        )
        
        visualizer.visualize_orthogonality_heatmap(
            prototypes,
            save_name="prototypes_orthogonality.png"
        )
        
        print(f"\n[Success] Prototype visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

