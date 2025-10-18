#!/usr/bin/env python3
"""
Generate CLIP text embeddings for class names and save as .npy file.

This script:
1. Loads class names or prompts from a JSON file
2. Encodes them using CLIP text encoder
3. Saves the embeddings as a .npy file with shape [num_classes, embed_dim]

Usage:
    python tools/generate_text_embeddings.py \
        --prompt-json dataset2/metadata/lvis_prompts.json \
        --clip-model pretrained_models/clip_convnext_large_head.pth \
        --output dataset2/metadata/lvis_tpa_prompts_convnextl.npy \
        --aggregate none
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_clip_text_encoder(checkpoint_path: str, device: str = "cuda"):
    """
    Load CLIP text encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to clip_convnext_large_head.pth
        device: Device to load model on
        
    Returns:
        model: CLIP text encoder model
        tokenizer: CLIP tokenizer (if available)
    """
    print(f"Loading CLIP text encoder from: {checkpoint_path}")
    # PyTorch 2.6+ changed weights_only default to True, but CLIP checkpoints need False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # The checkpoint might be a full model or just state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # Need to initialize model architecture first
            # For now, we'll use the state dict directly
            model = checkpoint
        else:
            model = checkpoint
    else:
        model = checkpoint
    
    # Try to import CLIP
    try:
        import clip
        print("Using OpenAI CLIP")
        clip_model, preprocess = clip.load("ViT-L/14", device=device)
        text_encoder = clip_model
        
        # Try to load custom weights if they match
        if isinstance(model, dict):
            try:
                clip_model.load_state_dict(model, strict=False)
                print("Loaded custom CLIP weights")
            except Exception as e:
                print(f"Could not load custom weights: {e}")
                print("Using default CLIP weights")
        
        return clip_model, clip.tokenize
        
    except ImportError:
        print("OpenAI CLIP not installed. Trying open_clip...")
        
    try:
        import open_clip
        print("Using OpenCLIP")
        # ConvNeXt-Large CLIP model
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'convnext_large_d_320', 
            pretrained='laion2b_s29b_b131k_ft_soup',
            device=device
        )
        
        # Try to load custom weights
        if isinstance(model, dict):
            try:
                clip_model.load_state_dict(model, strict=False)
                print("Loaded custom CLIP weights")
            except Exception as e:
                print(f"Could not load custom weights: {e}")
                print("Using default OpenCLIP weights")
        
        return clip_model, open_clip.tokenize
        
    except ImportError:
        raise ImportError(
            "Neither 'clip' nor 'open_clip' is installed. Please install one:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "  or\n"
            "  pip install open_clip_torch"
        )


def encode_texts(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int = 256,
    device: str = "cuda",
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a list of text prompts using CLIP text encoder.
    
    Args:
        texts: List of text prompts to encode
        model: CLIP model
        tokenizer: CLIP tokenizer function
        batch_size: Batch size for encoding
        device: Device to run on
        normalize: Whether to L2-normalize embeddings
        
    Returns:
        embeddings: numpy array of shape [num_texts, embed_dim]
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            try:
                tokens = tokenizer(batch_texts).to(device)
            except Exception as e:
                # Fallback for different tokenizer signatures
                tokens = tokenizer(batch_texts, context_length=77).to(device)
            
            # Encode
            if hasattr(model, 'encode_text'):
                feats = model.encode_text(tokens)
            else:
                feats = model.text(tokens)
            
            # Normalize if requested
            if normalize:
                feats = F.normalize(feats, p=2, dim=-1)
            
            embeddings.append(feats.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)


def aggregate_embeddings(
    embeddings: np.ndarray,
    method: str = "mean"
) -> np.ndarray:
    """
    Aggregate multiple embeddings per class into a single embedding.
    
    Args:
        embeddings: Array of shape [num_prompts, embed_dim]
        method: Aggregation method ('mean', 'max', 'first')
        
    Returns:
        aggregated: Single embedding vector
    """
    if method == "mean":
        return np.mean(embeddings, axis=0)
    elif method == "max":
        return np.max(embeddings, axis=0)
    elif method == "first":
        return embeddings[0]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-json",
        type=Path,
        required=True,
        help="Path to JSON file with {class_name: [prompts]} mapping or list of class names",
    )
    parser.add_argument(
        "--clip-model",
        type=Path,
        default=None,
        help="Path to CLIP model checkpoint (e.g., clip_convnext_large_head.pth). "
             "If not provided, will use default pretrained CLIP.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for .npy file with embeddings",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "max", "first", "none"],
        help="How to aggregate multiple prompts per class. 'none' saves all prompts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize embeddings (recommended for CLIP)",
    )
    
    args = parser.parse_args()
    
    # Load prompts
    print(f"Loading prompts from: {args.prompt_json}")
    with args.prompt_json.open("r") as f:
        data = json.load(f)
    
    # Parse prompts
    if isinstance(data, dict):
        # {class_name: [prompts]} format
        class_names = list(data.keys())
        all_prompts = []
        prompt_counts = []
        
        for name in class_names:
            prompts = data[name]
            if isinstance(prompts, str):
                prompts = [prompts]
            all_prompts.extend(prompts)
            prompt_counts.append(len(prompts))
            
        print(f"Loaded {len(class_names)} classes with {len(all_prompts)} total prompts")
        
    elif isinstance(data, list):
        # Simple list of class names
        class_names = data
        all_prompts = class_names
        prompt_counts = [1] * len(class_names)
        print(f"Loaded {len(class_names)} class names")
    else:
        raise ValueError(f"Unexpected JSON format: {type(data)}")
    
    # Load CLIP model
    if args.clip_model and args.clip_model.exists():
        model, tokenizer = load_clip_text_encoder(str(args.clip_model), args.device)
    else:
        print("No CLIP checkpoint provided or file not found, using default pretrained CLIP")
        try:
            import clip
            model, _ = clip.load("ViT-L/14", device=args.device)
            tokenizer = clip.tokenize
        except ImportError:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                'convnext_large_d_320', 
                pretrained='laion2b_s29b_b131k_ft_soup',
                device=args.device
            )
            tokenizer = open_clip.tokenize
    
    # Encode all prompts
    print(f"Encoding {len(all_prompts)} prompts...")
    all_embeddings = encode_texts(
        all_prompts,
        model,
        tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        normalize=args.normalize,
    )
    
    print(f"Encoded embeddings shape: {all_embeddings.shape}")
    
    # Aggregate per class if needed
    if args.aggregate != "none":
        print(f"Aggregating embeddings using method: {args.aggregate}")
        class_embeddings = []
        idx = 0
        
        for count in prompt_counts:
            prompts_for_class = all_embeddings[idx:idx + count]
            aggregated = aggregate_embeddings(prompts_for_class, args.aggregate)
            class_embeddings.append(aggregated)
            idx += count
        
        final_embeddings = np.stack(class_embeddings, axis=0)
    else:
        # Save all prompts (reshape to [num_classes, num_prompts_per_class, dim])
        # This only works if all classes have the same number of prompts
        if len(set(prompt_counts)) == 1:
            num_prompts_per_class = prompt_counts[0]
            final_embeddings = all_embeddings.reshape(
                len(class_names), num_prompts_per_class, -1
            )
        else:
            print("Warning: Classes have different numbers of prompts. Saving as flat array.")
            final_embeddings = all_embeddings
    
    # Save embeddings
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, final_embeddings)
    
    print(f"✓ Saved embeddings to: {args.output}")
    print(f"  Shape: {final_embeddings.shape}")
    print(f"  Dtype: {final_embeddings.dtype}")
    
    # Verify the file can be loaded
    loaded = np.load(args.output)
    print(f"✓ Verified: Successfully loaded back with shape {loaded.shape}")


if __name__ == "__main__":
    main()

