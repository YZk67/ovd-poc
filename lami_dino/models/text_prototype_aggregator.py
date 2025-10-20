from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPrototypeAggregator(nn.Module):
    """
    Learnable Semantic Aggregator
    
    Purpose: Aggregates per-class phrase embeddings (shape: [num_classes, num_phrases, dim]) 
             into K semantic prototypes per class.
    
    Use case: In open-vocabulary object detection, each class may have multiple text descriptions
              (e.g., "cat", "a cat", "feline animal"). This module uses an attention mechanism
              to aggregate these descriptions into a fixed number of representative prototypes.
    """

    def __init__(
        self,
        dim: int,                      # Dimension of input text features
        num_prototypes: int = 4,       # Number of prototypes to generate per class (K)
        hidden_dim: int = 256,         # Hidden dimension for attention mechanism
        dropout: float = 0.1,          # Dropout rate for regularization
        tau: float = 0.1,              # Temperature parameter controlling attention sharpness
    ) -> None:
        super().__init__()
        # Parameter validation
        if num_prototypes <= 0:
            raise ValueError(f"num_prototypes must be > 0, got {num_prototypes}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")

        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)  # Ensure tau doesn't become too small

        # Key projection: project text features to hidden space for attention computation
        self.key_proj = nn.Linear(dim, hidden_dim)
        # Value projection: project text features for weighted aggregation
        self.value_proj = nn.Linear(dim, dim)
        # Prototype queries: K learnable query vectors to "ask" for different semantic aspects
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize model parameters using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.prototype_queries)

    @property
    def hidden_dim(self) -> int:
        """Returns the hidden dimension"""
        return self.key_proj.out_features

    def forward(self, text_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: aggregates multiple phrase embeddings into fixed-number prototypes via attention
        
        Core idea: Uses Transformer-like cross-attention mechanism
        - K learnable query vectors (prototype_queries) represent different semantic aspects
        - Each query vector computes attention weights over all phrases
        - Phrases are aggregated via weighted sum according to attention weights to produce K prototypes
        
        Args:
            text_feats: Text feature tensor with shape [num_classes, num_phrases, dim]
                       Example: [1203 classes, 8 phrases per class, 768-dim features]

        Returns:
            Prototype tensor with shape [num_classes, num_prototypes, dim]
            Example: [1203 classes, 4 prototypes per class, 768-dim features]
        """
        # Input validation
        if text_feats.ndim != 3:
            raise ValueError(f"text_feats must be [C, N, D], got shape {tuple(text_feats.shape)}")

        class_count, phrase_count, feat_dim = text_feats.shape
        if phrase_count == 0:
            raise ValueError("num_phrases should be > 0.")

        # Step 1: Project to key and value spaces
        keys = self.key_proj(text_feats)      # [C, N, H] - for computing attention
        values = self.value_proj(text_feats)  # [C, N, D] - for weighted aggregation

        # Step 2: Compute attention weights
        # Einstein summation: each prototype query (K) computes similarity with phrase keys (N)
        logits = torch.einsum("kh,cnh->ckn", self.prototype_queries, keys)  # [C, K, N]
        logits = logits / math.sqrt(self.hidden_dim)  # Scale for gradient stability
        attn = F.softmax(logits / self.tau, dim=-1)    # [C, K, N] - attention weights
        # Smaller tau → sharper attention; larger tau → more diffuse attention

        # Step 3: Aggregate values according to attention weights
        # Each prototype (K) performs weighted sum over N phrases
        prototypes = torch.einsum("ckn,cnd->ckd", attn, values)  # [C, K, D]
        prototypes = self.dropout(prototypes)  # Regularization
        return prototypes


class TextPrototypeBank(nn.Module):
    """
    Text Prototype Bank
    
    Purpose: Manages raw phrase embeddings and provides aggregated prototype outputs on demand
    
    Responsibilities:
    1. Load and store pre-computed text embeddings (from CLIP or similar models)
    2. Provide caching mechanism during inference for efficiency
    3. Ensure proper gradient propagation during training
    
    Design highlights:
    - Uses buffer to store text embeddings (non-trainable but auto-transferred to device)
    - Implements smart caching: recompute during training, cache during inference
    - Supports automatic device alignment

    Args:
        embedding_path: Path to numpy array containing pre-computed text embeddings
                        Expected shape: [C, N, D] or [C, D]
                        C=num_classes, N=num_phrases per class, D=feature_dim
        aggregator: Optional TextPrototypeAggregator instance.
                    If None, a default aggregator will be created
        num_prototypes: Number of semantic prototypes per class (only used if aggregator is None)
        dtype: Target dtype for text embeddings
    """

    def __init__(
        self,
        embedding_path: str,
        aggregator: Optional[TextPrototypeAggregator] = None,
        *,
        num_prototypes: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Load pre-computed text embeddings
        raw = np.load(embedding_path)
        # If 2D ([C, D]), expand to 3D ([C, 1, D])
        if raw.ndim == 2:
            raw = raw[:, None, :]
        if raw.ndim != 3:
            raise ValueError(
                f"Expected text embedding array with 2 or 3 dims, got shape {raw.shape}."
            )

        # Convert to PyTorch tensor and register as buffer
        # persistent=False means it won't be saved in checkpoint (can be reloaded)
        text_feats = torch.from_numpy(raw).to(dtype=dtype)
        self.register_buffer("text_feats", text_feats, persistent=False)

        self.num_classes, self.num_phrases, self.feat_dim = text_feats.shape
        self.num_prototypes = num_prototypes

        # Create or use provided aggregator
        if aggregator is None:
            aggregator = TextPrototypeAggregator(
                dim=self.feat_dim,
                num_prototypes=num_prototypes,
            )
        self.aggregator = aggregator

        # Caching mechanism: avoid redundant computation during inference
        self._cached_step = -1
        self._cached_prototypes: Optional[torch.Tensor] = None

    def _ensure_device(self) -> torch.Tensor:
        """
        Ensure text embeddings are on the same device as the aggregator
        
        Important for multi-GPU training or CPU/GPU switching to ensure buffers
        and parameters reside on the same device
        """
        target_device = next(self.aggregator.parameters()).device
        if self.text_feats.device != target_device:
            self.text_feats = self.text_feats.to(device=target_device)
        return self.text_feats

    def reset_cache(self) -> None:
        """Reset cache (e.g., when switching between train/eval modes)"""
        self._cached_prototypes = None
        self._cached_step = -1

    def forward(self, step: int = -1, *, force_recompute: bool = False) -> torch.Tensor:
        """
        Forward pass: returns aggregated prototypes
        
        Smart caching strategy:
        - Training mode (self.training=True): always recompute to ensure gradient flow
        - Eval mode (self.training=False): use cache to speed up inference
        
        Cache invalidation conditions:
        1. First call (cache is empty)
        2. Forced recomputation (force_recompute=True)
        3. Step change (step differs)
        4. Device change
        
        Args:
            step: Current iteration step, used as cache key. -1 means don't care about step
            force_recompute: Whether to force recomputation (ignore cache)
        
        Returns:
            Prototype tensor with shape [num_classes, num_prototypes, feat_dim]
        """
        # Ensure device consistency
        text_feats = self._ensure_device()

        # Training mode: always recompute to ensure gradient flow
        if self.training or force_recompute:
            return self.aggregator(text_feats)

        # Eval mode: use caching mechanism
        if (
            self._cached_prototypes is None
            or force_recompute
            or step != self._cached_step
            or self._cached_prototypes.device != text_feats.device
        ):
            # Compute in no-grad mode to save memory
            with torch.no_grad():
                prototypes = self.aggregator(text_feats)
            self._cached_prototypes = prototypes
            self._cached_step = step
        
        return self._cached_prototypes
