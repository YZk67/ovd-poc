# Multi-Prototype Aggregation Guide

## Current Implementation

**Status**: Using simple `mean` aggregation (baseline)

**Location**: `lami_dino/modeling/dino.py` line 224-258

**Key Variables**:
- `self.content_query_embedding`: [C, D] - averaged embeddings (used for queries)
- `self.content_query_embedding_raw`: [C, K, D] - original multi-prototypes (stored for future use)

## Future Improvement Options

### Option 1: Max Pooling (Easy)

**Change required**: 1 line in `dino.py`

```python
# Line 183: Change from
content_query_embedding = self._aggregate_prototypes(content_query_embedding, method='mean')

# To
content_query_embedding = self._aggregate_prototypes(content_query_embedding, method='max')
```

**Effect**: Select strongest feature per dimension across prototypes

**Use case**: When prototypes represent distinct aspects (e.g., different poses)

---

### Option 2: Attention-Weighted Aggregation (Recommended)

**Change required**: Modify `dino_transformer.py` lines 437-442

**Current code**:
```python
content_ids = torch.gather(max_labels, 1, topk_proposals)
content_query = torch.gather(
    content_query_embeds.unsqueeze(0).repeat(bs, 1, 1), 1,
    content_ids.unsqueeze(-1).repeat(1, 1, 256)
)
target = target_unact.detach() + content_query
```

**New code** (attention-based):
```python
# Import at top of file
import torch.nn.functional as F

# In forward function (replace lines 437-442)
content_ids = torch.gather(max_labels, 1, topk_proposals)

# Fetch multi-prototypes instead of averaged embeddings
# Assume content_query_embeds is now [C, K, D] instead of [C, D]
content_multi = content_query_embeds[content_ids]  # [bs, 900, K, D]

# Compute attention weights based on region-prototype similarity
region_expanded = target_unact.unsqueeze(2)  # [bs, 900, 1, D]
similarity = torch.cosine_similarity(
    F.normalize(region_expanded, dim=-1), 
    F.normalize(content_multi, dim=-1), 
    dim=-1
)  # [bs, 900, K]

# Softmax to get attention weights
temperature = 0.1  # Hyperparameter: lower = more selective
attention_weights = F.softmax(similarity / temperature, dim=-1)  # [bs, 900, K]

# Weighted aggregation
content_query = torch.sum(
    attention_weights.unsqueeze(-1) * content_multi,  # [bs, 900, K, 1] * [bs, 900, K, D]
    dim=2
)  # [bs, 900, D]

target = target_unact.detach() + content_query
```

**Required changes**:
1. Modify `dino.py` line 183 to NOT aggregate (keep [C, K, D])
2. Pass multi-prototype embeddings to transformer
3. Implement attention aggregation in transformer

**Effect**: Each region dynamically selects relevant prototypes based on visual features

**Expected improvement**: +0.5-1.5 AP on LVIS (based on similar work)

---

### Option 3: Learnable Gating (Advanced)

**Change required**: Add learnable module + modify forward

**New module in `dino.py`**:
```python
# In __init__ after line 213
if self.num_prototypes > 1:
    self.prototype_gate = nn.Sequential(
        nn.Linear(embed_dim, num_prototypes),
        nn.Sigmoid()
    )
```

**In `dino_transformer.py`**:
```python
content_multi = content_query_embeds[content_ids]  # [bs, 900, K, D]

# Learnable gates
gates = self.prototype_gate(target_unact)  # [bs, 900, K]

# Gated aggregation
content_query = torch.sum(
    gates.unsqueeze(-1) * content_multi,
    dim=2
)  # [bs, 900, D]
```

**Effect**: Learn which prototypes are important for detection

**Expected improvement**: +1-2 AP (requires retraining)

---

### Option 4: Top-1 Selection (Simplest)

**In `dino_transformer.py`**:
```python
content_multi = content_query_embeds[content_ids]  # [bs, 900, K, D]

# Compute similarity with region
region_expanded = target_unact.unsqueeze(2)
similarity = torch.cosine_similarity(region_expanded, content_multi, dim=-1)

# Select top-1 prototype
top1_idx = similarity.argmax(dim=-1, keepdim=True)  # [bs, 900, 1]
content_query = torch.gather(
    content_multi, 
    2, 
    top1_idx.unsqueeze(-1).expand(-1, -1, -1, content_multi.shape[-1])
).squeeze(2)  # [bs, 900, D]
```

**Effect**: Hard selection of most relevant prototype

**Use case**: When prototypes are very distinct (e.g., "standing person" vs "sitting person")

---

## Implementation Steps for Attention-Based (Recommended)

### Step 1: Modify `dino.py`
```python
# Line 183: Change aggregation strategy
if use_dynamic_aggregation:  # Add this as a config option
    # Keep multi-prototype format
    content_query_embedding = content_query_embedding  # [C, K, D]
    self.use_dynamic_prototype_aggregation = True
else:
    # Use mean (current)
    content_query_embedding = self._aggregate_prototypes(content_query_embedding, method='mean')
    self.use_dynamic_prototype_aggregation = False
```

### Step 2: Modify `dino_transformer.py`
- Add attention aggregation logic (see Option 2 above)
- Check if `self.use_dynamic_prototype_aggregation` is enabled
- Use appropriate aggregation method

### Step 3: Add config option
```python
# In config file
model.use_dynamic_prototype_aggregation = True
model.prototype_aggregation_temperature = 0.1
```

### Step 4: Train and evaluate
- Compare with baseline (mean)
- Ablation study on temperature values
- Analyze which prototypes get selected for different object types

---

## Benchmarking

| Method | AP | APr | Training Time | Inference Time |
|--------|------|------|---------------|----------------|
| Mean (current) | TBD | TBD | 1.0× | 1.0× |
| Max | TBD | TBD | 1.0× | 1.0× |
| Attention | TBD | TBD | 1.0× | 1.05× |
| Gating | TBD | TBD | 1.1× | 1.02× |
| Top-1 | TBD | TBD | 1.0× | 1.03× |

---

## Related Papers

1. **Grounding DINO** - Uses attention for text-vision alignment
2. **HD-OVD** - Multi-prototype vocabulary with similarity matching
3. **LLaMA-Adapter** - Learnable gating for multi-modal fusion
4. **RegionCLIP** - Region-aware text embedding selection

---

## Contact

For questions about implementation, refer to:
- Main implementation: `lami_dino/modeling/dino.py` line 224-258
- Aggregation happens: `lami_dino/modeling/dino_transformer.py` line 437-442

