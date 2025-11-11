# TPA Diversity Loss 问题分析与修复方案

## 问题诊断

从attention热力图可以看出，所有prototypes的attention模式几乎相同，说明它们没有学习到不同的表示。

### 当前Diversity Loss的问题

```python
def _diversity_term(self, logits: torch.Tensor) -> torch.Tensor:
    C, K, N = logits.shape
    w = torch.softmax(logits, dim=1)  # [C, K, N]
    votes = w.sum(dim=-1)  # [C, K] - 每个prototype对所有prompts的总权重
    p = votes / (votes.sum(dim=1, keepdim=True) + 1e-8)  # [C, K] - 归一化
    entropy = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1) / math.log(K)
    return entropy.mean()
```

**问题**：
1. 这个loss鼓励所有prototypes的"使用频率"要均匀（高熵）
2. 但不鼓励不同prototypes关注不同的prompts
3. 如果所有prototypes都关注相同的prompts，只要它们的使用频率均匀，loss就会很小

### 正确的Diversity Loss应该

鼓励不同prototypes的attention分布不同，即：
- Proto1关注P1, P2
- Proto2关注P3, P4
- Proto3关注P5, P6
- ...

## 修复方案

### 方案1：最小化prototypes之间的attention相似度

```python
def _diversity_term_v2(self, logits: torch.Tensor) -> torch.Tensor:
    """
    鼓励不同prototypes的attention分布不同
    通过最小化不同prototypes之间的attention分布相似度
    """
    C, K, N = logits.shape
    w = torch.softmax(logits, dim=1)  # [C, K, N]
    
    # 对每个类别，计算不同prototypes之间的attention相似度
    diversity_loss = 0.0
    for c in range(C):
        attn_c = w[c]  # [K, N] - 该类别的所有prototypes的attention分布
        # 计算所有prototype对之间的余弦相似度
        attn_norm = F.normalize(attn_c, p=2, dim=1)  # [K, N]
        similarity_matrix = torch.mm(attn_norm, attn_norm.t())  # [K, K]
        
        # 只考虑非对角线元素（不同prototypes之间的相似度）
        mask = ~torch.eye(K, dtype=bool, device=similarity_matrix.device)
        off_diag_similarities = similarity_matrix[mask]
        
        # 最小化相似度 = 最大化多样性
        diversity_loss += off_diag_similarities.mean()
    
    return diversity_loss / C
```

### 方案2：鼓励每个prototype专注于不同的prompts

```python
def _diversity_term_v3(self, logits: torch.Tensor) -> torch.Tensor:
    """
    鼓励每个prototype专注于不同的prompts
    通过最大化不同prototypes的top-k prompts的重叠度惩罚
    """
    C, K, N = logits.shape
    w = torch.softmax(logits, dim=1)  # [C, K, N]
    
    diversity_loss = 0.0
    for c in range(C):
        attn_c = w[c]  # [K, N]
        
        # 对每个prototype，找到top-2 prompts
        topk = 2
        _, top_indices = torch.topk(attn_c, topk, dim=1)  # [K, topk]
        
        # 计算不同prototypes的top prompts重叠度
        overlap_penalty = 0.0
        for i in range(K):
            for j in range(i+1, K):
                # 计算两个prototypes的top prompts有多少重叠
                overlap = len(set(top_indices[i].cpu().tolist()) & 
                             set(top_indices[j].cpu().tolist()))
                overlap_penalty += overlap / topk
        
        diversity_loss += overlap_penalty / (K * (K-1) / 2)
    
    return diversity_loss / C
```

### 方案3：结合方案1和方案2

```python
def _diversity_term_combined(self, logits: torch.Tensor) -> torch.Tensor:
    """
    结合相似度惩罚和重叠度惩罚
    """
    C, K, N = logits.shape
    w = torch.softmax(logits, dim=1)  # [C, K, N]
    
    loss_sim = 0.0  # 相似度惩罚
    loss_overlap = 0.0  # 重叠度惩罚
    
    for c in range(C):
        attn_c = w[c]  # [K, N]
        
        # 1. 相似度惩罚
        attn_norm = F.normalize(attn_c, p=2, dim=1)
        sim_matrix = torch.mm(attn_norm, attn_norm.t())
        mask = ~torch.eye(K, dtype=bool, device=sim_matrix.device)
        loss_sim += sim_matrix[mask].mean()
        
        # 2. 重叠度惩罚
        _, top_indices = torch.topk(attn_c, 2, dim=1)
        overlap = 0.0
        for i in range(K):
            for j in range(i+1, K):
                overlap += len(set(top_indices[i].cpu().tolist()) & 
                              set(top_indices[j].cpu().tolist())) / 2.0
        loss_overlap += overlap / (K * (K-1) / 2)
    
    return (loss_sim / C) + 0.5 * (loss_overlap / C)
```

## 推荐的修复步骤

1. **立即修复**：使用方案1（最简单，最直接）
2. **调整超参数**：
   - `lambda_div`: 0.03 -> 0.10-0.15
   - `lambda_orth`: 0.10 -> 0.20
   - `tau`: 0.07 -> 0.10-0.12
3. **重新训练**：使用修复后的loss和调整后的超参数

## 验证方法

运行 `examples/analyze_tpa_prototypes.py` 来验证修复效果：
- Attention pattern similarity应该降低（< 0.5）
- Unique top prompts应该增加（接近K）
- Attention entropy应该适中（不要太均匀，也不要太集中）

