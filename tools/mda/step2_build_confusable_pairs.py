"""Task 1.2: Build confusable pairs via CLIP text similarity (Method A).

For each base class, finds top-3 most text-similar classes from all 1203.

Usage:
    python tools/mda/step2_build_confusable_pairs.py \
        --categories data/mda/lvis_categories.json \
        --out data/mda/confusable_pairs.json \
        [--top-k 3] [--device cuda]

Requires: pip install open_clip_torch  (or clip from OpenAI)
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F


def load_clip_text_encoder(device):
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        model = model.to(device).eval()
        return model, tokenizer, 'open_clip'
    except ImportError:
        pass
    try:
        import clip
        model, _ = clip.load('ViT-L/14', device=device)
        model.eval()
        return model, clip.tokenize, 'clip'
    except ImportError:
        raise ImportError(
            'Please install open_clip_torch or clip: '
            'pip install open_clip_torch')


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=64, mode='open_clip'):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if mode == 'open_clip':
            tokens = tokenizer(batch).to(device)
            embs = model.encode_text(tokens)
        else:
            tokens = tokenizer(batch, truncate=True).to(device)
            embs = model.encode_text(tokens)
        embs = F.normalize(embs.float(), dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', default='data/mda/lvis_categories.json')
    parser.add_argument('--out', default='data/mda/confusable_pairs.json')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.categories) as f:
        cat_data = json.load(f)

    all_cats = cat_data['categories']
    base_ids = set(cat_data['base'])  # frequent + common category ids

    all_names = [c['name'] for c in all_cats]
    all_ids = [c['id'] for c in all_cats]
    id_to_name = cat_data['id_to_name']

    print(f"Encoding {len(all_names)} category names with CLIP ViT-L/14 ...")
    device = torch.device(args.device)
    model, tokenizer, mode = load_clip_text_encoder(device)

    # Prompt template: "a photo of a {category}"
    prompts = [f'a photo of a {name}' for name in all_names]
    embs = encode_texts(model, tokenizer, prompts, device, mode=mode)
    # embs: [N, D], L2-normalised

    # Cosine similarity matrix
    sim = embs @ embs.T  # [N, N]

    result = {}
    for row_idx, cat in enumerate(all_cats):
        if cat['id'] not in base_ids:
            continue  # only build pairs for base classes

        row_sim = sim[row_idx].clone()
        row_sim[row_idx] = -1.0  # exclude self

        top_k_idx = row_sim.topk(args.top_k).indices.tolist()
        top_k_names = [all_names[j] for j in top_k_idx]
        result[cat['name']] = top_k_names

    # Print 10 examples for sanity check
    print("\n--- Sanity check: 10 confusable pairs ---")
    for name, negs in list(result.items())[:10]:
        print(f"  {name:30s} → {negs}")

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {len(result)} entries to {args.out}")


if __name__ == '__main__':
    main()
