"""
Generate Figure 2 of the paper: TPA prototype t-SNE + nearest-prompt table.

Inputs:
  --prototypes      .npy with shape (C=1203, K=5, D=768)  (from extract_tpa_prototypes.py)
  --prompt-embeds   .npy with shape (C=1203, 8, D=768)    (the lvis_claude_prompts_*.npy)
  --prompts-json    .json with {class_name: [8 prompts]}  (lvis_prompts_claude.json)

Outputs (in --output-dir):
  tpa_tsne.pdf         t-SNE scatter, 20 classes x 5 prototypes
  nearest_prompts.json {class_name: [{proto, sim, prompt_text}, ...]}
  nearest_prompts.tex  LaTeX-ready table for 3 example classes

Run locally with the .venv (sklearn + matplotlib):
    .venv/bin/python tools/plot_tpa_figure.py \
        --prototypes    tpa_prototypes.npy \
        --prompt-embeds dataset2/metadata/lvis_claude_prompts_convnextl.npy \
        --prompts-json  dataset2/metadata/lvis_prompts_claude.json \
        --output-dir    figs/tpa_analysis

If a class name in --classes is not found, the script searches for fuzzy
substring matches and prints what it picked.
"""
import argparse
import json
from pathlib import Path

import numpy as np


# ----- default class shortlist (visually diverse, mostly common in LVIS) -----
DEFAULT_CLASSES = [
    "dog", "cat", "horse", "bear", "elephant", "bird",
    "car", "bicycle", "motorcycle", "airplane",
    "fork", "knife", "scissors", "hammer",
    "chair", "lamp", "bed",
    "apple", "banana", "pizza",
]
# 3 classes that get a nearest-prompt table in the figure caption / appendix
DEFAULT_TABLE_CLASSES = ["dog", "fork", "pizza"]


def find_class_index(name: str, class_names):
    """Exact match first, then case-insensitive substring. Returns (idx, name) or None."""
    if name in class_names:
        return class_names.index(name), name
    lname = name.lower()
    for i, n in enumerate(class_names):
        if lname == n.lower():
            return i, n
    for i, n in enumerate(class_names):
        if lname in n.lower():
            return i, n
    return None


def run_tsne(points, perplexity, seed):
    from sklearn.manifold import TSNE

    print(f"[tsne] {points.shape}  perplexity={perplexity}  seed={seed}")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(points)


def plot_scatter(coords, K, names, out_pdf):
    import matplotlib.pyplot as plt

    n_classes = len(names)
    cmap = plt.cm.get_cmap("tab20", max(n_classes, 20))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, name in enumerate(names):
        pts = coords[i * K : (i + 1) * K]
        color = cmap(i % cmap.N)
        ax.scatter(
            pts[:, 0], pts[:, 1],
            color=color, s=55, alpha=0.85,
            edgecolors="white", linewidth=0.5,
            label=name,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 2-column legend for compactness in single-column NeurIPS layout
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=7, ncol=1, frameon=False, handletextpad=0.3,
        borderpad=0.3, labelspacing=0.25,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    print(f"[save] {out_pdf}")


def nearest_prompt_table(prototypes, prompt_embeds, prompts_dict, table_classes, class_names):
    """For each class in table_classes: for each of K prototypes, find nearest prompt
    by cosine similarity (over the 8 prompts of that class)."""
    K = prototypes.shape[1]
    table = {}
    for cn in table_classes:
        hit = find_class_index(cn, class_names)
        if hit is None:
            print(f"[warn] '{cn}' not found in class names; skipping")
            continue
        idx, resolved = hit
        cls_protos = prototypes[idx]                  # [K, D]
        cls_prompts = prompt_embeds[idx]              # [8, D]
        cls_texts = prompts_dict[resolved]            # list of 8 strings

        # cosine similarity
        protos_n = cls_protos / (np.linalg.norm(cls_protos, axis=-1, keepdims=True) + 1e-9)
        prompts_n = cls_prompts / (np.linalg.norm(cls_prompts, axis=-1, keepdims=True) + 1e-9)
        sims = protos_n @ prompts_n.T                  # [K, 8]
        nearest = sims.argmax(axis=1)                  # [K]
        sim_vals = sims[np.arange(K), nearest]         # [K]

        rows = []
        for k in range(K):
            rows.append({
                "proto": k,
                "nearest_prompt_idx": int(nearest[k]),
                "nearest_prompt": cls_texts[nearest[k]],
                "similarity": float(sim_vals[k]),
            })
        table[resolved] = rows
    return table


def write_latex_table(table, out_tex):
    """Format the nearest-prompt table as a LaTeX subfloat-friendly tabular."""
    lines = []
    lines.append(r"% nearest-prompt table for TPA Figure 2")
    lines.append(r"\begin{tabular}{c c c l}")
    lines.append(r"\toprule")
    lines.append(r"Class & Proto & Sim. & Nearest LLM-generated prompt \\")
    lines.append(r"\midrule")
    for cls_name, rows in table.items():
        for j, r in enumerate(rows):
            cls_cell = cls_name if j == 0 else ""
            prompt_safe = r["nearest_prompt"].replace("&", r"\&").replace("_", r"\_")
            lines.append(
                f"{cls_cell} & P{r['proto']} & {r['similarity']:.2f} & {prompt_safe} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[save] {out_tex}")


def print_table_pretty(table, K):
    print(f"\n=== Nearest-prompt table ===")
    for cls_name, rows in table.items():
        print(f"\n[{cls_name}]")
        for r in rows:
            print(f"  P{r['proto']}  sim={r['similarity']:.3f}  "
                  f"\"{r['nearest_prompt']}\"")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prototypes", required=True, help=".npy of shape (C, K, D)")
    p.add_argument("--prompt-embeds", required=True, help=".npy of shape (C, 8, D)")
    p.add_argument("--prompts-json", required=True, help="lvis prompts dict json")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                   help="class names to include in the t-SNE plot")
    p.add_argument("--table-classes", nargs="+", default=DEFAULT_TABLE_CLASSES,
                   help="class names for the nearest-prompt table")
    p.add_argument("--perplexity", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="figs/tpa_analysis")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    prototypes = np.load(args.prototypes)
    print(f"[load] prototypes: {prototypes.shape}")
    prompt_embeds = np.load(args.prompt_embeds)
    print(f"[load] prompt embeddings: {prompt_embeds.shape}")
    with open(args.prompts_json) as f:
        prompts_dict = json.load(f)
    class_names = list(prompts_dict.keys())
    print(f"[load] {len(class_names)} class names from prompts json")

    assert prototypes.shape[0] == prompt_embeds.shape[0] == len(class_names), (
        f"class-count mismatch: protos={prototypes.shape[0]} "
        f"embeds={prompt_embeds.shape[0]} names={len(class_names)}"
    )

    C, K, D = prototypes.shape

    # ---- pick classes for the t-SNE plot ----
    selected = []
    for cn in args.classes:
        hit = find_class_index(cn, class_names)
        if hit is None:
            print(f"[warn] '{cn}' not found; skipping")
            continue
        idx, resolved = hit
        if idx in (s[0] for s in selected):
            continue
        selected.append((idx, resolved))
        print(f"  resolved '{cn}' -> [{idx}] '{resolved}'")
    if len(selected) < 5:
        raise SystemExit("[!] fewer than 5 classes resolved; pass --classes")

    sel_idx = [s[0] for s in selected]
    sel_names = [s[1] for s in selected]
    sel_protos = prototypes[sel_idx].reshape(-1, D)  # [N*K, D]

    # ---- t-SNE ----
    coords = run_tsne(sel_protos, args.perplexity, args.seed)
    plot_scatter(coords, K, sel_names, out_dir / "tpa_tsne.pdf")

    # ---- nearest-prompt table ----
    table = nearest_prompt_table(
        prototypes, prompt_embeds, prompts_dict, args.table_classes, class_names
    )
    (out_dir / "nearest_prompts.json").write_text(json.dumps(table, indent=2))
    print(f"[save] {out_dir / 'nearest_prompts.json'}")
    write_latex_table(table, out_dir / "nearest_prompts.tex")
    print_table_pretty(table, K)


if __name__ == "__main__":
    main()
