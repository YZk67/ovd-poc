"""Task 1.1: Extract LVIS v1 category info and generate label map files.

Outputs:
  1. data/mda/lvis_categories.json  — internal format for steps 2 & 3
  2. data/coco/annotations/lvis_v1_label_map.json      — all 1203 classes
  3. data/coco/annotations/lvis_v1_label_map_norare.json — 866 base classes only

Index convention (matches lvis2ovd.py):
  continuous_index = category_id - 1   (LVIS ids are 1-1203, consecutive)

Usage:
    python tools/mda/step1_extract_lvis_categories.py \
        --ann /path/to/lvis_v1_val.json \
        --out-dir data/mda \
        --ann-out-dir data/coco/annotations
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann',
        default='data/coco/annotations/lvis_v1_val.json',
        help='Path to lvis_v1_val.json')
    parser.add_argument(
        '--out-dir',
        default='data/mda',
        help='Output dir for lvis_categories.json')
    parser.add_argument(
        '--ann-out-dir',
        default='data/coco/annotations',
        help='Output dir for lvis_v1_label_map*.json (used by mmdetection)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ann_out_dir, exist_ok=True)

    with open(args.ann) as f:
        data = json.load(f)

    cats = data['categories']  # list of dicts with id, name, frequency

    # ── Internal format ───────────────────────────────────────────────────────
    result = {
        'categories': [],
        'name_to_id': {},       # name → original LVIS category_id
        'id_to_name': {},       # str(category_id) → name
        'cont_to_name': {},     # str(cont_idx) → name  (cont_idx = id - 1)
        'name_to_cont': {},     # name → cont_idx
        'frequent': [],         # cont_idx list
        'common': [],
        'rare': [],             # novel classes in OV-LVIS (~337)
        'base': [],             # frequent + common (~866)
    }

    # ── Label map for mmdetection ─────────────────────────────────────────────
    # lvis_v1_label_map.json: {str(cont_idx): name} for all 1203 classes
    label_map_all = {}
    # lvis_v1_label_map_norare.json: same but only base classes
    label_map_norare = {}

    for cat in sorted(cats, key=lambda c: c['id']):
        name = cat['name'].replace('_', ' ')
        cid = cat['id']               # original LVIS id (1-1203)
        cont = cid - 1                # continuous index (0-1202)
        freq = cat.get('frequency', 'r')

        result['categories'].append({
            'id': cid,
            'cont_idx': cont,
            'name': name,
            'frequency': freq,
        })
        result['name_to_id'][name] = cid
        result['id_to_name'][str(cid)] = name
        result['cont_to_name'][str(cont)] = name
        result['name_to_cont'][name] = cont

        label_map_all[str(cont)] = name

        if freq == 'r':
            result['rare'].append(cont)
        elif freq == 'f':
            result['frequent'].append(cont)
            result['base'].append(cont)
            label_map_norare[str(cont)] = name
        elif freq == 'c':
            result['common'].append(cont)
            result['base'].append(cont)
            label_map_norare[str(cont)] = name

    # ── Print stats ───────────────────────────────────────────────────────────
    print(f"Total categories: {len(result['categories'])}")
    print(f"  frequent:   {len(result['frequent'])}")
    print(f"  common:     {len(result['common'])}")
    print(f"  rare:       {len(result['rare'])}")
    print(f"  base (f+c): {len(result['base'])}")
    print(f"\nIndex range (all):  0 – {max(int(k) for k in label_map_all)}")
    print(f"Index range (base): sample = {sorted(result['base'])[:5]} ...")

    # ── Write files ───────────────────────────────────────────────────────────
    cats_path = os.path.join(args.out_dir, 'lvis_categories.json')
    with open(cats_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {cats_path}")

    lm_all_path = os.path.join(args.ann_out_dir, 'lvis_v1_label_map.json')
    with open(lm_all_path, 'w') as f:
        json.dump(label_map_all, f, indent=2)
    print(f"Saved: {lm_all_path}")

    lm_norare_path = os.path.join(args.ann_out_dir,
                                  'lvis_v1_label_map_norare.json')
    with open(lm_norare_path, 'w') as f:
        json.dump(label_map_norare, f, indent=2)
    print(f"Saved: {lm_norare_path}")


if __name__ == '__main__':
    main()
