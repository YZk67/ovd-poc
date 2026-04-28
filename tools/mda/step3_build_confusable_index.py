"""Task 1.3: Convert confusable_pairs.json (names) to confusable_index.json
(mmdetection continuous indices, i.e. category_id - 1).

Usage:
    python tools/mda/step3_build_confusable_index.py \
        --pairs data/mda/confusable_pairs.json \
        --categories data/mda/lvis_categories.json \
        --out data/mda/confusable_index.json
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', default='data/mda/confusable_pairs.json')
    parser.add_argument('--categories', default='data/mda/lvis_categories.json')
    parser.add_argument('--out', default='data/mda/confusable_index.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.pairs) as f:
        pairs = json.load(f)  # {name: [neg_name1, neg_name2, neg_name3]}

    with open(args.categories) as f:
        cat_data = json.load(f)

    # name → continuous index (= category_id - 1)
    name_to_cont = cat_data['name_to_cont']  # {name: int}
    cont_to_name = {int(k): v for k, v in cat_data['cont_to_name'].items()}

    result = {}
    missing_pos = 0
    missing_neg = 0

    for pos_name, neg_names in pairs.items():
        pos_idx = name_to_cont.get(pos_name)
        if pos_idx is None:
            missing_pos += 1
            continue

        neg_idxs = []
        for neg_name in neg_names:
            neg_idx = name_to_cont.get(neg_name)
            if neg_idx is None:
                missing_neg += 1
                continue
            neg_idxs.append(neg_idx)

        if neg_idxs:
            result[str(pos_idx)] = neg_idxs

    print(f"Built {len(result)} entries "
          f"(skipped {missing_pos} pos, {missing_neg} neg not in categories)")

    # Sanity check: reverse-verify 5 entries
    print("\n--- Sanity check: 5 entries ---")
    for str_idx, neg_idxs in list(result.items())[:5]:
        pos_n = cont_to_name.get(int(str_idx), '?')
        neg_ns = [cont_to_name.get(ni, '?') for ni in neg_idxs]
        print(f"  cont[{str_idx:4s}] {pos_n:35s} → {neg_ns}")

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
