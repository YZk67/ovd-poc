#!/usr/bin/env python3
"""Apply human review corrections to VLM crop description JSON files.

Corrections from manual review of 855 flagged annotations:
1. best_class overrides (where GT was wrong or VLM class should differ)
2. confidence overrides (where certainty is lower than VLM originally reported)
3. top5_similar additions (co-occurring objects noted during review)
"""

import json
import os
import argparse
from collections import defaultdict

DESC_DIR = "dataset/vlm_targets_crop/descriptions"

# ── Category 1: best_class corrections ────────────────────────────────
# (img_id, ann_id) → new best_class, new confidence
BEST_CLASS_OVERRIDES = {
    (33946, 1906718): ("orange", 0.55),   # GT=apple, actually orange fruit
    (34104, 532709):  ("bowl", 0.55),     # GT=person, crop is salad bowl
    (34397, 1938027): ("couch", 0.60),    # GT=chair, actually single sofa
    (34993, 29661):   ("laptop", 0.70),   # GT=tv, actually ThinkPad laptop screen
    (35825, 1930506): ("couch", 0.60),    # GT=chair, actually dark sofa
}

# ── Category 2: confidence-only overrides ─────────────────────────────
# (img_id, ann_id) → new confidence (best_class stays same as current)
CONFIDENCE_OVERRIDES = {
    (34086, 1047194):  0.55,   # GT=apple, pear-shaped fruit but apple is closest COCO
    (34104, 1048628):  0.50,   # GT=apple, unclear fruit slices in salad
    (34128, 1969461):  0.75,   # GT=tv, background monitor (not primary subject)
    (34279, 1039447):  0.65,   # GT=bowl, plastic container
    (34482, 1799017):  0.35,   # GT=truck, crop mainly shows parking meter
    (34579, 1152960):  0.60,   # GT=vase, glass bowl with roses (vase/bowl ambiguous)
    (34579, 1532627):  0.65,   # GT=bowl, same flower arrangement diff frame
    (34689, 1153977):  0.60,   # GT=vase, toothbrush holder (vase-like shape)
    (34732, 276588):   0.40,   # GT=sheep, actually Nara deer (no COCO deer class)
    (34930, 1797928):  0.50,   # GT=truck, background SUV partially visible
    (35190, 1534146):  0.50,   # GT=bowl, actually a frying pan
    (35741, 1056099):  0.40,   # GT=broccoli, looks more like parsley/cilantro
    (37616, 1978621):  0.40,   # GT=microwave, unopened Sunbeam box
}

# ── Category 3: top5_similar additions ────────────────────────────────
# (img_id, ann_id) → {class: score} to merge into existing top5
# Scores: 0.4 = clearly visible co-object, 0.3 = present in scene
TOP5_ADDITIONS = {
    # img 33946
    (33946, 1053922): {"banana": 0.3},
    (33946, 1906718): {"orange": 0.7, "banana": 0.5, "bowl": 0.4},
    # img 33955
    (33955, 1094551): {"sink": 0.3},
    # img 33979
    (33979, 900100033979): {"kite": 0.4},
    # img 33991
    (33991, 481425): {"elephant": 0.4},
    (33991, 1227109): {"elephant": 0.4},
    # img 33995
    (33995, 1075162): {"bowl": 0.3},
    # img 34013
    (34013, 602187): {"dog": 0.4},
    # img 34038
    (34038, 28299): {"cat": 0.4},
    # img 34074
    (34074, 424974): {"sports ball": 0.3},
    (34074, 496660): {"sports ball": 0.3},
    (34074, 507344): {"sports ball": 0.3},
    (34074, 510927): {"sports ball": 0.3},
    # img 34086
    (34086, 1047194): {"orange": 0.4},
    # img 34097
    (34097, 206890): {"horse": 0.4},
    # img 34104
    (34104, 532709): {"fork": 0.3, "person": 0.3},
    (34104, 691508): {"bowl": 0.3, "knife": 0.3},
    (34104, 1048583): {"orange": 0.3},
    # img 34121
    (34121, 1534247): {"orange": 0.3, "banana": 0.3},
    # img 34128
    (34128, 1969461): {"keyboard": 0.3, "mouse": 0.3, "donut": 0.3},
    # img 34138
    (34138, 2155717): {"knife": 0.3, "donut": 0.3},
    # img 34151
    (34151, 1138635): {"pizza": 0.3, "bottle": 0.3},
    # img 34180
    (34180, 442831): {"banana": 0.4},
    (34180, 443033): {"banana": 0.4},
    (34180, 473122): {"banana": 0.4},
    # img 34196
    (34196, 31941): {"laptop": 0.3, "keyboard": 0.3},
    (34196, 36035): {"laptop": 0.3, "keyboard": 0.3},
    # img 34221
    (34221, 448214): {"sink": 0.3},
    (34221, 1727771): {"sink": 0.3},
    # img 34234
    (34234, 2166366): {"skateboard": 0.4},
    # img 34235
    (34235, 83146): {"laptop": 0.3},
    (34235, 187571): {"laptop": 0.4, "couch": 0.3, "cup": 0.3, "bottle": 0.3},
    # img 34246
    (34246, 134110): {"motorcycle": 0.4, "horse": 0.3},
    # img 34279
    (34279, 1039447): {"hot dog": 0.4},
    # img 34301
    (34301, 1095173): {"sink": 0.3},
    # img 34304
    (34304, 1076542): {"bowl": 0.3},
    # img 34333
    (34333, 438669): {"tie": 0.4},
    # img 34354
    (34354, 487805): {"baseball bat": 0.4, "baseball glove": 0.3},
    # img 34381
    (34381, 906200034381): {"umbrella": 0.3, "person": 0.3},
    # img 34397
    (34397, 1938027): {"chair": 0.4, "book": 0.3},
    # img 34404
    (34404, 467515): {"skis": 0.4, "snowboard": 0.3},
    (34404, 515738): {"skis": 0.4},
    # img 34423
    (34423, 685227): {"knife": 0.3, "wine glass": 0.3},
    # img 34428
    (34428, 1729631): {"snowboard": 0.4},
    # img 34437
    (34437, 517620): {"skis": 0.4},
    (34437, 521242): {"skis": 0.4},
    (34437, 529681): {"skis": 0.4},
    (34437, 1216513): {"skis": 0.4},
    # img 34439
    (34439, 425715): {"skateboard": 0.4},
    (34439, 455516): {"skateboard": 0.4},
    (34439, 500478): {"skateboard": 0.4},
    # img 34454
    (34454, 647672): {"dog": 0.4, "person": 0.4},
    # img 34464
    (34464, 1375405): {"person": 0.3, "surfboard": 0.3},
    # img 34475
    (34475, 1732064): {"tie": 0.4},
    # img 34478
    (34478, 511450): {"bowl": 0.3, "cup": 0.3, "bottle": 0.3},
    (34478, 703810): {"fork": 0.3, "bowl": 0.3},
    (34478, 1533408): {"spoon": 0.3},
    # img 34480
    (34480, 2157891): {"remote": 0.4, "couch": 0.3},
    # img 34482
    (34482, 1799017): {"parking meter": 0.5, "car": 0.3},
    # img 34487
    (34487, 455619): {"bench": 0.4, "backpack": 0.3, "dog": 0.3},
    # img 34500
    (34500, 431772): {"tennis racket": 0.4, "sports ball": 0.3},
    # img 34508
    (34508, 436342): {"snowboard": 0.4},
    # img 34523
    (34523, 687095): {"hot dog": 0.4, "knife": 0.3},
    # img 34535
    (34535, 1931328): {"handbag": 0.4},
    # img 34579
    (34579, 1152960): {"bowl": 0.4},
    (34579, 1532627): {"vase": 0.4},
    # img 34597
    (34597, 467567): {"skateboard": 0.4},
    (34597, 533534): {"skateboard": 0.4},
    (34597, 535175): {"skateboard": 0.4},
    (34597, 537463): {"skateboard": 0.4},
    # img 34611
    (34611, 78778): {"cell phone": 0.3, "cup": 0.3},
    (34611, 454268): {"cup": 0.3, "bottle": 0.3, "cell phone": 0.3},
    # img 34616
    (34616, 425159): {"skateboard": 0.4},
    # img 34617
    (34617, 595337): {"person": 0.4},
    # img 34657
    (34657, 1716093): {"remote": 0.4, "couch": 0.3},
    # img 34682
    (34682, 425074): {"surfboard": 0.4},
    (34682, 650712): {"person": 0.4},
    # img 34689
    (34689, 1153977): {"toothbrush": 0.3, "cup": 0.3},
    # img 34732
    (34732, 276588): {"horse": 0.3, "cow": 0.3},
    # img 34754
    (34754, 421317): {"baseball bat": 0.4, "sports ball": 0.3, "baseball glove": 0.3},
    # img 34785
    (34785, 80152): {"hot dog": 0.4, "cup": 0.3},
    # img 34800
    (34800, 434576): {"skateboard": 0.4},
    # img 34815
    (34815, 124972): {"person": 0.4, "train": 0.4},
    (34815, 126702): {"person": 0.4, "train": 0.4},
    # img 34816
    (34816, 1226956): {"tie": 0.3, "tv": 0.3},
    # img 34820
    (34820, 396635): {"person": 0.3, "bus": 0.3, "traffic light": 0.3},
    # img 34854
    (34854, 1553734): {"bowl": 0.3},
    # img 34861
    (34861, 53943): {"person": 0.4},
    (34861, 199895): {"horse": 0.4},
    # img 34874
    (34874, 421534): {"remote": 0.4, "chair": 0.3},
    # img 34882
    (34882, 463540): {"baseball glove": 0.4, "sports ball": 0.3},
    # img 34904
    (34904, 571603): {"book": 0.4},
    # img 34930
    (34930, 141401): {"person": 0.3, "umbrella": 0.3, "truck": 0.3},
    (34930, 1797928): {"person": 0.3, "umbrella": 0.3, "car": 0.3},
    # img 34938
    (34938, 314736): {"suitcase": 0.4},
    # img 34993
    (34993, 29661): {"tv": 0.5, "camera": 0.3},
    # img 35069
    (35069, 470535): {"toothbrush": 0.4},
    # img 35074
    (35074, 1124683): {"cat": 0.5},
    # img 35107
    (35107, 1063169): {"bowl": 0.3},
    # img 35127
    (35127, 1099362): {"cup": 0.3, "book": 0.3, "keyboard": 0.3},
    # img 35146
    (35146, 1073327): {"bowl": 0.3, "fork": 0.3, "knife": 0.3},
    # img 35190
    (35190, 1534146): {"oven": 0.3},
    # img 35211
    (35211, 213569): {"dog": 0.4, "tennis racket": 0.4, "sports ball": 0.3, "chair": 0.3},
    # img 35240
    (35240, 175725): {"person": 0.3},
    # img 35245
    (35245, 54059): {"person": 0.4},
    (35245, 56427): {"person": 0.4},
    # img 35248
    (35248, 34231): {"person": 0.4, "remote": 0.4},
    # img 35269
    (35269, 309857): {"bowl": 0.3, "fork": 0.3, "bottle": 0.3},
    # img 35316
    (35316, 427732): {"tie": 0.4},
    # img 35361
    (35361, 478586): {"toothbrush": 0.4, "sink": 0.3},
    # img 35368
    (35368, 1046395): {"orange": 0.3, "bowl": 0.3},
    # img 35407
    (35407, 312736): {"cup": 0.3, "bowl": 0.3},
    # img 35508
    (35508, 1076771): {"fork": 0.3, "knife": 0.3},
    # img 35567
    (35567, 445218): {"pizza": 0.3, "bowl": 0.3, "spoon": 0.3},
    # img 35664
    (35664, 315443): {"chair": 0.3, "couch": 0.3},
    # img 35668
    (35668, 1643038): {"vase": 0.4},
    # img 35741
    (35741, 1056099): {"person": 0.3, "orange": 0.3, "apple": 0.3},
    # img 35825
    (35825, 1930506): {"chair": 0.4, "person": 0.3, "remote": 0.3},
    # img 35864
    (35864, 31441): {"laptop": 0.4, "keyboard": 0.3, "mouse": 0.3, "chair": 0.3},
    # img 35964
    (35964, 337733): {"person": 0.3},
    # img 35975
    (35975, 89212): {"pizza": 0.4},
    # img 36059
    (36059, 1047507): {"knife": 0.4},
    # img 36157
    (36157, 622803): {"person": 0.4, "tie": 0.3},
    # img 36204
    (36204, 1049270): {"person": 0.4},
    # img 36262
    (36262, 1047965): {"orange": 0.4, "bowl": 0.3, "cup": 0.3},
    # img 36322
    (36322, 1078186): {"person": 0.4},
    # img 36349
    (36349, 2036422): {"truck": 0.3, "carrot": 0.4},
    # img 36417
    (36417, 1044713): {"apple": 0.4, "bottle": 0.3, "cell phone": 0.3},
    # img 36484
    (36484, 1105328): {"laptop": 0.3, "keyboard": 0.3},
    # img 36569
    (36569, 38239): {"person": 0.4, "apple": 0.3},
    # img 37089
    (37089, 312365): {"bowl": 0.3, "bottle": 0.3},
    # img 37160
    (37160, 274452): {"person": 0.3},
    # img 37210
    (37210, 1056010): {"bowl": 0.3, "fork": 0.3},
    # img 37394
    (37394, 311623): {"cup": 0.3, "bottle": 0.3},
    # img 37616
    (37616, 1978621): {"couch": 0.3, "chair": 0.3},
    # img 37697
    (37697, 902000037697): {"person": 0.3, "dog": 0.3},
}


def apply_corrections(desc_dir, dry_run=False):
    # Group all corrections by image_id
    corrections_by_img = defaultdict(dict)

    for (img_id, ann_id), (new_cls, new_conf) in BEST_CLASS_OVERRIDES.items():
        corrections_by_img[img_id].setdefault(ann_id, {})
        corrections_by_img[img_id][ann_id]["best_class"] = new_cls
        corrections_by_img[img_id][ann_id]["confidence"] = new_conf

    for (img_id, ann_id), new_conf in CONFIDENCE_OVERRIDES.items():
        corrections_by_img[img_id].setdefault(ann_id, {})
        corrections_by_img[img_id][ann_id]["confidence"] = new_conf

    for (img_id, ann_id), additions in TOP5_ADDITIONS.items():
        corrections_by_img[img_id].setdefault(ann_id, {})
        corrections_by_img[img_id][ann_id]["top5_add"] = additions

    stats = {
        "files_modified": 0,
        "best_class_changed": 0,
        "confidence_changed": 0,
        "top5_added": 0,
        "ann_not_found": [],
    }

    for img_id, ann_corrections in sorted(corrections_by_img.items()):
        fpath = os.path.join(desc_dir, f"{img_id}.json")
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found")
            continue

        with open(fpath) as f:
            data = json.load(f)

        modified = False
        ann_id_map = {a["ann_id"]: a for a in data["annotations"]}

        for ann_id, corr in ann_corrections.items():
            if ann_id not in ann_id_map:
                stats["ann_not_found"].append((img_id, ann_id))
                print(f"WARNING: ann_id {ann_id} not found in {fpath}")
                continue

            ann = ann_id_map[ann_id]
            vlm = ann.get("vlm", {})

            # Apply best_class override
            if "best_class" in corr:
                old_cls = vlm.get("best_class")
                new_cls = corr["best_class"]
                if old_cls != new_cls:
                    if dry_run:
                        print(f"  WOULD change best_class: {old_cls} → {new_cls} "
                              f"(img={img_id} ann={ann_id})")
                    else:
                        vlm["best_class"] = new_cls
                    stats["best_class_changed"] += 1
                    modified = True

            # Apply confidence override
            if "confidence" in corr:
                old_conf = vlm.get("confidence")
                new_conf = corr["confidence"]
                if old_conf != new_conf:
                    if dry_run:
                        print(f"  WOULD change confidence: {old_conf} → {new_conf} "
                              f"(img={img_id} ann={ann_id})")
                    else:
                        vlm["confidence"] = new_conf
                    stats["confidence_changed"] += 1
                    modified = True

            # Apply top5 additions
            if "top5_add" in corr:
                top5 = vlm.get("top5_similar", {})
                for cls_name, score in corr["top5_add"].items():
                    if cls_name not in top5 or top5[cls_name] < score:
                        if dry_run:
                            old_val = top5.get(cls_name, None)
                            print(f"  WOULD add top5: {cls_name}={score} "
                                  f"(was {old_val}) (img={img_id} ann={ann_id})")
                        else:
                            top5[cls_name] = score
                        stats["top5_added"] += 1
                        modified = True
                if not dry_run:
                    vlm["top5_similar"] = top5

            # Mark as human-reviewed
            if not dry_run:
                vlm["human_reviewed"] = True

        if modified:
            stats["files_modified"] += 1
            if not dry_run:
                with open(fpath, "w") as f:
                    json.dump(data, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Apply review corrections to VLM descriptions")
    parser.add_argument("--desc-dir", default=DESC_DIR, help="Description JSON directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    print(f"Corrections to apply:")
    print(f"  best_class overrides: {len(BEST_CLASS_OVERRIDES)}")
    print(f"  confidence overrides: {len(CONFIDENCE_OVERRIDES)}")
    print(f"  top5 additions:       {len(TOP5_ADDITIONS)} annotations")
    print()

    stats = apply_corrections(args.desc_dir, dry_run=args.dry_run)

    print(f"\n{'DRY RUN ' if args.dry_run else ''}Results:")
    print(f"  Files modified:       {stats['files_modified']}")
    print(f"  best_class changed:   {stats['best_class_changed']}")
    print(f"  confidence changed:   {stats['confidence_changed']}")
    print(f"  top5 entries added:   {stats['top5_added']}")
    if stats["ann_not_found"]:
        print(f"  Annotations not found: {stats['ann_not_found']}")


if __name__ == "__main__":
    main()
