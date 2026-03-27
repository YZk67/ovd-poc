#!/usr/bin/env python3
"""Generate visual attributes for COCO 80 classes without GPT API.

Uses a curated template-based approach to generate attribute descriptions
for each class based on their super-category and visual properties.
"""

import json
from pathlib import Path

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Visual property templates - applied per class with class-specific details
SHAPE_ATTRS = [
    "has a round shape", "has a rectangular shape", "has an elongated shape",
    "has a cylindrical shape", "has a flat shape", "has an irregular shape",
    "has a boxy shape", "has a curved shape", "has a pointed shape",
    "has a spherical shape", "has a triangular shape", "has a disc shape",
]

MATERIAL_ATTRS = [
    "made of metal", "made of plastic", "made of wood", "made of glass",
    "made of fabric", "made of rubber", "made of ceramic", "made of leather",
    "made of paper", "made of foam", "made of stone", "made of concrete",
]

TEXTURE_ATTRS = [
    "has a smooth surface", "has a rough surface", "has a shiny surface",
    "has a matte surface", "has a furry texture", "has a woven texture",
    "has a glossy finish", "has a transparent surface", "has a patterned surface",
    "has a striped pattern", "has a spotted pattern", "has a solid color",
]

SIZE_ATTRS = [
    "is very small", "is small", "is medium sized", "is large", "is very large",
    "fits in one hand", "is person-height", "is taller than a person",
    "is wider than it is tall", "is taller than it is wide",
]

PART_ATTRS = [
    "has wheels", "has legs", "has a handle", "has a screen", "has buttons",
    "has a door", "has windows", "has a lid or cover", "has a seat",
    "has a pointed end", "has an opening", "has a base that sits flat",
    "has a frame", "has a strap", "has a blade or edge", "has a cord or wire",
    "has a motor or engine", "has hinges", "has a reflective surface",
    "has a container or cavity",
]

CONTEXT_ATTRS = [
    "found indoors", "found outdoors", "found in kitchen", "found on roads",
    "found in sky", "found on water", "found on tables", "found on ground",
    "found on walls", "held by person", "worn by person", "carried by person",
    "found in bathroom", "found in bedroom", "found in office",
]

FUNCTION_ATTRS = [
    "used for transportation", "used for eating", "used for sitting",
    "used for cutting", "used for cooking", "used for communication",
    "used for sports", "used for decoration", "used for storage",
    "used for cleaning", "used for entertainment", "used for drinking",
    "is alive or organic", "produces light", "produces sound",
    "contains liquid", "is edible", "is wearable",
]

ANIMAL_ATTRS = [
    "has four legs", "has two legs", "has wings", "has a tail",
    "has fur or hair", "has feathers", "has scales", "has a beak",
    "has horns or antlers", "has hooves", "has claws or paws",
    "has a snout or muzzle", "has a long neck", "has stripes",
    "has spots or patches", "has a mane", "has tusks",
    "is a domestic animal", "is a wild animal", "has ears that stick up",
]

# Class-to-supercategory mapping for targeted attribute assignment
SUPER_CATEGORIES = {
    "person": "person",
    "bicycle": "vehicle", "car": "vehicle", "motorcycle": "vehicle",
    "airplane": "vehicle", "bus": "vehicle", "train": "vehicle",
    "truck": "vehicle", "boat": "vehicle",
    "traffic light": "outdoor", "fire hydrant": "outdoor",
    "stop sign": "outdoor", "parking meter": "outdoor", "bench": "outdoor",
    "bird": "animal", "cat": "animal", "dog": "animal", "horse": "animal",
    "sheep": "animal", "cow": "animal", "elephant": "animal", "bear": "animal",
    "zebra": "animal", "giraffe": "animal",
    "backpack": "accessory", "umbrella": "accessory", "handbag": "accessory",
    "tie": "accessory", "suitcase": "accessory",
    "frisbee": "sports", "skis": "sports", "snowboard": "sports",
    "sports ball": "sports", "kite": "sports", "baseball bat": "sports",
    "baseball glove": "sports", "skateboard": "sports", "surfboard": "sports",
    "tennis racket": "sports",
    "bottle": "kitchen", "wine glass": "kitchen", "cup": "kitchen",
    "fork": "kitchen", "knife": "kitchen", "spoon": "kitchen", "bowl": "kitchen",
    "banana": "food", "apple": "food", "sandwich": "food", "orange": "food",
    "broccoli": "food", "carrot": "food", "hot dog": "food", "pizza": "food",
    "donut": "food", "cake": "food",
    "chair": "furniture", "couch": "furniture", "potted plant": "furniture",
    "bed": "furniture", "dining table": "furniture",
    "toilet": "furniture", "tv": "electronic", "laptop": "electronic",
    "mouse": "electronic", "remote": "electronic", "keyboard": "electronic",
    "cell phone": "electronic", "microwave": "appliance", "oven": "appliance",
    "toaster": "appliance", "sink": "appliance", "refrigerator": "appliance",
    "book": "indoor", "clock": "indoor", "vase": "indoor",
    "scissors": "indoor", "teddy bear": "indoor", "hair drier": "indoor",
    "toothbrush": "indoor",
}

# Per-class specific attributes (most distinctive visual features)
CLASS_SPECIFIC = {
    "person": ["walks upright", "has a face", "wears clothing", "has hands", "bipedal posture"],
    "bicycle": ["has two wheels", "has pedals", "has handlebars", "has a chain", "has spokes"],
    "car": ["has four wheels", "has headlights", "has a windshield", "has doors", "has a license plate"],
    "motorcycle": ["has two wheels", "has an engine visible", "has a fuel tank", "has exhaust pipes"],
    "airplane": ["has wings", "has a fuselage", "has a tail fin", "has jet engines", "very large"],
    "bus": ["has many windows in rows", "has a flat front", "very large", "has a destination sign"],
    "train": ["runs on rails", "has multiple carriages", "very long", "has a locomotive"],
    "truck": ["has a cargo area", "has a cab", "larger than car", "has dual rear wheels sometimes"],
    "boat": ["has a hull", "floats on water", "has a deck", "has a bow and stern"],
    "traffic light": ["has three colored lights", "red yellow green", "mounted on pole", "vertical box shape"],
    "fire hydrant": ["short and stout", "red or yellow color", "has outlet nozzles", "barrel shape"],
    "stop sign": ["octagonal shape", "red with white text", "says STOP", "reflective surface"],
    "parking meter": ["has a coin slot", "has a display", "mounted on post", "about waist height"],
    "bench": ["has a seat and backrest", "has slats", "seats multiple people", "found in parks"],
    "bird": ["has feathers", "has a beak", "has two wings", "can fly", "has two legs"],
    "cat": ["has pointed ears", "has whiskers", "has a long tail", "has retractable claws", "has fur"],
    "dog": ["has a snout", "has a tail", "has floppy or pointed ears", "has paws", "has fur"],
    "horse": ["has a mane", "has hooves", "large muscular body", "has a long face", "can be ridden"],
    "sheep": ["has woolly fleece", "has a stocky body", "white colored usually", "found in flocks"],
    "cow": ["has a large body", "has a udder", "black and white spots often", "has horns sometimes"],
    "elephant": ["has a long trunk", "has large floppy ears", "has tusks", "gray wrinkled skin", "very large"],
    "bear": ["has thick fur", "has a broad head", "has large paws with claws", "large muscular body"],
    "zebra": ["has black and white stripes", "horse-like body", "has a mane", "has hooves"],
    "giraffe": ["has a very long neck", "has spots", "has ossicones", "very tall", "long legs"],
    "backpack": ["has shoulder straps", "has a main compartment", "has zippers", "worn on back"],
    "umbrella": ["has a canopy", "has a handle", "circular when open", "has ribs and shaft"],
    "handbag": ["has a handle or strap", "has a closure", "carried by person", "holds personal items"],
    "tie": ["narrow elongated shape", "worn around neck", "has a pointed end", "has a knot"],
    "suitcase": ["has wheels", "has a telescoping handle", "rectangular shape", "has a hard or soft shell"],
    "frisbee": ["circular disc shape", "flat and thin", "plastic material", "throwable"],
    "skis": ["long narrow planks", "come in pairs", "has bindings", "has curved tips"],
    "snowboard": ["single wide plank", "has bindings on top", "has a graphic design", "has metal edges"],
    "sports ball": ["spherical shape", "bounces", "has a textured surface", "various sizes"],
    "kite": ["has a flat surface", "has a string", "flies in wind", "colorful design"],
    "baseball bat": ["elongated cylindrical", "has a handle and barrel", "made of wood or metal"],
    "baseball glove": ["has a catching pocket", "made of leather", "has finger stalls", "has webbing"],
    "skateboard": ["has a flat deck", "has four wheels", "has grip tape", "has trucks"],
    "surfboard": ["elongated streamlined shape", "has fins underneath", "floats on water"],
    "tennis racket": ["has an oval head", "has strings", "has a handle", "has a frame"],
    "bottle": ["has a cylindrical body", "has a narrow neck", "has a cap", "holds liquid"],
    "wine glass": ["has a bowl", "has a stem", "has a base", "transparent glass"],
    "cup": ["has a handle", "has a cylindrical body", "holds liquid", "ceramic or paper material"],
    "fork": ["has tines or prongs", "has a handle", "metallic material", "used for eating"],
    "knife": ["has a blade", "has a handle", "has a cutting edge", "elongated shape"],
    "spoon": ["has a bowl-shaped end", "has a handle", "concave surface", "used for scooping"],
    "bowl": ["has a concave interior", "round shape from above", "has a rim", "holds food or liquid"],
    "banana": ["curved elongated shape", "yellow when ripe", "has a peel", "grows in bunches"],
    "apple": ["round shape", "has a stem", "red green or yellow", "has a smooth skin"],
    "sandwich": ["has bread layers", "has filling visible", "rectangular or triangular", "handheld food"],
    "orange": ["round shape", "orange colored skin", "dimpled texture", "citrus fruit"],
    "broccoli": ["green color", "tree-like shape", "has florets", "has a thick stem"],
    "carrot": ["orange color", "tapered elongated shape", "has a green top", "root vegetable"],
    "hot dog": ["sausage in a bun", "elongated shape", "has condiment toppings", "handheld food"],
    "pizza": ["circular flat shape", "has cheese and toppings", "has a crust edge", "triangular slices"],
    "donut": ["ring shape with hole", "has glazed surface", "round pastry", "has sprinkles sometimes"],
    "cake": ["has layers", "has frosting", "round or rectangular", "has decorations"],
    "chair": ["has a seat", "has a backrest", "has four legs usually", "made of wood or metal"],
    "couch": ["has cushions", "has armrests", "seats multiple people", "has upholstered surface"],
    "potted plant": ["has a pot container", "has green leaves", "has soil visible", "has a growing plant"],
    "bed": ["has a mattress", "has pillows", "has bedding", "has a headboard", "large flat surface"],
    "dining table": ["has a flat top surface", "has legs", "has chairs around", "found in dining area"],
    "toilet": ["has a bowl and seat", "has a tank", "has a flush mechanism", "white porcelain"],
    "tv": ["has a flat screen", "rectangular display", "has a thin profile", "shows images"],
    "laptop": ["has a screen and keyboard", "folds open", "has a trackpad", "thin clamshell design"],
    "mouse": ["small handheld device", "has buttons", "has a scroll wheel", "connected to computer"],
    "remote": ["has many buttons", "rectangular handheld", "has IR emitter", "controls devices"],
    "keyboard": ["has many keys in rows", "has a spacebar", "rectangular flat shape", "has letters and numbers"],
    "cell phone": ["has a screen", "thin rectangular shape", "has cameras", "has a glass front"],
    "microwave": ["has a door with window", "has a control panel", "box shape", "has a turntable"],
    "oven": ["has a door with window", "has racks inside", "has temperature controls", "large box shape"],
    "toaster": ["has slots on top", "has a lever", "compact box shape", "chrome or colored"],
    "sink": ["has a basin", "has a faucet", "has a drain", "concave shape"],
    "refrigerator": ["has a door", "tall rectangular shape", "has shelves inside", "keeps things cold"],
    "book": ["has a cover and pages", "has a spine", "rectangular shape", "has text inside"],
    "clock": ["has a face with numbers", "has hands", "circular face usually", "tells time"],
    "vase": ["has a narrow neck", "has a wide body", "holds flowers", "decorative container"],
    "scissors": ["has two blades", "has finger holes", "X-shape when open", "cutting tool"],
    "teddy bear": ["has a soft plush body", "has button eyes", "has a round head", "stuffed animal"],
    "hair drier": ["has a handle and nozzle", "pistol-grip shape", "has a cord", "blows hot air"],
    "toothbrush": ["has a handle", "has bristles", "has a small head", "elongated shape"],
}


def generate_all_attributes():
    """Generate attributes for all 80 COCO classes."""
    all_class_attrs = {}

    for cls in COCO_CLASSES:
        attrs = set()

        # Add class-specific attributes
        if cls in CLASS_SPECIFIC:
            attrs.update(CLASS_SPECIFIC[cls])

        # Add super-category relevant attributes
        supercat = SUPER_CATEGORIES.get(cls, "indoor")

        if supercat == "animal":
            attrs.update(ANIMAL_ATTRS[:15])
        elif supercat == "vehicle":
            attrs.update(["used for transportation", "has wheels", "has a motor or engine",
                         "has a metallic surface", "found on roads"])
        elif supercat == "food":
            attrs.update(["is edible", "has a natural color", "is organic",
                         "found in kitchen", "found on tables", "has a natural shape"])
        elif supercat == "kitchen":
            attrs.update(["found in kitchen", "used for eating", "found on tables",
                         "has a handle", "is small to medium size"])
        elif supercat == "electronic":
            attrs.update(["has a screen", "has buttons", "has a cord or wire",
                         "made of plastic", "found on desks"])
        elif supercat == "appliance":
            attrs.update(["found in kitchen", "has a door", "has a control panel",
                         "uses electricity", "has a cord"])
        elif supercat == "furniture":
            attrs.update(["found indoors", "is large", "has legs", "made of wood",
                         "functional furniture"])
        elif supercat == "sports":
            attrs.update(["used for sports", "found outdoors", "has a handle sometimes",
                         "lightweight", "used for recreation"])
        elif supercat == "accessory":
            attrs.update(["carried by person", "is portable", "has a strap or handle",
                         "made of fabric or leather", "is personal item"])
        elif supercat == "outdoor":
            attrs.update(["found outdoors", "found on streets", "mounted or fixed position",
                         "weather resistant", "made of metal"])

        # Add some general attributes from templates
        for attr_list in [SHAPE_ATTRS, MATERIAL_ATTRS, TEXTURE_ATTRS,
                         SIZE_ATTRS, PART_ATTRS, CONTEXT_ATTRS, FUNCTION_ATTRS]:
            for attr in attr_list:
                attrs.add(attr)

        all_class_attrs[cls] = sorted(attrs)

    return all_class_attrs


def main():
    attributes = generate_all_attributes()

    # Flatten to unique attribute list
    all_attrs = []
    attr_to_classes = {}
    for cls, attrs in attributes.items():
        for attr in attrs:
            attr = attr.strip().lower()
            if attr not in attr_to_classes:
                attr_to_classes[attr] = []
                all_attrs.append(attr)
            attr_to_classes[attr].append(cls)

    print(f"Total classes: {len(attributes)}")
    print(f"Total unique attributes: {len(all_attrs)}")
    print(f"Attributes shared across 3+ classes: {sum(1 for v in attr_to_classes.values() if len(v) >= 3)}")
    print(f"Attributes shared across 10+ classes: {sum(1 for v in attr_to_classes.values() if len(v) >= 10)}")

    output = {
        "attributes": attributes,
        "all_attributes": all_attrs,
        "attribute_to_classes": attr_to_classes,
    }

    out_path = Path("tools/owovd/coco_attributes.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
