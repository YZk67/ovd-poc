#!/usr/bin/env python3
"""
Helper script to generate high-quality prompts for LVIS classes.
This script creates a template that I (Claude) will fill in.
"""

import json
from pathlib import Path

# Load the ordered class list
with open('dataset2/metadata/lvis_prompts.json', 'r') as f:
    existing_prompts = json.load(f)
    
classes = list(existing_prompts.keys())
print(f"Total classes to generate: {len(classes)}")
print(f"First 10: {classes[:10]}")
print(f"Last 10: {classes[-10:]}")

# Save ordered list for reference
with open('/tmp/all_classes_ordered.txt', 'w') as f:
    for i, cls in enumerate(classes, 1):
        f.write(f"{i}. {cls}\n")

print(f"\n✓ Saved class list to /tmp/all_classes_ordered.txt")
print(f"\nReady to generate {len(classes)} classes with 8 prompts each")

