#!/bin/bash

# Script to move checkpoints to /root/autodl-tmp/ to save space
# Usage: ./move_checkpoints.sh

SOURCE_DIR="/root/LaMI-DETR/output/lami_convnext_large_12ep_lvis"
TARGET_DIR="/root/autodl-tmp/lami_convnext_large_12ep_lvis"

echo "Moving checkpoints to save space..."

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Move all .pth files (checkpoints)
echo "Moving checkpoint files..."
find "$SOURCE_DIR" -name "*.pth" -exec mv {} "$TARGET_DIR/" \;

# Create symbolic links back to original location
echo "Creating symbolic links..."
cd "$SOURCE_DIR"
for file in "$TARGET_DIR"/*.pth; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        ln -sf "$file" "$filename"
        echo "Created symlink for $filename"
    fi
done

echo "Checkpoint migration completed!"
echo "Checkpoints are now stored in: $TARGET_DIR"
echo "Symlinks created in: $SOURCE_DIR"

# Show space saved
echo ""
echo "Space usage:"
echo "Source directory:"
du -sh "$SOURCE_DIR"
echo "Target directory:"
du -sh "$TARGET_DIR"

