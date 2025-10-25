#!/bin/bash

# 项目打包脚本 (使用zip)
# 排除 dataset2 和 pretrained_models2 文件夹

# 设置项目根目录
PROJECT_ROOT="/Users/zhengjiankang/Downloads/research/research/ovd-poc"
PACKAGE_NAME="ovd-poc-$(date +%Y%m%d_%H%M%S).zip"

# 进入项目根目录
cd "$PROJECT_ROOT"

# 创建临时目录，复制需要的文件
TEMP_DIR="temp_package"
mkdir -p "$TEMP_DIR"

# 复制文件，排除不需要的文件夹
rsync -av --exclude="dataset2" \
          --exclude="pretrained_models2" \
          --exclude=".venv" \
          --exclude=".git" \
          --exclude="__pycache__" \
          --exclude="*.pyc" \
          --exclude=".DS_Store" \
          --exclude="*.log" \
          --exclude="output" \
          --exclude="*.tar.gz" \
          --exclude="*.zip" \
          --exclude="$TEMP_DIR" \
          . "$TEMP_DIR/"

# 创建zip文件
cd "$TEMP_DIR"
zip -r "../$PACKAGE_NAME" . -x "*.tar.gz" "*.zip"
cd ..

# 清理临时目录
rm -rf "$TEMP_DIR"

echo "项目已打包为: $PACKAGE_NAME"
echo "文件大小: $(du -h "$PACKAGE_NAME" | cut -f1)"
echo "排除的文件夹: dataset2, pretrained_models2, .venv, .git, __pycache__, output"
