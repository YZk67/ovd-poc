#!/bin/bash

# 项目打包脚本
# 排除 dataset2 和 pretrained_models2 文件夹

# 设置项目根目录
PROJECT_ROOT="/Users/zhengjiankang/Downloads/research/research/ovd-poc"
PACKAGE_NAME="ovd-poc-$(date +%Y%m%d_%H%M%S).tar.gz"

# 进入项目根目录
cd "$PROJECT_ROOT"

# 创建打包文件，排除指定文件夹
tar -czf "$PACKAGE_NAME" \
    --exclude="dataset2" \
    --exclude="pretrained_models2" \
    --exclude=".venv" \
    --exclude=".git" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".DS_Store" \
    --exclude="*.log" \
    --exclude="output" \
    --exclude="*.tar.gz" \
    .

echo "项目已打包为: $PACKAGE_NAME"
echo "文件大小: $(du -h "$PACKAGE_NAME" | cut -f1)"
echo "排除的文件夹: dataset2, pretrained_models2, .venv, .git, __pycache__, output"
