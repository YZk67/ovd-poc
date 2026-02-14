#!/bin/bash
# 一键复制所有必需文件到正确位置
# 运行: chmod +x copy_files.sh && ./copy_files.sh

set -e  # 遇到错误立即退出

BASE_DIR="/Users/zhengjiankang/Downloads/research/research/ovd-poc"
cd "$BASE_DIR"

echo "=================================================="
echo "开始复制文件到正确位置..."
echo "=================================================="

# 1. 类别定义文件
echo ""
echo "📝 1. 复制类别定义文件..."
if [ -f dataset2/metadata/ovcoco_seen_classes.json ]; then
    cp dataset2/metadata/ovcoco_seen_classes.json dataset/metadata/
    echo "   ✅ ovcoco_seen_classes.json"
else
    echo "   ⚠️  源文件不存在: dataset2/metadata/ovcoco_seen_classes.json"
fi

if [ -f dataset2/metadata/ovcoco_all_classes.json ]; then
    cp dataset2/metadata/ovcoco_all_classes.json dataset/metadata/
    echo "   ✅ ovcoco_all_classes.json"
else
    echo "   ⚠️  源文件不存在: dataset2/metadata/ovcoco_all_classes.json"
fi

# 2. 文本嵌入文件
echo ""
echo "📊 2. 复制文本嵌入文件..."
if [ -f dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy ]; then
    cp dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy \
       dataset/metadata/ovdcoco_prompts_list8_v2.npy
    echo "   ✅ ovdcoco_prompts_list8_v2.npy"
    
    cp dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy \
       dataset/metadata/ovdcoco_vlm_query_convnextl.npy
    echo "   ✅ ovdcoco_vlm_query_convnextl.npy"
else
    echo "   ⚠️  源文件不存在: dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy"
    echo "   💡 你可能需要生成这些文件，参见 FILE_ORGANIZATION_GUIDE.md"
fi

# 3. CLIP 预训练模型
echo ""
echo "🤖 3. 复制 CLIP 预训练模型..."
if [ -f pretrained_models2/clip_convnext_large_head.pth ]; then
    cp pretrained_models2/clip_convnext_large_head.pth pretrained_models/
    echo "   ✅ clip_convnext_large_head.pth"
else
    echo "   ⚠️  源文件不存在: pretrained_models2/clip_convnext_large_head.pth"
fi

# 4. COCO 标注文件
echo ""
echo "📋 4. 复制 COCO 标注文件..."
if [ -f dataset2/coco/annotations/ovd_ins_train2017_b.json ]; then
    cp dataset2/coco/annotations/ovd_ins_train2017_b.json \
       dataset/coco/annotations/
    echo "   ✅ ovd_ins_train2017_b.json"
else
    echo "   ⚠️  源文件不存在: dataset2/coco/annotations/ovd_ins_train2017_b.json"
fi

if [ -f dataset2/coco/annotations/ovd_ins_val2017_all.json ]; then
    cp dataset2/coco/annotations/ovd_ins_val2017_all.json \
       dataset/coco/annotations/
    echo "   ✅ ovd_ins_val2017_all.json"
else
    echo "   ⚠️  源文件不存在: dataset2/coco/annotations/ovd_ins_val2017_all.json"
fi

echo ""
echo "=================================================="
echo "✅ 文件复制完成！"
echo "=================================================="

# 验证文件
echo ""
echo "🔍 验证已复制的文件..."
echo ""

check_file() {
    if [ -f "$1" ]; then
        size=$(du -h "$1" | cut -f1)
        echo "✅ $1 ($size)"
        return 0
    else
        echo "❌ $1 [缺失]"
        return 1
    fi
}

all_good=true

check_file "dataset/metadata/ovcoco_seen_classes.json" || all_good=false
check_file "dataset/metadata/ovcoco_all_classes.json" || all_good=false
check_file "dataset/metadata/ovdcoco_prompts_list8_v2.npy" || all_good=false
check_file "dataset/metadata/ovdcoco_vlm_query_convnextl.npy" || all_good=false
check_file "pretrained_models/clip_convnext_large_head.pth" || all_good=false
check_file "dataset/coco/annotations/ovd_ins_train2017_b.json" || all_good=false
check_file "dataset/coco/annotations/ovd_ins_val2017_all.json" || all_good=false

echo ""
if [ "$all_good" = true ]; then
    echo "🎉 所有必需文件都已就绪！"
    echo ""
    echo "📝 下一步："
    echo "   1. 如果还没有训练模型，运行训练:"
    echo "      python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py --num-gpus 4"
    echo ""
    echo "   2. 训练完成后，运行可视化:"
    echo "      python tools/vis_airplane.py --config ... --weights output/model_final.pth --input ... --output ..."
else
    echo "⚠️  仍有文件缺失，请检查上面的错误信息"
    echo "💡 详细说明请查看: FILE_ORGANIZATION_GUIDE.md"
fi

echo ""
echo "=================================================="
