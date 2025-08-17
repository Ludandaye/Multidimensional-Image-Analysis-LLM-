#!/bin/bash

echo "🚀 GPT因果语言模型训练项目快速启动"
echo "====================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU训练（速度较慢）"
fi

# 安装依赖
echo "📦 安装依赖包..."
pip3 install -r requirements.txt

# 检查数据文件
if [ ! -f "generated_sequences_super_enhanced/sequences_labels_fixed.jsonl" ]; then
    echo "❌ 错误: 未找到训练数据文件"
    echo "请确保数据文件存在于正确路径"
    exit 1
fi

if [ ! -f "generated_sequences_super_enhanced/vocab.json" ]; then
    echo "❌ 错误: 未找到词汇表文件"
    echo "请确保词汇表文件存在于正确路径"
    exit 1
fi

echo "✅ 数据文件检查通过"

# 开始训练
echo "🎯 开始训练GPT因果语言模型..."
echo "训练参数:"
echo "  - 批次大小: 8"
echo "  - 训练轮数: 20"
echo "  - 学习率: 5e-5"
echo "  - 最大长度: 1024"
echo "  - 任务类型: 因果语言建模 (next-token预测)"
echo ""

python3 train_gpt.py \
    --data_path generated_sequences_super_enhanced/sequences_labels_fixed.jsonl \
    --vocab_path generated_sequences_super_enhanced/vocab.json \
    --batch_size 8 \
    --num_epochs 20 \
    --learning_rate 5e-5 \
    --max_length 1024

echo ""
echo "🎉 训练完成！"
echo "模型已保存为HuggingFace标准格式:"
echo "  - outputs/best_model/ (最佳模型)"
echo "  - outputs/final_model/ (最终模型)"
echo ""
echo "🚀 现在可以使用 from_pretrained() 直接加载模型！"
echo ""

# 询问是否进行推理测试
read -p "是否现在测试模型？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 开始模型推理测试..."
    python3 inference.py --model_path outputs/best_model
fi

echo ""
echo "✨ 项目完成！"
echo "查看 README.md 了解更多使用方法"
