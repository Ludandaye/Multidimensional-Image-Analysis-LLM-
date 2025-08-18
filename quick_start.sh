#!/bin/bash

# 多维图片分析LLM项目 - 快速启动脚本
# 使用方法: ./quick_start.sh [选项]

echo "🚀 多维图片分析LLM项目 - 快速启动"
echo "=================================="

# 检查是否在正确的目录
if [ ! -d "training_v1" ] || [ ! -d "training_v2" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    exit 1
fi

# 显示菜单
show_menu() {
    echo ""
    echo "请选择要执行的操作："
    echo "1) 安装依赖包"
    echo "2) 使用已训练模型进行推理"
    echo "3) 使用修正数据重新训练模型"
    echo "4) 查看项目结构"
    echo "5) 查看训练结果"
    echo "6) 测试模型加载"
    echo "0) 退出"
    echo ""
    read -p "请输入选项 (0-6): " choice
}

# 安装依赖
install_dependencies() {
    echo "📦 安装依赖包..."
    cd training_v1
    pip install -r requirements.txt
    cd ..
    echo "✅ 依赖安装完成"
}

# 推理
run_inference() {
    echo "🤖 启动推理模式..."
    cd training_v1
    if [ -d "outputs/best_model" ]; then
        echo "使用已训练的模型进行推理..."
        python inference.py --model_path outputs/best_model --mode generate
    else
        echo "❌ 未找到训练好的模型，请先训练模型"
    fi
    cd ..
}

# 重新训练
retrain_model() {
    echo "🏋️ 使用修正数据重新训练模型..."
    cd training_v1
    
    # 检查数据文件是否存在
    if [ ! -f "../training_v2/generated_sequences_super_enhanced/sequences_labels_fixed_tail_fixed.jsonl" ]; then
        echo "❌ 未找到修正后的训练数据"
        cd ..
        return
    fi
    
    echo "开始训练..."
    python train_gpt.py \
        --data_path ../training_v2/generated_sequences_super_enhanced/sequences_labels_fixed_tail_fixed.jsonl \
        --vocab_path generated_sequences_super_enhanced/vocab.json \
        --batch_size 16 \
        --num_epochs 30 \
        --learning_rate 1e-4 \
        --max_length 512
    
    cd ..
}

# 查看项目结构
show_structure() {
    echo "📁 项目结构："
    echo ""
    tree -L 3 -I '__pycache__|*.pyc|.git' || find . -type d -maxdepth 3 | head -20
}

# 查看训练结果
show_results() {
    echo "📊 训练结果："
    echo ""
    if [ -d "training_v1/outputs" ]; then
        echo "模型输出目录："
        ls -la training_v1/outputs/
        echo ""
        if [ -d "training_v1/training_plots" ]; then
            echo "训练图表："
            ls -la training_v1/training_plots/
        fi
    else
        echo "❌ 未找到训练输出目录"
    fi
}

# 测试模型加载
test_model() {
    echo "🧪 测试模型加载..."
    cd training_v1
    
    if [ -d "outputs/best_model" ]; then
        echo "测试加载本地模型..."
        python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
try:
    model = GPT2LMHeadModel.from_pretrained('outputs/best_model')
    tokenizer = GPT2Tokenizer.from_pretrained('outputs/best_model')
    print('✅ 本地模型加载成功！')
except Exception as e:
    print(f'❌ 本地模型加载失败: {e}')
"
    else
        echo "❌ 未找到本地模型"
    fi
    
    echo "测试加载Hugging Face模型..."
    python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
try:
    model = GPT2LMHeadModel.from_pretrained('ludandaye/gpt-causal-lm')
    tokenizer = GPT2Tokenizer.from_pretrained('ludandaye/gpt-causal-lm')
    print('✅ Hugging Face模型加载成功！')
except Exception as e:
    print(f'❌ Hugging Face模型加载失败: {e}')
"
    
    cd ..
}

# 主循环
while true; do
    show_menu
    
    case $choice in
        1)
            install_dependencies
            ;;
        2)
            run_inference
            ;;
        3)
            retrain_model
            ;;
        4)
            show_structure
            ;;
        5)
            show_results
            ;;
        6)
            test_model
            ;;
        0)
            echo "👋 再见！"
            exit 0
            ;;
        *)
            echo "❌ 无效选项，请重新选择"
            ;;
    esac
    
    echo ""
    read -p "按回车键继续..."
done
