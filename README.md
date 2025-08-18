# 多维图片分析LLM项目

## 项目简介

本项目使用 HuggingFace Transformers 从零训练一个小型 GPT 语言模型，实现**因果语言模型（Causal LM）**任务：给定前文tokens序列，预测下一个token。项目包含两次训练版本，已按训练轮次归档。

## 🏗️ 项目结构

```
.
├── training_v1/                    # 第一次训练版本
│   ├── train_gpt.py               # 主训练脚本
│   ├── inference.py               # 推理脚本（已修复）
│   ├── requirements.txt           # 依赖包列表
│   ├── config.yaml               # 训练配置文件
│   ├── README.md                 # 详细项目说明
│   ├── outputs/                  # 训练好的模型
│   │   ├── best_model/          # 最佳验证性能的模型
│   │   └── final_model/         # 最终训练完成的模型
│   ├── generated_sequences_super_enhanced/
│   │   ├── sequences_labels_fixed.jsonl  # 训练数据
│   │   └── vocab.json           # 词汇表
│   ├── training_plots/          # 训练过程可视化
│   └── unified_codebook/        # 码本与相关统计
│
└── training_v2/                    # 第二次训练版本（数据修正版）
    └── generated_sequences_super_enhanced/
        └── sequences_labels_fixed_tail_fixed.jsonl  # 修正后的训练数据
```

## 🎯 主要任务：因果语言模型（Causal LM）

- **目标**: 给定前文，预测下一个token（逐位置右移标签）
- **适用场景**: 模型可以"续写/生成"token序列
- **训练方式**: LM头 + 交叉熵损失，对每个位置进行监督（pad位置设label=-100）
- **评估指标**: Perplexity (PPL)、next-token accuracy
- **推理方式**: 自回归解码（greedy/top-k/nucleus sampling）

## 🔄 第二次训练（数据修正点）

### 主要改进
- **尾部格式对齐分类监督**: 由"`… <CLS> <CLS> <EOS>`"修正为"`… <CLS> <CLS_{label}> <EOS>`"
- **长度与截断策略**: 最大长度 512，永远保留结尾"`</IMG> <CLS> <CLS_{label}> <EOS>`"
- **词表与tokenizer一致**: `<CLS>`、`<EOS>`、`<CLS_0>…<CLS_9>` 均存在且ID稳定

### 使用修正数据进行训练

```bash
# 进入training_v1目录
cd training_v1

# 使用修正后的数据进行第二次训练
python train_gpt.py \
  --data_path ../training_v2/generated_sequences_super_enhanced/sequences_labels_fixed_tail_fixed.jsonl \
  --vocab_path generated_sequences_super_enhanced/vocab.json \
  --batch_size 16 \
  --num_epochs 30 \
  --learning_rate 1e-4 \
  --max_length 512
```

### 训练要点
- 训练输入到 `<CLS>` 截止，模型学习预测下一token为 `<CLS_{label}>`
- 评估时取 `<CLS>` 后一步的分布，仅在 `<CLS_0>…<CLS_9>` 上取最大作为类别
- 训练/验证需保持相同的 `max_length`、截断与padding策略

## 🚀 快速开始

### 1. 安装依赖

```bash
cd training_v1
pip install -r requirements.txt
```

### 2. 使用已训练模型进行推理

```bash
cd training_v1
python inference.py --model_path outputs/best_model --mode generate
```

### 3. 重新训练模型

```bash
cd training_v1
python train_gpt.py \
  --data_path ../training_v2/generated_sequences_super_enhanced/sequences_labels_fixed_tail_fixed.jsonl \
  --vocab_path generated_sequences_super_enhanced/vocab.json \
  --batch_size 16 \
  --num_epochs 30 \
  --learning_rate 1e-4 \
  --max_length 512
```

## 🏆 模型性能

### 第一次训练结果
- **训练损失**: 0.0095
- **验证损失**: 0.0100  
- **训练困惑度 (PPL)**: 1.01
- **验证困惑度 (PPL)**: 1.01
- **总改善率**: 99.38%

### 模型架构
- **模型类型**: GPT-2 架构（因果语言模型）
- **层数**: 6层 Transformer
- **注意力头数**: 8头
- **嵌入维度**: 384
- **词汇表大小**: 516 tokens
- **参数数量**: ~850万

## 📊 训练数据

- **数据格式**: JSONL文件
- **词汇表**: JSON格式，516个独特token
- **序列长度**: 最大1024 tokens
- **数据划分**: 80%训练，20%验证

## 🔧 环境要求

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.41+
- CUDA支持（可选，用于GPU加速）
- 支持 bf16 混合精度训练

## 📁 输出文件

训练完成后会生成HuggingFace标准格式的模型：

```
outputs/
├── best_model/           # 最佳验证性能的模型
│   ├── config.json       # 模型配置
│   ├── pytorch_model.bin # 模型权重
│   ├── tokenizer.json    # tokenizer配置
│   ├── special_tokens_map.json
│   └── training_args.bin # 训练状态
└── final_model/          # 最终训练完成的模型
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── special_tokens_map.json
```

## 🌐 模型发布

模型已发布到Hugging Face Hub：
**https://huggingface.co/ludandaye/gpt-causal-lm**

### 使用方法

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载模型和tokenizer
model = GPT2LMHeadModel.from_pretrained("ludandaye/gpt-causal-lm")
tokenizer = GPT2Tokenizer.from_pretrained("ludandaye/gpt-causal-lm")

# 设置为评估模式
model.eval()
```

## 📝 注意事项

1. 确保有足够的GPU内存（建议8GB+）
2. 训练时间约1-3小时（取决于硬件）
3. 模型会自动保存最佳检查点
4. 支持CPU训练（但速度较慢）

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**: 减小batch_size
2. **训练不收敛**: 调整学习率或增加训练轮数
3. **数据加载错误**: 检查文件路径和格式

### 性能优化

1. 使用混合精度训练（FP16）
2. 启用梯度累积
3. 使用多GPU训练

## 📚 扩展功能

- 支持自定义模型配置
- 可添加更多评估指标（BLEU、ROUGE等）
- 支持模型导出为ONNX格式
- 可集成到Web服务中
- **文本生成**: 支持不同采样策略（greedy、top-k、nucleus）
- **序列续写**: 给定前缀，自动生成后续内容
- **模型微调**: 支持在其他领域数据上继续训练

## 📄 许可证

本项目仅供学习和研究使用。

## 👨‍💻 作者

[ludandaye](https://huggingface.co/ludandaye)

## 📖 详细文档

- [第一次训练详细说明](training_v1/README.md)
- [训练过程可视化](training_v1/training_plots/)
- [项目总结](training_v1/PROJECT_SUMMARY.md)


