## 项目说明（两次训练归档）

本仓库已将历史内容按训练轮次归档：

- training_v1: 第一次训练的全部代码、数据与模型输出（原仓库内容）
- training_v2: 第二次训练使用的数据修正版（仅数据起步，可直接用于重新训练）

### 目录结构

```
training_v1/
  ├── train_gpt.py               # 训练脚本（可复用进行第二次训练）
  ├── generated_sequences_super_enhanced/
  │   └── vocab.json             # 词表文件（与tokenizer一致）
  ├── outputs/                   # 第一次训练产物（best_model / final_model）
  └── unified_codebook/          # 码本与相关统计

training_v2/
  └── generated_sequences_super_enhanced/
      └── sequences_labels_fixed_tail_fixed.jsonl  # 修正后的训练数据
```

### 第二次训练（数据修正点）

- 尾部格式对齐分类监督：由“`… <CLS> <CLS> <EOS>`”修正为“`… <CLS> <CLS_{label}> <EOS>`”
- 长度与截断策略：最大长度 512，永远保留结尾“`</IMG> <CLS> <CLS_{label}> <EOS>`”，如需截断从前部/中部的 Z tokens 裁剪
- 词表与 tokenizer 一致：`<CLS>`、`<EOS>`、`<CLS_0>…<CLS_9>` 均存在且 ID 稳定

### 使用修正数据进行训练（示例命令）

复用第一次训练的脚本进行第二次训练：

```bash
python training_v1/train_gpt.py \
  --data_path training_v2/generated_sequences_super_enhanced/sequences_labels_fixed_tail_fixed.jsonl \
  --vocab_path training_v1/generated_sequences_super_enhanced/vocab.json \
  --batch_size 16 \
  --num_epochs 30 \
  --learning_rate 1e-4 \
  --max_length 512
```

要点：
- 训练输入到 `<CLS>` 截止，模型学习预测下一 token 为 `<CLS_{label}>`
- 评估时取 `<CLS>` 后一步的分布，仅在 `<CLS_0>…<CLS_9>` 上取最大作为类别
- 训练/验证需保持相同的 `max_length`、截断与 padding 策略

### 推理/加载模型

训练完成后将生成 HuggingFace 标准格式模型，可直接：

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("outputs/best_model")
tokenizer = GPT2Tokenizer.from_pretrained("outputs/best_model")
```

### 备注

- 如需更稳的分类，可在 `<CLS>` 位置并联一个线性分类头，与 LM 损失联合训练（可选）。
- 若使用 GPU，建议 PyTorch 2.0+ 与 Transformers 4.41+（支持 bf16）。


