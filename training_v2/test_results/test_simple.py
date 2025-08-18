import json
import torch
from transformers import GPT2LMHeadModel

# 加载词汇表
with open('generated_sequences_super_enhanced/vocab.json', 'r') as f:
    vocab = json.load(f)
inv_vocab = {v: k for k, v in vocab.items()}

# 加载模型
model = GPT2LMHeadModel.from_pretrained('outputs/best_model')
model.eval()
device = 'cpu'
model = model.to(device)

print(f"模型词汇表大小: {model.config.vocab_size}")
print(f"训练词汇表大小: {len(vocab)}")
print(f"模型最大位置: {getattr(model.config, 'n_positions', 512)}")

# 加载一个测试样本
with open('generated_sequences_super_enhanced/sequences_labels_fixed.jsonl', 'r') as f:
    sample = json.loads(f.readline().strip())

tokens = sample['tokens'].split()
print(f"样本token数量: {len(tokens)}")

# 找到CLS位置
try:
    cls_pos = tokens.index('<CLS>')
    input_tokens = tokens[:cls_pos+1]
except ValueError:
    input_tokens = tokens[:100]  # 取前100个

# 转换为ID
input_ids = [vocab.get(t, vocab['<UNK>']) for t in input_tokens]
input_ids = input_ids[:min(len(input_ids), 400)]  # 截断到400

print(f"输入长度: {len(input_ids)}")

# 推理
input_tensor = torch.tensor([input_ids], dtype=torch.long)
with torch.no_grad():
    outputs = model(input_tensor)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 5)

print("Top-5预测:")
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    token_name = inv_vocab.get(idx.item(), f'UNK_{idx.item()}')
    print(f"  {i+1}. {token_name}: {prob.item():.4f}")

print("测试完成!")
