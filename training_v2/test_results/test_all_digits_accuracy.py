import json
import torch
from transformers import GPT2LMHeadModel
import pandas as pd
from collections import defaultdict
import random

def main():
    print('🎯 测试所有数字的标签预测准确性')
    print('=' * 80)
    
    # 加载训练时的词汇表
    with open('generated_sequences_super_enhanced/vocab.json', 'r') as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # 查找标签token
    label_tokens = {}
    for token, idx in vocab.items():
        if '<CLS_' in token and '>' in token:
            label = token.replace('<CLS_', '').replace('>', '')
            if label.isdigit():
                label_tokens[int(label)] = idx
    
    print(f'✅ 找到{len(label_tokens)}个数字标签token')
    
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained('outputs/best_model')
    model.eval()
    device = 'cpu'
    model = model.to(device)
    print(f'✅ 模型加载成功')
    
    # 加载所有测试样本
    all_samples = []
    with open('generated_sequences_super_enhanced/sequences_labels_fixed.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line.strip()))
    
    print(f'✅ 加载{len(all_samples)}个测试样本')
    
    # 按数字分组样本
    samples_by_digit = defaultdict(list)
    for sample in all_samples:
        samples_by_digit[sample['label']].append(sample)
    
    # 从每个数字中随机选择10个样本进行测试
    test_samples = []
    samples_per_digit = 10
    
    for digit in range(10):
        if digit in samples_by_digit:
            digit_samples = samples_by_digit[digit]
            # 随机选择样本
            selected = random.sample(digit_samples, min(samples_per_digit, len(digit_samples)))
            test_samples.extend(selected)
    
    random.shuffle(test_samples)  # 打乱顺序
    print(f'✅ 选择了{len(test_samples)}个测试样本（每个数字{samples_per_digit}个）')
    
    # 测试准确性
    correct_predictions = 0
    total_predictions = 0
    label_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    print('\n🧪 开始测试...')
    
    for i, sample in enumerate(test_samples):
        if (i + 1) % 20 == 0:
            print(f'   进度: {i+1}/{len(test_samples)}')
        
        true_label = sample['label']
        tokens = sample['tokens'].split()
        
        # 找到<CLS>位置
        try:
            cls_pos = tokens.index('<CLS>')
            input_tokens = tokens[:cls_pos+1]
        except ValueError:
            continue
        
        # 转换为ID序列并截断
        input_ids = [vocab.get(t, vocab['<UNK>']) for t in input_tokens]
        max_len = min(len(input_ids), 400)
        input_ids = input_ids[:max_len]
        
        # 进行预测
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        # 检查所有标签token的概率
        label_probs = {}
        for label_num, token_id in label_tokens.items():
            label_probs[label_num] = probs[token_id].item()
        
        # 找到概率最高的标签
        if label_probs:
            predicted_label = max(label_probs, key=label_probs.get)
            
            # 统计
            total_predictions += 1
            label_accuracy[true_label]['total'] += 1
            confusion_matrix[true_label][predicted_label] += 1
            
            if predicted_label == true_label:
                correct_predictions += 1
                label_accuracy[true_label]['correct'] += 1
    
    # 计算总体准确率
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print('\n' + '=' * 80)
    print('📈 详细测试结果')
    print('=' * 80)
    print(f'总体准确率: {correct_predictions}/{total_predictions} = {overall_accuracy:.2%}')
    
    print('\n📊 各数字的预测准确率:')
    for label in sorted(label_accuracy.keys()):
        acc_data = label_accuracy[label]
        if acc_data['total'] > 0:
            acc_rate = acc_data['correct'] / acc_data['total']
            print(f'   数字 {label}: {acc_data["correct"]}/{acc_data["total"]} = {acc_rate:.2%}')
    
    # 显示混淆矩阵
    print('\n📋 混淆矩阵 (行=真实标签, 列=预测标签):')
    print('     ', end='')
    for pred in range(10):
        print(f'{pred:4d}', end='')
    print()
    
    for true in range(10):
        print(f'{true:2d}: ', end='')
        for pred in range(10):
            count = confusion_matrix[true][pred]
            print(f'{count:4d}', end='')
        print()
    
    # 找出最容易混淆的数字对
    print('\n🔍 最容易混淆的数字对:')
    confusion_pairs = []
    for true in range(10):
        for pred in range(10):
            if true != pred and confusion_matrix[true][pred] > 0:
                confusion_pairs.append((true, pred, confusion_matrix[true][pred]))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (true, pred, count) in enumerate(confusion_pairs[:5]):
        print(f'   {i+1}. 数字{true} 被误认为 数字{pred}: {count} 次')
    
    # 分析结果
    print('\n🔍 结果分析:')
    if overall_accuracy > 0.8:
        print('   🎉 模型表现优秀！准确率超过80%')
    elif overall_accuracy > 0.5:
        print('   �� 模型表现良好，准确率超过50%')
    elif overall_accuracy > 0.2:
        print('   ⚠️ 模型有一定预测能力，但需要改进')
    else:
        print('   ❌ 模型预测准确率较低，可能需要更多训练')
    
    random_accuracy = 1/len(label_tokens)
    print(f'\n💡 随机猜测的期望准确率: {random_accuracy:.2%}')
    if overall_accuracy > random_accuracy:
        improvement = overall_accuracy / random_accuracy
        print(f'📈 模型比随机猜测好 {improvement:.1f} 倍')
    
    # 给出改进建议
    print('\n💡 改进建议:')
    if overall_accuracy < 0.5:
        print('   - 可能需要更多的训练轮数')
        print('   - 考虑调整学习率或优化器参数')
        print('   - 检查数据预处理是否正确')
        print('   - 考虑增加模型复杂度')
    else:
        print('   - 模型已显示出学习能力')
        print('   - 可以尝试微调超参数进一步提升')

if __name__ == "__main__":
    main()
