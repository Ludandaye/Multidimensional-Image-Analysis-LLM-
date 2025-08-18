import json
import torch
from transformers import GPT2LMHeadModel
import pandas as pd
from collections import defaultdict

def main():
    print('🎯 测试模型生成label的正确性')
    print('=' * 80)
    
    # 加载训练时的词汇表
    with open('generated_sequences_super_enhanced/vocab.json', 'r') as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    print(f'✅ 加载词汇表成功，共{len(vocab)}个token')
    
    # 查看词汇表中的标签token
    label_tokens = {}
    print('\n📋 词汇表中的标签token:')
    for token, idx in vocab.items():
        if '<CLS_' in token and '>' in token:
            label = token.replace('<CLS_', '').replace('>', '')
            if label.isdigit():
                label_tokens[int(label)] = idx
                print(f'   {token}: ID {idx}')
    
    print(f'\n✅ 找到{len(label_tokens)}个数字标签token')
    
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained('outputs/best_model')
    model.eval()
    device = 'cpu'
    model = model.to(device)
    print(f'✅ 模型加载成功')
    
    # 加载所有测试样本（不只是前几个）
    test_samples = []
    with open('generated_sequences_super_enhanced/sequences_labels_fixed.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line.strip()))
    
    print(f'✅ 加载{len(test_samples)}个测试样本')
    
    # 统计每个数字的样本数量
    label_counts = defaultdict(int)
    for sample in test_samples:
        label_counts[sample['label']] += 1
    
    print('\n📊 数据集中各数字的样本数量:')
    for label in sorted(label_counts.keys()):
        print(f'   数字 {label}: {label_counts[label]} 个样本')
    
    # 测试准确性
    correct_predictions = 0
    total_predictions = 0
    label_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    print('\n🧪 开始测试标签预测准确性...')
    
    # 只测试前50个样本以节省时间
    test_count = min(50, len(test_samples))
    print(f'测试前{test_count}个样本')
    
    for i, sample in enumerate(test_samples[:test_count]):
        if (i + 1) % 10 == 0:
            print(f'   进度: {i+1}/{test_count}')
        
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
        
        # 检查所有可能的标签token的概率
        label_probs = {}
        for label_num, token_id in label_tokens.items():
            label_probs[label_num] = probs[token_id].item()
        
        # 找到概率最高的标签
        if label_probs:
            predicted_label = max(label_probs, key=label_probs.get)
            predicted_prob = label_probs[predicted_label]
            
            # 统计准确性
            total_predictions += 1
            label_accuracy[true_label]['total'] += 1
            
            if predicted_label == true_label:
                correct_predictions += 1
                label_accuracy[true_label]['correct'] += 1
            
            # 显示详细结果（只显示前10个）
            if i < 10:
                print(f'\n📊 样本 {i+1}:')
                print(f'   真实标签: {true_label}')
                print(f'   预测标签: {predicted_label} (概率: {predicted_prob:.6f})')
                print(f'   预测结果: {"✅ 正确" if predicted_label == true_label else "❌ 错误"}')
                
                # 显示所有标签的概率
                print(f'   所有标签概率:')
                for label_num in sorted(label_probs.keys()):
                    prob = label_probs[label_num]
                    marker = "👉" if label_num == predicted_label else "  "
                    print(f'     {marker} 数字{label_num}: {prob:.6f}')
    
    # 计算总体准确率
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print('\n' + '=' * 80)
    print('📈 测试结果统计')
    print('=' * 80)
    print(f'总体准确率: {correct_predictions}/{total_predictions} = {overall_accuracy:.2%}')
    
    print('\n📊 各数字的预测准确率:')
    for label in sorted(label_accuracy.keys()):
        acc_data = label_accuracy[label]
        if acc_data['total'] > 0:
            acc_rate = acc_data['correct'] / acc_data['total']
            print(f'   数字 {label}: {acc_data["correct"]}/{acc_data["total"]} = {acc_rate:.2%}')
    
    # 分析结果
    print('\n🔍 结果分析:')
    if overall_accuracy > 0.8:
        print('   🎉 模型表现优秀！准确率超过80%')
    elif overall_accuracy > 0.5:
        print('   👍 模型表现良好，准确率超过50%')
    elif overall_accuracy > 0.1:
        print('   ⚠️ 模型有一定预测能力，但需要改进')
    else:
        print('   ❌ 模型预测准确率较低，可能需要更多训练')
    
    print(f'\n💡 随机猜测的期望准确率: {1/len(label_tokens):.2%}')
    if overall_accuracy > 1/len(label_tokens):
        improvement = overall_accuracy / (1/len(label_tokens))
        print(f'📈 模型比随机猜测好 {improvement:.1f} 倍')

if __name__ == "__main__":
    main()
