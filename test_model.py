#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练时的codebook和词汇表测试模型
"""

import json
import torch
from transformers import GPT2LMHeadModel
import pandas as pd

def main():
    print('🧪 使用训练数据格式测试模型（CPU模式）...')
    print('=' * 80)
    
    # 加载训练时的codebook
    print('📥 加载训练时使用的codebook...')
    codebook = pd.read_csv('unified_codebook/unified_codebook.csv')
    print(f'✅ 加载codebook成功，共{len(codebook)}个聚类')
    
    # 加载训练时的词汇表
    print('📥 加载训练时使用的词汇表...')
    with open('generated_sequences_super_enhanced/vocab.json', 'r') as f:
        vocab = json.load(f)
    print(f'✅ 加载词汇表成功，共{len(vocab)}个token')
    
    # 创建反向词汇表（ID -> Token）
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # 加载测试样本
    print('📥 加载测试样本...')
    test_samples = []
    with open('generated_sequences_super_enhanced/sequences_labels_fixed.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 只取前3个样本
                break
            if line.strip():
                data = json.loads(line.strip())
                test_samples.append(data)
    print(f'✅ 加载测试样本成功，共{len(test_samples)}个样本')
    
    # 加载模型
    print('📥 加载训练好的模型...')
    model = GPT2LMHeadModel.from_pretrained('outputs/best_model')
    device = 'cpu'  # 强制使用CPU
    model = model.to(device)
    model.eval()
    print(f'✅ 模型加载成功，使用设备: {device}')
    
    # 获取模型的最大位置编码
    max_positions = getattr(model.config, 'n_positions', 512)
    print(f'📊 模型最大位置编码: {max_positions}')
    print(f'📊 模型词汇表大小: {model.config.vocab_size}')
    print(f'📊 训练词汇表大小: {len(vocab)}')
    
    # 测试每个样本
    for i, sample in enumerate(test_samples):
        print(f'\n📊 测试样本 {i+1}:')
        print(f'   标签: {sample["label"]}')
        print(f'   文件名: {sample["meta"]["filename"]}')
        print(f'   原始数字: {sample["meta"]["original_digit"]}')
        
        # 获取token序列
        tokens_str = sample['tokens']
        tokens = tokens_str.split()
        print(f'   Token序列长度: {len(tokens)}')
        
        # 找到<CLS>位置
        cls_pos = -1
        try:
            cls_pos = tokens.index('<CLS>')
        except ValueError:
            print('   ⚠️ 未找到<CLS>token，使用完整序列')
            cls_pos = len(tokens)
        
        # 训练时输入到<CLS>截止
        input_tokens = tokens[:cls_pos+1] if cls_pos < len(tokens) else tokens
        
        # 转换为ID序列
        input_ids = []
        for token in input_tokens:
            if token in vocab:
                input_ids.append(vocab[token])
            else:
                input_ids.append(vocab['<UNK>'])
        
        print(f'   训练输入长度: {len(input_ids)}')
        
        # 截断到模型最大位置编码（留一些余量）
        max_len = min(len(input_ids), max_positions - 1)
        input_ids = input_ids[:max_len]
        
        print(f'   截断后长度: {len(input_ids)}')
        print(f'   输入tokens前10个: {" ".join(input_tokens[:10])}...')
        
        # 转换为tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # 进行推理
        try:
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits
                
                # 获取最后一个位置的预测
                next_token_logits = logits[0, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # 获取top-5预测
                top_k = 5
                top_probs, top_indices = torch.topk(probs, top_k)
                
                print(f'   🔮 下一个token的top-{top_k}预测:')
                for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    idx_val = idx.item()
                    token_name = inv_vocab.get(idx_val, f'<UNK_ID_{idx_val}>')
                    print(f'     {j+1}. {token_name} (ID: {idx_val}): 概率 {prob.item():.4f}')
                
                # 检查是否有期望的标签token
                expected_label = sample['label']
                # 查找可能的标签token格式
                possible_labels = [f'<CLS_{expected_label}>', f'{expected_label}', str(expected_label)]
                
                found_expected = False
                for expected_token in possible_labels:
                    if expected_token in vocab:
                        expected_id = vocab[expected_token]
                        expected_prob = probs[expected_id].item()
                        print(f'   🎯 期望标签: {expected_token} (ID: {expected_id}), 概率: {expected_prob:.4f}')
                        
                        # 检查是否在top-5中
                        if expected_id in top_indices:
                            rank = (top_indices == expected_id).nonzero(as_tuple=True)[0].item()
                            print(f'   ✅ 期望标签在top-{top_k}中，排名: {rank+1}')
                        else:
                            print(f'   ❌ 期望标签不在top-{top_k}中')
                        found_expected = True
                        break
                
                if not found_expected:
                    print(f'   ⚠️ 未找到期望标签token（尝试了: {possible_labels}）')
            
        except Exception as e:
            print(f'   ❌ 推理失败: {e}')
        
        print('-' * 80)
    
    print('\n✅ 测试完成！')
    
    # 简单的生成测试
    print('\n🚀 测试文本生成能力...')
    try:
        # 使用简单的输入序列
        test_input = ['<IMG>', '<Z_100>', '<Z_200>', '<CLS>']
        test_ids = [vocab.get(token, vocab['<UNK>']) for token in test_input]
        test_tensor = torch.tensor([test_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                test_tensor,
                max_new_tokens=3,
                temperature=0.8,
                do_sample=True,
                pad_token_id=vocab['<PAD>'],
                eos_token_id=vocab.get('<EOS>', vocab['<UNK>'])
            )
            
            # 转换回token名称
            generated_tokens = []
            for token_id in generated[0]:
                token_name = inv_vocab.get(token_id.item(), f'<UNK_ID_{token_id.item()}>')
                generated_tokens.append(token_name)
            
            print(f'   📝 输入: {test_input}')
            print(f'   🎯 生成结果: {generated_tokens}')
            print(f'   🆕 新增tokens: {generated_tokens[len(test_input):]}')
    
    except Exception as e:
        print(f'   ❌ 生成测试失败: {e}')

if __name__ == "__main__":
    main()
