#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用测试集内容测试模型，确保格式与训练时一致
使用训练时的codebook和词汇表
"""

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np

def load_training_codebook():
    """加载训练时使用的codebook"""
    codebook_path = 'unified_codebook/unified_codebook.csv'
    codebook = pd.read_csv(codebook_path)
    print(f"✅ 加载codebook成功，共{len(codebook)}个聚类")
    return codebook

def load_vocab():
    """加载训练时使用的词汇表"""
    vocab_path = 'generated_sequences_super_enhanced/vocab.json'
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    print(f"✅ 加载词汇表成功，共{len(vocab)}个token")
    return vocab

def load_test_samples():
    """加载测试样本"""
    test_data_path = 'generated_sequences_super_enhanced/sequences_labels_fixed.jsonl'
    samples = []
    
    with open(test_data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 只取前5个样本
                break
            data = json.loads(line.strip())
            samples.append(data)
    
    print(f"✅ 加载测试样本成功，共{len(samples)}个样本")
    return samples

def tokenize_sequence(tokens_str, vocab):
    """将token字符串转换为ID序列"""
    tokens = tokens_str.split()
    token_ids = []
    
    for token in tokens:
        if token in vocab:
            token_ids.append(vocab[token])
        else:
            token_ids.append(vocab['<UNK>'])
    
    return token_ids

def test_model_with_training_data():
    """使用训练数据格式测试模型"""
    print("🧪 使用训练数据格式测试模型...")
    
    # 加载必要的数据
    codebook = load_training_codebook()
    vocab = load_vocab()
    test_samples = load_test_samples()
    
    # 加载模型
    print("📥 加载训练好的模型...")
    try:
        model = GPT2LMHeadModel.from_pretrained('outputs/best_model')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        print(f"✅ 模型加载成功，使用设备: {device}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试每个样本
    for i, sample in enumerate(test_samples):
        print(f"\n📊 测试样本 {i+1}:")
        print(f"   标签: {sample['label']}")
        print(f"   文件名: {sample['meta']['filename']}")
        print(f"   原始数字: {sample['meta']['original_digit']}")
        
        # 获取token序列
        tokens_str = sample['tokens']
        print(f"   Token序列长度: {len(tokens_str.split())}")
        
        # 转换为ID序列
        token_ids = tokenize_sequence(tokens_str, vocab)
        print(f"   Token ID序列长度: {len(token_ids)}")
        
        # 截取到<CLS>位置（训练时的格式）
        cls_pos = -1
        for j, token in enumerate(tokens_str.split()):
            if token == '<CLS>':
                cls_pos = j
                break
        
        if cls_pos != -1:
            # 训练时输入到<CLS>截止
            input_tokens = tokens_str.split()[:cls_pos+1]
            input_ids = tokenize_sequence(' '.join(input_tokens), vocab)
            
            print(f"   训练输入长度: {len(input_ids)}")
            print(f"   训练输入tokens: {' '.join(input_tokens)}")
            
            # 转换为tensor
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            # 进行推理
            try:
                with torch.no_grad():
                    outputs = model(input_tensor)
                    logits = outputs.logits
                    
                    # 获取<CLS>后一步的预测（训练时的目标）
                    next_token_logits = logits[0, -1, :]
                    probs = torch.softmax(next_token_logits, dim=-1)
                    
                    # 获取top-5预测
                    top_k = 5
                    top_probs, top_indices = torch.topk(probs, top_k)
                    
                    print(f"   🔮 <CLS>后一步的top-{top_k}预测:")
                    for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                        # 找到对应的token
                        token_name = None
                        for name, id_val in vocab.items():
                            if id_val == idx.item():
                                token_name = name
                                break
                        
                        print(f"     {j+1}. {token_name} (ID: {idx.item()}): 概率 {prob.item():.4f}")
                    
                    # 检查是否预测了正确的标签token
                    expected_label = sample['label']
                    expected_token = f"<CLS_{expected_label}>"
                    
                    if expected_token in vocab:
                        expected_id = vocab[expected_token]
                        expected_prob = probs[expected_id].item()
                        print(f"   🎯 期望标签: {expected_token} (ID: {expected_id}), 概率: {expected_prob:.4f}")
                        
                        # 检查是否在top-5中
                        if expected_id in top_indices:
                            rank = (top_indices == expected_id).nonzero(as_tuple=True)[0].item()
                            print(f"   ✅ 期望标签在top-{top_k}中，排名: {rank+1}")
                        else:
                            print(f"   ❌ 期望标签不在top-{top_k}中")
                    else:
                        print(f"   ⚠️ 期望标签token {expected_token} 不在词汇表中")
                
            except Exception as e:
                print(f"   ❌ 推理失败: {e}")
        
        print("-" * 80)

def test_text_generation():
    """测试文本生成能力"""
    print("\n🚀 测试文本生成能力...")
    
    try:
        model = GPT2LMHeadModel.from_pretrained('outputs/best_model')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        # 使用训练时的特殊token
        vocab = load_vocab()
        
        # 测试不同的输入
        test_inputs = [
            ["<IMG>", "<Z_100>", "<Z_200>"],
            ["<IMG>", "<Z_300>", "<Z_400>", "<Z_500>"],
            ["<IMG>", "<Z_100>", "<Z_200>", "<Z_300>", "<CLS>"]
        ]
        
        for i, input_tokens in enumerate(test_inputs):
            print(f"\n📝 测试输入 {i+1}: {input_tokens}")
            
            # 转换为ID
            input_ids = []
            for token in input_tokens:
                if token in vocab:
                    input_ids.append(vocab[token])
                else:
                    input_ids.append(vocab['<UNK>'])
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            try:
                with torch.no_grad():
                    # 生成文本
                    generated = model.generate(
                        input_tensor,
                        max_new_tokens=5,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=vocab['<PAD>'],
                        eos_token_id=vocab['<EOS>']
                    )
                    
                    # 转换回token名称
                    generated_tokens = []
                    for token_id in generated[0]:
                        for name, id_val in vocab.items():
                            if id_val == token_id.item():
                                generated_tokens.append(name)
                                break
                    
                    print(f"   🎯 生成结果: {generated_tokens}")
                    print(f"   🆕 新增tokens: {generated_tokens[len(input_tokens):]}")
                    
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
    
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 使用训练数据格式测试模型")
    print("=" * 80)
    
    # 测试1: 使用训练数据格式
    test_model_with_training_data()
    
    # 测试2: 文本生成
    test_text_generation()
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
