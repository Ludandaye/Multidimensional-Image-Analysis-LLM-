#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本：使用训练好的GPT因果语言模型进行文本生成和next-token预测
"""

import torch
import json
from transformers import GPT2Tokenizer
from train_gpt import CausalGPT2LM, CustomGPT2Config
import argparse

def load_model(model_path, device):
    """加载训练好的模型（支持HuggingFace标准格式）"""
    if os.path.isdir(model_path):
        # HuggingFace标准格式目录
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # 直接加载模型和tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        model.to(device)
        model.eval()
        return model, tokenizer
    else:
        # 兼容旧的.pth格式
        from train_gpt import CausalGPT2LM, CustomGPT2Config
        
        # 需要手动创建配置和模型
        raise ValueError("请使用HuggingFace标准格式的模型目录，或更新推理脚本以支持.pth格式")

def generate_next_tokens(model, tokenizer, prompt, device, max_new_tokens=10, temperature=0.8):
    """生成下一个token序列"""
    # 将prompt转换为token IDs
    token_ids = []
    for token in prompt.split():
        if token in tokenizer.get_vocab():
            token_ids.append(tokenizer.get_vocab()[token])
        else:
            token_ids.append(tokenizer.unk_token_id)
    
    # 转换为tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.transformer.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成的tokens
    generated_tokens = outputs[0][len(token_ids):]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text

def predict_next_token(model, tokenizer, prompt, device):
    """预测下一个token的概率分布"""
    # 将prompt转换为token IDs
    token_ids = []
    for token in prompt.split():
        if token in tokenizer.get_vocab():
            token_ids.append(tokenizer.get_vocab()[token])
        else:
            token_ids.append(tokenizer.unk_token_id)
    
    # 转换为tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]  # 最后一个位置的logits
        probs = torch.softmax(next_token_logits, dim=-1)
    
    # 获取top-k tokens
    top_k = 5
    top_probs, top_indices = torch.topk(probs, top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx])
        results.append((token, prob.item()))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='使用训练好的GPT因果语言模型进行推理')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='模型路径')
    parser.add_argument('--vocab_path', type=str, default='generated_sequences_super_enhanced/vocab.json', help='词汇表路径')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--mode', type=str, default='interactive', choices=['interactive', 'generate'], help='推理模式')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载模型（包含tokenizer）
    print("加载模型...")
    model, tokenizer = load_model(args.model_path, device)
    print("模型加载完成！")
    
    if args.mode == 'generate':
        # 文本生成模式
        print("\n文本生成模式（输入 'quit' 退出）")
        while True:
            user_input = input("\n请输入前缀tokens: ").strip()
            if user_input.lower() == 'quit':
                break
            
            try:
                generated_text = generate_next_tokens(model, tokenizer, user_input, device, max_new_tokens=20)
                print(f"生成结果: {user_input} {generated_text}")
            except Exception as e:
                print(f"生成失败: {e}")
    
    else:
        # 交互式next-token预测模式
        print("\nNext-token预测模式（输入 'quit' 退出）")
        while True:
            user_input = input("\n请输入前缀tokens: ").strip()
            if user_input.lower() == 'quit':
                break
            
            try:
                top_tokens = predict_next_token(model, tokenizer, user_input, device)
                print(f"前缀: {user_input}")
                print("下一个token的top-5预测:")
                for i, (token, prob) in enumerate(top_tokens, 1):
                    print(f"  {i}. '{token}': {prob:.3f}")
            except Exception as e:
                print(f"预测失败: {e}")

if __name__ == "__main__":
    main()
