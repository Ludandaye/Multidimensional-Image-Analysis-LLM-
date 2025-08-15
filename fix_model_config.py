#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复模型配置文件，确保模型可以正常加载
"""

import json
import os
import torch
from transformers import GPT2Config, GPT2Tokenizer
import pickle

def fix_model_configs():
    """修复模型配置文件"""
    print("🔧 开始修复模型配置文件...")
    
    # 从检查点加载配置信息
    checkpoint_path = 'outputs/checkpoints/checkpoint.pkl'
    if os.path.exists(checkpoint_path):
        print(f"📂 从检查点加载配置信息: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ 检查点加载成功，轮次: {checkpoint.get('epoch', 'unknown')}")
    else:
        print("❌ 检查点文件不存在")
        return False
    
    # 加载词汇表信息
    vocab_path = 'generated_sequences_super_enhanced/vocab.json'
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab_size = len(vocab) + 50257  # 自定义vocab + GPT2默认vocab
        print(f"📚 词汇表大小: {vocab_size}")
    else:
        vocab_size = 50773  # 从训练日志中获取的大小
        print(f"⚠️ 使用默认词汇表大小: {vocab_size}")
    
    # 创建正确的配置
    config_dict = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": vocab_size,
        "n_positions": 1024,
        "n_embd": 384,
        "n_layer": 6,
        "n_head": 8,
        "n_inner": 1536,  # 通常是 n_embd * 4
        "activation_function": "gelu_new",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-05,
        "initializer_range": 0.02,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "summary_activation": None,
        "summary_proj_to_labels": True,
        "summary_first_dropout": 0.1,
        "scale_attn_weights": True,
        "use_cache": True,
        "scale_attn_by_inverse_layer_idx": False,
        "reorder_and_upcast_attn": False,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50257,  # 假设PAD token是第一个添加的特殊token
        "transformers_version": "4.41.0"
    }
    
    # 修复最佳模型目录
    best_model_dir = 'outputs/best_model'
    if os.path.exists(best_model_dir):
        print(f"🔧 修复最佳模型配置: {best_model_dir}")
        
        # 保存config.json
        config_path = os.path.join(best_model_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ 配置文件已保存: {config_path}")
        
        # 复制tokenizer文件到最佳模型目录
        final_model_dir = 'outputs/final_model'
        if os.path.exists(final_model_dir):
            tokenizer_files = [
                'vocab.json',
                'merges.txt', 
                'added_tokens.json',
                'special_tokens_map.json',
                'tokenizer_config.json'
            ]
            
            for file_name in tokenizer_files:
                src_path = os.path.join(final_model_dir, file_name)
                dst_path = os.path.join(best_model_dir, file_name)
                
                if os.path.exists(src_path):
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    print(f"✅ 复制文件: {file_name}")
                else:
                    print(f"⚠️ 文件不存在: {file_name}")
    
    # 修复最终模型目录
    final_model_dir = 'outputs/final_model'
    if os.path.exists(final_model_dir):
        print(f"🔧 修复最终模型配置: {final_model_dir}")
        
        # 保存config.json
        config_path = os.path.join(final_model_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ 配置文件已保存: {config_path}")
    
    return True

def test_model_loading():
    """测试模型是否可以正常加载"""
    print("\n🧪 测试模型加载...")
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # 测试加载最佳模型
        best_model_dir = 'outputs/best_model'
        if os.path.exists(os.path.join(best_model_dir, 'config.json')):
            print(f"📂 测试加载最佳模型: {best_model_dir}")
            
            # 加载配置
            config = GPT2Config.from_pretrained(best_model_dir)
            print(f"✅ 配置加载成功，词汇表大小: {config.vocab_size}")
            
            # 加载tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(best_model_dir)
            print(f"✅ Tokenizer加载成功，词汇表大小: {len(tokenizer)}")
            
            # 加载模型
            model = GPT2LMHeadModel.from_pretrained(best_model_dir)
            print(f"✅ 模型加载成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
            print("🎉 最佳模型修复成功，可以正常加载！")
            return True
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始修复GPT模型配置文件...")
    
    # 修复配置文件
    if fix_model_configs():
        print("\n✅ 配置文件修复完成！")
        
        # 测试加载
        if test_model_loading():
            print("\n🎉 模型修复成功！现在可以使用以下方式加载模型：")
            print("```python")
            print("from transformers import GPT2LMHeadModel, GPT2Tokenizer")
            print("model = GPT2LMHeadModel.from_pretrained('outputs/best_model')")
            print("tokenizer = GPT2Tokenizer.from_pretrained('outputs/best_model')")
            print("```")
        else:
            print("\n⚠️ 配置修复完成，但模型加载测试失败，可能需要进一步调试")
    else:
        print("\n❌ 配置文件修复失败")
