#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从零训练小GPT因果语言模型用于next-token预测
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import os
import argparse
from typing import List, Dict, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CausalLMDataset(Dataset):
    """因果语言模型数据集类"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载JSONL数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"加载了 {len(data)} 条数据")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        
        # 将tokens转换为token IDs
        token_ids = []
        for token in tokens.split():
            if token in self.tokenizer.get_vocab():
                token_ids.append(self.tokenizer.get_vocab()[token])
            else:
                token_ids.append(self.tokenizer.unk_token_id)
        
        # 截断或填充到指定长度
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(token_ids))
        
        # 对于因果语言模型，labels是向右移动一位的input_ids
        # pad位置的label设为-100（不参与损失计算）
        labels = token_ids[1:] + [-100]  # 右移一位，最后一个位置设为-100
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor([1 if tid != self.tokenizer.pad_token_id else 0 for tid in token_ids])
        }

class CustomGPT2Config(GPT2Config):
    """自定义GPT2配置"""
    
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        # 减小模型大小，适合从零训练
        self.n_layer = 6  # 原始是12层
        self.n_head = 8   # 原始是12头
        self.n_embd = 384 # 原始是768维
        self.n_positions = 1024
        self.n_ctx = 1024

class CausalGPT2LM(nn.Module):
    """因果语言模型GPT2"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 使用标准的GPT2LMHeadModel，它已经包含了语言建模头
        self.transformer = GPT2LMHeadModel(config)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 直接使用GPT2LMHeadModel的前向传播
        # 它会自动处理因果语言建模任务
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs

def create_tokenizer(vocab_path: str, model_name: str = "gpt2"):
    """创建tokenizer"""
    # 加载自定义词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # 创建tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # 添加特殊token
    special_tokens = ['<PAD>', '<UNK>', '<IMG>', '</IMG>', '<CLS>', '<EOS>']
    tokenizer.add_special_tokens({'pad_token': '<PAD>', 'unk_token': '<UNK>'})
    
    # 添加自定义token
    tokenizer.add_tokens(list(vocab.keys()))
    
    # 设置pad_token_id
    tokenizer.pad_token = '<PAD>'
    tokenizer.pad_token_id = vocab['<PAD>']
    
    logger.info(f"词汇表大小: {len(tokenizer)}")
    return tokenizer

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=10, learning_rate=5e-5):
    """训练模型"""
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算困惑度 (perplexity)
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PPL': f'{torch.exp(loss).item():.2f}'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                val_loss += loss.item()
                
                # 计算困惑度 (perplexity)
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PPL': f'{torch.exp(loss).item():.2f}'
                })
        
        # 计算平均损失和困惑度
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        avg_train_ppl = torch.exp(torch.tensor(avg_train_loss))
        avg_val_ppl = torch.exp(torch.tensor(avg_val_loss))
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.4f}, Train PPL: {avg_val_ppl:.2f}')
        logger.info(f'  Val Loss: {avg_val_loss:.4f}, Val PPL: {avg_val_ppl:.2f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 保存为HuggingFace标准格式
            best_model_dir = 'outputs/best_model'
            os.makedirs(best_model_dir, exist_ok=True)
            
            # 保存模型权重
            model.save_pretrained(best_model_dir)
            # 保存tokenizer
            tokenizer.save_pretrained(best_model_dir)
            # 保存训练配置
            config.save_pretrained(best_model_dir)
            
            # 保存训练状态
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_ppl': avg_val_ppl.item()
            }, os.path.join(best_model_dir, 'training_args.bin'))
            
            logger.info(f'保存最佳模型到 {best_model_dir}, 验证损失: {best_val_loss:.4f}, 困惑度: {avg_val_ppl:.2f}')
        
        scheduler.step()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='训练GPT因果语言模型用于next-token预测')
    parser.add_argument('--data_path', type=str, default='generated_sequences_super_enhanced/sequences_labels_fixed.jsonl',
                       help='训练数据路径')
    parser.add_argument('--vocab_path', type=str, default='generated_sequences_super_enhanced/vocab.json',
                       help='词汇表路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--max_length', type=int, default=1024, help='最大序列长度')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    
    # 创建tokenizer
    tokenizer = create_tokenizer(args.vocab_path)
    
    # 加载数据
    dataset = CausalLMDataset(args.data_path, tokenizer, args.max_length)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型配置
    config = CustomGPT2Config(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 创建模型
    model = CausalGPT2LM(config)
    model.to(device)
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # 保存最终模型为HuggingFace标准格式
    final_model_dir = 'outputs/final_model'
    os.makedirs(final_model_dir, exist_ok=True)
    
    # 保存模型权重
    trained_model.save_pretrained(final_model_dir)
    # 保存tokenizer
    tokenizer.save_pretrained(final_model_dir)
    # 保存训练配置
    config.save_pretrained(final_model_dir)
    
    logger.info(f"训练完成！模型已保存为HuggingFace标准格式到 {final_model_dir}")
    logger.info("现在可以使用 from_pretrained() 直接加载模型！")

if __name__ == "__main__":
    main()
