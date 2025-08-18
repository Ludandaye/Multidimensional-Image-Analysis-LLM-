#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的训练脚本
解决所有已识别的问题：特殊符号统一、监督对齐、截断策略、评估方式等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import os
import logging
import argparse
from tqdm import tqdm
import json

from config.model_config import get_config
from data_processor_fixed import create_datasets
from classification_evaluator import ClassificationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedGPT2Model(nn.Module):
    """修复后的GPT2模型，确保词汇表大小一致"""
    
    def __init__(self, config):
        super().__init__()
        
        # 创建GPT2配置，确保词汇表大小正确
        gpt2_config = GPT2Config(
            vocab_size=config.model.vocab_size,  # 使用实际词汇表大小
            n_positions=config.model.n_positions,
            n_embd=config.model.n_embd,
            n_layer=config.model.n_layer,
            n_head=config.model.n_head,
            bos_token_id=config.vocab.get(config.special_tokens.eos_token, 5),  # 没有BOS，用EOS
            eos_token_id=config.vocab.get(config.special_tokens.eos_token, 5),
            pad_token_id=config.vocab.get(config.special_tokens.pad_token, 0)
        )
        
        self.transformer = GPT2LMHeadModel(gpt2_config)
        self.config = config
        
        logger.info(f"✅ 模型初始化完成，词汇表大小: {gpt2_config.vocab_size}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
    
    def save_pretrained(self, save_path):
        """保存模型为HuggingFace格式"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存transformer部分
        self.transformer.save_pretrained(save_path)
        
        # 保存配置
        config_dict = self.config.model.__dict__.copy()
        config_dict.update({
            'special_tokens': self.config.special_tokens.__dict__,
            'experiment': self.config.experiment.__dict__
        })
        
        with open(os.path.join(save_path, 'unified_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 模型已保存到: {save_path}")

def train_fixed_model(config, device='cpu', num_epochs=5):
    """使用修复后的配置训练模型"""
    
    logger.info(f"🚀 开始修复后的训练...")
    config.print_summary()
    
    # 创建数据集
    train_dataset, val_dataset = create_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=config.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.model.batch_size, shuffle=False)
    
    # 创建模型
    model = FixedGPT2Model(config)
    model.to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.model.learning_rate)
    
    # 创建评估器
    evaluator = ClassificationEvaluator(model.transformer, config, device)
    
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\n📚 第 {epoch+1}/{num_epochs} 轮训练")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_steps
        
        # 验证阶段 - 使用分类评估
        logger.info("🔍 进行分类评估...")
        val_results = evaluator.evaluate_dataset(val_dataset, batch_size=8)
        val_accuracy = val_results['accuracy']
        
        logger.info(f"📊 第{epoch+1}轮结果:")
        logger.info(f"   训练损失: {avg_train_loss:.4f}")
        logger.info(f"   验证准确率: {val_accuracy:.2%}")
        logger.info(f"   验证置信度: {val_results['avg_confidence']:.4f}")
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = 'outputs/best_model_fixed'
            model.save_pretrained(save_path)
            
            # 保存词汇表
            import shutil
            shutil.copy(config.data.vocab_path, os.path.join(save_path, 'vocab.json'))
            
            logger.info(f"🎉 新的最佳模型！准确率: {val_accuracy:.2%}")
    
    # 最终评估
    logger.info(f"\n🏆 最终评估结果:")
    logger.info(f"   最佳验证准确率: {best_val_accuracy:.2%}")
    
    return best_val_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"🖥️ 使用设备: {device}")
    
    # 获取配置
    config = get_config()
    
    # 训练模型
    best_accuracy = train_fixed_model(config, device, args.epochs)
    
    print(f"\n🎯 训练完成！最佳准确率: {best_accuracy:.2%}")
    print(f"📁 模型保存在: outputs/best_model_fixed")
