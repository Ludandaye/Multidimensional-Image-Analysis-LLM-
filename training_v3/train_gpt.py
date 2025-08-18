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
import time
import datetime
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import pickle

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建富文本控制台
console = Console()

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, best_val_loss, checkpoint_dir):
    """保存训练检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.datetime.now().isoformat()
    }
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"检查点已保存到: {checkpoint_path}")

def load_checkpoint(checkpoint_dir):
    """加载训练检查点"""
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"从检查点恢复训练: {checkpoint_path}")
        logger.info(f"上次训练时间: {checkpoint['timestamp']}")
        logger.info(f"恢复到第 {checkpoint['epoch']} 轮")
        return checkpoint
    return None

def display_training_stats(epoch, num_epochs, train_loss, val_loss, train_ppl, val_ppl, 
                          best_val_loss, learning_rate, elapsed_time, eta):
    """显示训练统计信息"""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan", no_wrap=True)
    table.add_column("数值", style="yellow")
    
    table.add_row("轮次", f"{epoch}/{num_epochs}")
    table.add_row("训练损失", f"{train_loss:.4f}")
    table.add_row("验证损失", f"{val_loss:.4f}")
    table.add_row("训练困惑度", f"{train_ppl:.2f}")
    table.add_row("验证困惑度", f"{val_ppl:.2f}")
    table.add_row("最佳验证损失", f"{best_val_loss:.4f}")
    table.add_row("学习率", f"{learning_rate:.2e}")
    table.add_row("已用时间", f"{elapsed_time}")
    table.add_row("预计剩余", f"{eta}")
    
    console.print(Panel(table, title=f"[bold green]训练进度 - 第 {epoch} 轮[/bold green]", 
                       border_style="green"))

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
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """从预训练模型加载配置"""
        config = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # 确保所有必要的参数都被设置
        if not hasattr(config, 'vocab_size'):
            config.vocab_size = 50257  # GPT2默认词汇表大小
        return config

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

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=10, learning_rate=5e-5, tokenizer=None, config=None, resume_from_checkpoint=True):
    """训练模型 - 支持断点续训和详细进度显示"""
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 初始化训练状态
    start_epoch = 0
    best_val_loss = float('inf')
    train_start_time = time.time()
    
    # 尝试加载检查点
    checkpoint_dir = 'outputs/checkpoints'
    if resume_from_checkpoint:
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            console.print(f"[green]✅ 从第 {checkpoint['epoch']} 轮恢复训练[/green]")
    
    # 显示训练开始信息
    console.print(Panel(f"[bold blue]开始训练 GPT 模型[/bold blue]\n"
                       f"总轮数: {num_epochs}\n"
                       f"开始轮数: {start_epoch + 1}\n"
                       f"学习率: {learning_rate}\n"
                       f"设备: {device}", title="训练配置", border_style="blue"))
    
    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段 - 使用Rich进度条
        model.train()
        train_loss = 0.0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            train_task = progress.add_task(f"[cyan]第 {epoch+1}/{num_epochs} 轮 - 训练", total=len(train_dataloader))
            
            for batch_idx, batch in enumerate(train_dataloader):
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
                current_ppl = torch.exp(loss).item()
                
                # 更新进度条
                progress.update(train_task, advance=1, 
                              description=f"[cyan]第 {epoch+1}/{num_epochs} 轮 - 训练 Loss: {loss.item():.4f} PPL: {current_ppl:.2f}")
        
        # 验证阶段 - 使用Rich进度条
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                val_task = progress.add_task(f"[yellow]第 {epoch+1}/{num_epochs} 轮 - 验证", total=len(val_dataloader))
                
                for batch_idx, batch in enumerate(val_dataloader):
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
                    current_ppl = torch.exp(loss).item()
                    
                    # 更新进度条
                    progress.update(val_task, advance=1,
                                  description=f"[yellow]第 {epoch+1}/{num_epochs} 轮 - 验证 Loss: {loss.item():.4f} PPL: {current_ppl:.2f}")
        
        # 计算平均损失和困惑度
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        avg_train_ppl = torch.exp(torch.tensor(avg_train_loss))
        avg_val_ppl = torch.exp(torch.tensor(avg_val_loss))
        
        # 计算时间统计
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - train_start_time
        remaining_epochs = num_epochs - epoch - 1
        eta = remaining_epochs * (total_elapsed / (epoch - start_epoch + 1)) if epoch > start_epoch else 0
        
        # 格式化时间显示
        elapsed_str = str(datetime.timedelta(seconds=int(total_elapsed)))
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        
        # 显示详细统计信息
        display_training_stats(
            epoch + 1, num_epochs, avg_train_loss, avg_val_loss,
            avg_train_ppl, avg_val_ppl, best_val_loss,
            scheduler.get_last_lr()[0], elapsed_str, eta_str
        )
        
        # 保存检查点
        save_checkpoint(model, optimizer, scheduler, epoch, avg_train_loss, avg_val_loss, best_val_loss, checkpoint_dir)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 保存为HuggingFace标准格式
            best_model_dir = 'outputs/best_model'
            os.makedirs(best_model_dir, exist_ok=True)
            
            # 保存模型权重 - 使用内部的transformer模型
            try:
                model.transformer.save_pretrained(best_model_dir)
            except Exception as e:
                logger.warning(f"保存模型时出现警告: {e}")
                # 使用torch.save作为备选方案
                torch.save(model.state_dict(), os.path.join(best_model_dir, 'pytorch_model.bin'))
                logger.info("使用torch.save保存模型权重")
            
            # 保存tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(best_model_dir)
            
            # 保存训练配置
            if config is not None:
                try:
                    config.save_pretrained(best_model_dir)
                except Exception as e:
                    logger.warning(f"保存配置时出现警告: {e}")
                    # 手动保存配置
                    config_dict = config.to_dict()
                    with open(os.path.join(best_model_dir, 'config.json'), 'w') as f:
                        json.dump(config_dict, f, indent=2)
                    logger.info("手动保存配置文件")
            
            console.print(f"[green]🎉 新的最佳模型！验证损失: {best_val_loss:.4f}, 困惑度: {avg_val_ppl:.2f}[/green]")
            logger.info(f'保存最佳模型到 {best_model_dir}, 验证损失: {best_val_loss:.4f}, 困惑度: {avg_val_ppl:.2f}')
        
        scheduler.step()
        
        # 添加分隔线
        console.print("─" * 80)
    
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
    console.print(Panel("[bold green]开始训练 GPT 模型[/bold green]", border_style="green"))
    
    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        tokenizer=tokenizer,
        config=config,
        resume_from_checkpoint=True
    )
    
    # 保存最终模型为HuggingFace标准格式
    final_model_dir = 'outputs/final_model'
    os.makedirs(final_model_dir, exist_ok=True)
    
    # 保存模型权重
    try:
        trained_model.transformer.save_pretrained(final_model_dir)
    except Exception as e:
        logger.warning(f"保存最终模型时出现警告: {e}")
        # 使用torch.save作为备选方案
        torch.save(trained_model.state_dict(), os.path.join(final_model_dir, 'pytorch_model.bin'))
        logger.info("使用torch.save保存最终模型权重")
    # 保存tokenizer
    tokenizer.save_pretrained(final_model_dir)
    # 保存训练配置
    try:
        config.save_pretrained(final_model_dir)
    except Exception as e:
        logger.warning(f"保存配置时出现警告: {e}")
        # 手动保存配置
        config_dict = config.to_dict()
        with open(os.path.join(final_model_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info("手动保存配置文件")
    
    # 显示训练完成信息
    console.print(Panel(f"[bold green]🎉 训练完成！[/bold green]\n\n"
                       f"📁 最佳模型: outputs/best_model/\n"
                       f"📁 最终模型: {final_model_dir}/\n"
                       f"💡 现在可以使用 from_pretrained() 直接加载模型！\n"
                       f"🚀 运行推理: python inference.py --model_path outputs/best_model",
                       title="训练完成", border_style="green"))
    
    logger.info(f"训练完成！模型已保存为HuggingFace标准格式到 {final_model_dir}")
    logger.info("现在可以使用 from_pretrained() 直接加载模型！")

if __name__ == "__main__":
    main()
