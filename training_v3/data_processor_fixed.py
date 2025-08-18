#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的数据处理器
解决监督目标对齐、截断策略、数据泄漏等问题
"""

import json
import torch
import hashlib
import random
from typing import List, Dict, Tuple, Any
from torch.utils.data import Dataset
from config.model_config import UnifiedConfig
import logging

logger = logging.getLogger(__name__)

class FixedCausalLMDataset(Dataset):
    """修复后的因果语言模型数据集"""
    
    def __init__(self, data_path: str, config: UnifiedConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.max_length = config.model.max_length
        
        # 加载和处理数据
        self.data = self.load_and_process_data(data_path)
        
        # 验证数据格式
        self.validate_data()
        
        logger.info(f"✅ {split}集加载完成: {len(self.data)}条样本")
    
    def load_and_process_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载并处理数据，解决数据泄漏问题"""
        # 1. 加载所有数据
        all_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        # 为每个样本生成唯一hash用于划分
                        item['sample_hash'] = self._generate_sample_hash(item, line_no)
                        all_data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效JSON行 {line_no}: {e}")
        
        logger.info(f"📥 总共加载 {len(all_data)} 条数据")
        
        # 2. 去重（基于内容hash）
        unique_data = {}
        for item in all_data:
            content_hash = self._generate_content_hash(item['tokens'])
            if content_hash not in unique_data:
                unique_data[content_hash] = item
            else:
                logger.debug(f"发现重复样本，已跳过")
        
        deduplicated_data = list(unique_data.values())
        logger.info(f"🔄 去重后剩余 {len(deduplicated_data)} 条数据")
        
        # 3. 按hash值稳定划分训练/验证集
        train_data, val_data = self._split_data_by_hash(deduplicated_data)
        
        if self.split == "train":
            return train_data
        else:
            return val_data
    
    def _generate_sample_hash(self, item: Dict, line_no: int) -> str:
        """为样本生成唯一hash"""
        # 使用文件名、标签、行号生成稳定的hash
        content = f"{item.get('meta', {}).get('filename', '')}-{item.get('label', '')}-{line_no}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_content_hash(self, tokens: str) -> str:
        """为token序列生成内容hash用于去重"""
        return hashlib.md5(tokens.encode()).hexdigest()
    
    def _split_data_by_hash(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """基于hash值稳定划分数据集"""
        train_data = []
        val_data = []
        
        for item in data:
            # 使用hash的最后一位决定划分（确保稳定且平衡）
            hash_int = int(item['sample_hash'][-1], 16)  # 0-15
            if hash_int < 13:  # 约80%用于训练
                train_data.append(item)
            else:  # 约20%用于验证
                val_data.append(item)
        
        logger.info(f"📊 数据划分: 训练集{len(train_data)}条, 验证集{len(val_data)}条")
        return train_data, val_data
    
    def validate_data(self):
        """验证数据格式的正确性"""
        issues = []
        
        for i, item in enumerate(self.data[:100]):  # 检查前100个样本
            tokens = item['tokens'].split()
            
            # 检查必要的token
            if self.config.special_tokens.cls_token not in tokens:
                issues.append(f"样本{i}: 缺少<CLS>标记")
            
            if self.config.special_tokens.eos_token not in tokens:
                issues.append(f"样本{i}: 缺少<EOS>标记")
            
            # 检查分类标签格式
            label = item.get('label')
            expected_cls_token = f"<CLS_{label}>"
            if expected_cls_token not in tokens:
                issues.append(f"样本{i}: 缺少分类标签{expected_cls_token}")
            
            # 检查序列格式: ... <CLS> <CLS_y> <EOS>
            if len(tokens) >= 3:
                if (tokens[-3] == self.config.special_tokens.cls_token and
                    tokens[-2] == expected_cls_token and
                    tokens[-1] == self.config.special_tokens.eos_token):
                    continue  # 格式正确
                else:
                    issues.append(f"样本{i}: 尾部格式不正确，应为 <CLS> <CLS_y> <EOS>")
        
        if issues:
            logger.warning(f"⚠️ 数据验证发现 {len(issues)} 个问题:")
            for issue in issues[:10]:  # 只显示前10个问题
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ 数据格式验证通过")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens_str = item['tokens']
        label = item['label']
        
        # 转换为token列表
        tokens = tokens_str.split()
        
        # 应用左截断策略，保留关键尾部
        processed_tokens = self._apply_left_truncation(tokens)
        
        # 转换为ID
        input_ids = self._tokens_to_ids(processed_tokens)
        
        # 创建标签（用于因果语言建模）
        labels = self._create_causal_labels(input_ids, processed_tokens, label)
        
        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)
        
        # 填充到固定长度
        input_ids, labels, attention_mask = self._pad_sequences(
            input_ids, labels, attention_mask
        )
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'true_label': label,
            'sample_id': item.get('sample_hash', f'{idx}')
        }
    
    def _apply_left_truncation(self, tokens: List[str]) -> List[str]:
        """应用左截断策略，确保保留关键尾部"""
        if len(tokens) <= self.max_length:
            return tokens
        
        # 保留最后N个关键token
        preserve_count = self.config.data.preserve_tail_tokens
        
        # 找到关键尾部的开始位置
        tail_start = len(tokens) - preserve_count
        for i in range(len(tokens) - preserve_count, len(tokens)):
            if i >= 0 and tokens[i] == self.config.special_tokens.cls_token:
                tail_start = i
                break
        
        # 计算可用的前部长度
        available_front = self.max_length - (len(tokens) - tail_start)
        
        if available_front <= 0:
            # 如果尾部太长，只保留尾部
            return tokens[tail_start:]
        else:
            # 保留前部分 + 尾部
            return tokens[:available_front] + tokens[tail_start:]
    
    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """将token转换为ID"""
        vocab = self.config.vocab
        unk_id = vocab.get(self.config.special_tokens.unk_token, 1)
        
        return [vocab.get(token, unk_id) for token in tokens]
    
    def _create_causal_labels(self, input_ids: List[int], tokens: List[str], true_label: int) -> List[int]:
        """创建因果语言建模的标签，确保<CLS>后预测<CLS_y>"""
        labels = [-100] * len(input_ids)  # -100表示不计算loss
        
        # 找到<CLS>的位置
        cls_id = self.config.vocab.get(self.config.special_tokens.cls_token, -1)
        cls_label_id = self.config.vocab.get(f"<CLS_{true_label}>", -1)
        
        for i in range(len(input_ids) - 1):
            if input_ids[i] == cls_id and i + 1 < len(input_ids):
                # 确保<CLS>后面预测的是正确的<CLS_y>
                if input_ids[i + 1] == cls_label_id:
                    labels[i] = cls_label_id  # <CLS>位置的标签是<CLS_y>
                    logger.debug(f"设置<CLS>位置{i}的标签为<CLS_{true_label}>")
                break
        
        return labels
    
    def _pad_sequences(self, input_ids: List[int], labels: List[int], attention_mask: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """填充序列到固定长度"""
        pad_id = self.config.vocab.get(self.config.special_tokens.pad_token, 0)
        
        # 截断或填充
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:
            pad_length = self.max_length - len(input_ids)
            input_ids.extend([pad_id] * pad_length)
            labels.extend([-100] * pad_length)  # 填充位置不计算loss
            attention_mask.extend([0] * pad_length)  # 填充位置不参与注意力
        
        return input_ids, labels, attention_mask

def create_datasets(config: UnifiedConfig) -> Tuple[FixedCausalLMDataset, FixedCausalLMDataset]:
    """创建训练和验证数据集"""
    train_dataset = FixedCausalLMDataset(
        data_path=config.data.train_data_path,
        config=config,
        split="train"
    )
    
    val_dataset = FixedCausalLMDataset(
        data_path=config.data.train_data_path,
        config=config,
        split="val"
    )
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    # 测试数据处理器
    from config.model_config import get_config
    
    logging.basicConfig(level=logging.INFO)
    
    config = get_config()
    train_dataset, val_dataset = create_datasets(config)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 测试一个样本
    sample = train_dataset[0]
    print(f"样本形状:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # 验证<CLS>位置的标签设置
    input_ids = sample['input_ids'].tolist()
    labels = sample['labels'].tolist()
    
    cls_id = config.vocab[config.special_tokens.cls_token]
    for i, (inp_id, label) in enumerate(zip(input_ids, labels)):
        if inp_id == cls_id and label != -100:
            cls_token_name = None
            for token, token_id in config.vocab.items():
                if token_id == label:
                    cls_token_name = token
                    break
            print(f"✅ <CLS>在位置{i}，标签为{cls_token_name} (ID: {label})")
