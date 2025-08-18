#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的模型配置文件
解决特殊符号不一致问题，确保训练、推理、评估都使用相同的配置
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class SpecialTokensConfig:
    """特殊token配置"""
    # 基础特殊token（根据实际词汇表）
    pad_token: str = "<PAD>"      # ID: 0
    unk_token: str = "<UNK>"      # ID: 1
    eos_token: str = "<EOS>"      # ID: 5
    
    # 图像相关token
    img_start_token: str = "<IMG>"    # ID: 2
    img_end_token: str = "</IMG>"     # ID: 3
    
    # 分类相关token
    cls_token: str = "<CLS>"          # ID: 4
    cls_tokens: List[str] = None      # <CLS_0>, <CLS_1>, ..., <CLS_9> (ID: 506-515)
    
    def __post_init__(self):
        if self.cls_tokens is None:
            self.cls_tokens = [f"<CLS_{i}>" for i in range(10)]
    
    def get_all_special_tokens(self) -> List[str]:
        """获取所有特殊token列表"""
        tokens = [
            self.pad_token,
            self.unk_token,
            self.eos_token,
            self.img_start_token,
            self.img_end_token,
            self.cls_token
        ]
        tokens.extend(self.cls_tokens)
        return tokens
    
    def get_token_ids(self, vocab: Dict[str, int]) -> Dict[str, int]:
        """获取特殊token的ID映射"""
        token_ids = {}
        for token_name, token_value in asdict(self).items():
            if isinstance(token_value, str) and token_value in vocab:
                token_ids[token_name] = vocab[token_value]
            elif isinstance(token_value, list):
                token_ids[token_name] = [vocab.get(t, vocab.get(self.unk_token, 1)) for t in token_value]
        return token_ids

@dataclass
class ModelArchConfig:
    """模型架构配置"""
    model_type: str = "gpt2"
    vocab_size: int = 516  # 实际词汇表大小
    n_positions: int = 512
    n_ctx: int = 512
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 8
    max_length: int = 512
    
    # 训练相关
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 30
    warmup_steps: int = 100

@dataclass
class DataConfig:
    """数据配置"""
    # 数据文件路径
    train_data_path: str = "generated_sequences_super_enhanced/sequences_labels_fixed_v2.jsonl"
    vocab_path: str = "generated_sequences_super_enhanced/vocab.json"
    codebook_path: str = "unified_codebook/unified_codebook.csv"
    
    # 数据处理参数
    max_length: int = 512
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    
    # 截断策略：从左边截断，保留关键尾部
    truncation_strategy: str = "left"  # "left" 或 "right"
    preserve_tail_tokens: int = 10  # 保留最后N个token

@dataclass
class ExperimentConfig:
    """实验配置和元信息"""
    experiment_name: str = "training_v2_fixed"
    experiment_version: str = "2.1"
    created_time: str = None
    
    # 数据版本信息
    data_version: str = "sequences_labels_fixed_tail_fixed"
    vocab_version: str = "v1_516_tokens"
    
    # 模型版本信息
    model_version: str = "gpt2_6layer_384dim"
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now().isoformat()

class UnifiedConfig:
    """统一配置管理器"""
    
    def __init__(self):
        self.special_tokens = SpecialTokensConfig()
        self.model = ModelArchConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
        
        # 加载词汇表
        self.vocab = self._load_vocab()
        
        # 更新模型配置中的词汇表大小
        self.model.vocab_size = len(self.vocab)
        
        # 获取特殊token的ID映射
        self.token_ids = self.special_tokens.get_token_ids(self.vocab)
    
    def _load_vocab(self) -> Dict[str, int]:
        """加载词汇表"""
        vocab_path = self.data.vocab_path
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")
    
    def validate_config(self) -> bool:
        """验证配置的一致性"""
        issues = []
        
        # 检查特殊token是否都在词汇表中
        for token in self.special_tokens.get_all_special_tokens():
            if token not in self.vocab:
                issues.append(f"特殊token '{token}' 不在词汇表中")
        
        # 检查词汇表大小一致性
        if self.model.vocab_size != len(self.vocab):
            issues.append(f"模型词汇表大小({self.model.vocab_size})与实际词汇表大小({len(self.vocab)})不一致")
        
        # 检查分类token是否存在且连续
        cls_token_ids = []
        for cls_token in self.special_tokens.cls_tokens:
            if cls_token in self.vocab:
                cls_token_ids.append(self.vocab[cls_token])
            else:
                issues.append(f"分类token '{cls_token}' 不在词汇表中")
        
        # 检查分类token ID是否连续
        if len(cls_token_ids) == 10:
            cls_token_ids.sort()
            for i in range(1, len(cls_token_ids)):
                if cls_token_ids[i] != cls_token_ids[i-1] + 1:
                    issues.append(f"分类token ID不连续: {cls_token_ids}")
        
        if issues:
            print("❌ 配置验证失败:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ 配置验证通过")
            return True
    
    def save_config(self, save_path: str):
        """保存配置到文件"""
        config_dict = {
            'special_tokens': asdict(self.special_tokens),
            'model': asdict(self.model),
            'data': asdict(self.data),
            'experiment': asdict(self.experiment),
            'token_ids': self.token_ids
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置已保存到: {save_path}")
    
    def load_config(self, config_path: str):
        """从文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 更新配置对象
        self.special_tokens = SpecialTokensConfig(**config_dict['special_tokens'])
        self.model = ModelArchConfig(**config_dict['model'])
        self.data = DataConfig(**config_dict['data'])
        self.experiment = ExperimentConfig(**config_dict['experiment'])
        self.token_ids = config_dict['token_ids']
        
        print(f"✅ 配置已从文件加载: {config_path}")
    
    def print_summary(self):
        """打印配置摘要"""
        print("=" * 80)
        print(f"🔧 实验配置摘要: {self.experiment.experiment_name} v{self.experiment.experiment_version}")
        print("=" * 80)
        
        print(f"📊 模型架构:")
        print(f"  - 类型: {self.model.model_type}")
        print(f"  - 词汇表大小: {self.model.vocab_size}")
        print(f"  - 层数: {self.model.n_layer}, 维度: {self.model.n_embd}, 头数: {self.model.n_head}")
        print(f"  - 最大长度: {self.model.max_length}")
        
        print(f"📝 特殊Token:")
        print(f"  - PAD: {self.special_tokens.pad_token} (ID: {self.token_ids.get('pad_token', 'N/A')})")
        print(f"  - EOS: {self.special_tokens.eos_token} (ID: {self.token_ids.get('eos_token', 'N/A')})")
        print(f"  - CLS: {self.special_tokens.cls_token} (ID: {self.token_ids.get('cls_token', 'N/A')})")
        print(f"  - 分类标签: {self.special_tokens.cls_tokens[:3]}...{self.special_tokens.cls_tokens[-1]}")
        
        print(f"📁 数据配置:")
        print(f"  - 训练数据: {self.data.train_data_path}")
        print(f"  - 词汇表: {self.data.vocab_path}")
        print(f"  - 截断策略: {self.data.truncation_strategy}")
        print(f"  - 保留尾部: {self.data.preserve_tail_tokens} tokens")
        
        print(f"🧪 实验信息:")
        print(f"  - 数据版本: {self.experiment.data_version}")
        print(f"  - 模型版本: {self.experiment.model_version}")
        print(f"  - 创建时间: {self.experiment.created_time}")

# 创建全局配置实例
def get_config() -> UnifiedConfig:
    """获取统一配置实例"""
    return UnifiedConfig()

if __name__ == "__main__":
    # 测试配置
    config = get_config()
    config.validate_config()
    config.print_summary()
    
    # 保存配置示例
    config.save_config("config/unified_config.json")
