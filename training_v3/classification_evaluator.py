#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真正的分类评估器
在<CLS>位置直接进行分类预测，而不是生成式评估
"""

import torch
import numpy as np
import os
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import GPT2LMHeadModel
from config.model_config import UnifiedConfig
from data_processor_fixed import FixedCausalLMDataset
import logging

logger = logging.getLogger(__name__)

class ClassificationEvaluator:
    """分类评估器"""
    
    def __init__(self, model: GPT2LMHeadModel, config: UnifiedConfig, device: str = 'cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        # 获取分类相关的token ID
        self.cls_token_id = config.vocab[config.special_tokens.cls_token]
        self.cls_label_ids = [config.vocab[cls_token] for cls_token in config.special_tokens.cls_tokens]
        
        logger.info(f"✅ 分类评估器初始化完成")
        logger.info(f"   <CLS> token ID: {self.cls_token_id}")
        logger.info(f"   分类标签 IDs: {self.cls_label_ids}")
    
    def predict_single(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        """
        对单个样本进行分类预测
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            predicted_class: 预测的类别 (0-9)
            confidence: 预测置信度
            class_probs: 所有类别的概率分布
        """
        self.model.eval()
        
        with torch.no_grad():
            # 前向传播
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # 找到<CLS>位置
            batch_size = input_ids.size(0)
            cls_positions = []
            
            for b in range(batch_size):
                # 找到最后一个<CLS>的位置
                cls_pos = -1
                for pos in range(input_ids.size(1) - 1, -1, -1):
                    if input_ids[b, pos] == self.cls_token_id:
                        cls_pos = pos
                        break
                cls_positions.append(cls_pos)
            
            # 在<CLS>位置提取分类logits
            predictions = []
            confidences = []
            all_class_probs = []
            
            for b in range(batch_size):
                cls_pos = cls_positions[b]
                if cls_pos == -1:
                    # 没找到<CLS>，随机预测
                    predictions.append(0)
                    confidences.append(0.1)
                    all_class_probs.append(np.ones(10) / 10)
                    continue
                
                # 获取<CLS>位置的logits
                cls_logits = logits[b, cls_pos, :]  # [vocab_size]
                
                # 只考虑分类标签的logits
                class_logits = cls_logits[self.cls_label_ids]  # [10]
                class_probs = torch.softmax(class_logits, dim=-1).cpu().numpy()
                
                # 预测类别和置信度
                predicted_class = np.argmax(class_probs)
                confidence = class_probs[predicted_class]
                
                predictions.append(predicted_class)
                confidences.append(confidence)
                all_class_probs.append(class_probs)
            
            if batch_size == 1:
                return predictions[0], confidences[0], all_class_probs[0]
            else:
                return predictions, confidences, all_class_probs
    
    def evaluate_dataset(self, dataset: FixedCausalLMDataset, batch_size: int = 16) -> Dict:
        """评估整个数据集"""
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_true_labels = []
        all_confidences = []
        all_class_probs = []
        
        logger.info(f"🧪 开始评估数据集，共{len(dataset)}个样本...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                true_labels = batch['true_label'].tolist()
                
                # 批量预测
                predictions, confidences, class_probs = self.predict_single(input_ids, attention_mask)
                
                # 收集结果
                if isinstance(predictions, list):
                    all_predictions.extend(predictions)
                    all_confidences.extend(confidences)
                    all_class_probs.extend(class_probs)
                else:
                    all_predictions.append(predictions)
                    all_confidences.append(confidences)
                    all_class_probs.append(class_probs)
                
                all_true_labels.extend(true_labels)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"   评估进度: {(batch_idx + 1) * batch_size}/{len(dataset)}")
        
        # 计算评估指标
        results = self._compute_metrics(all_true_labels, all_predictions, all_confidences, all_class_probs)
        
        return results
    
    def _compute_metrics(self, true_labels: List[int], predictions: List[int], 
                        confidences: List[float], class_probs: List[np.ndarray]) -> Dict:
        """计算详细的评估指标"""
        
        # 基础指标
        accuracy = accuracy_score(true_labels, predictions)
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions, labels=list(range(10)))
        
        # 分类报告
        class_report = classification_report(true_labels, predictions, 
                                           labels=list(range(10)), 
                                           target_names=[f'数字{i}' for i in range(10)],
                                           output_dict=True)
        
        # 置信度统计
        avg_confidence = np.mean(confidences)
        confidence_by_class = {}
        correct_confidence = []
        incorrect_confidence = []
        
        for true_label, pred_label, conf in zip(true_labels, predictions, confidences):
            if true_label not in confidence_by_class:
                confidence_by_class[true_label] = []
            confidence_by_class[true_label].append(conf)
            
            if true_label == pred_label:
                correct_confidence.append(conf)
            else:
                incorrect_confidence.append(conf)
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'avg_confidence': avg_confidence,
            'correct_confidence_avg': np.mean(correct_confidence) if correct_confidence else 0,
            'incorrect_confidence_avg': np.mean(incorrect_confidence) if incorrect_confidence else 0,
            'confidence_by_class': {k: np.mean(v) for k, v in confidence_by_class.items()},
            'total_samples': len(true_labels),
            'correct_predictions': sum(1 for t, p in zip(true_labels, predictions) if t == p)
        }
        
        return results
    
    def print_evaluation_results(self, results: Dict):
        """打印评估结果"""
        print("=" * 80)
        print("📊 分类评估结果")
        print("=" * 80)
        
        print(f"🎯 总体准确率: {results['accuracy']:.2%} ({results['correct_predictions']}/{results['total_samples']})")
        print(f"🔮 平均置信度: {results['avg_confidence']:.4f}")
        print(f"✅ 正确预测置信度: {results['correct_confidence_avg']:.4f}")
        print(f"❌ 错误预测置信度: {results['incorrect_confidence_avg']:.4f}")
        
        print(f"\n📋 各类别准确率:")
        for i in range(10):
            class_metrics = results['classification_report'][f'数字{i}']
            precision = class_metrics['precision']
            recall = class_metrics['recall']
            f1 = class_metrics['f1-score']
            support = class_metrics['support']
            confidence = results['confidence_by_class'].get(i, 0)
            
            print(f"  数字{i}: P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
                  f"样本={support:2d} 置信度={confidence:.3f}")
        
        print(f"\n📈 混淆矩阵:")
        print("     ", end="")
        for i in range(10):
            print(f"{i:4d}", end="")
        print()
        
        cm = results['confusion_matrix']
        for i in range(10):
            print(f"{i:2d}: ", end="")
            for j in range(10):
                print(f"{cm[i,j]:4d}", end="")
            print()
        
        # 分析最容易混淆的类别
        print(f"\n🔍 最容易混淆的数字对:")
        confusion_pairs = []
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((i, j, cm[i, j]))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        for rank, (true_class, pred_class, count) in enumerate(confusion_pairs[:5], 1):
            print(f"  {rank}. 数字{true_class} → 数字{pred_class}: {count}次")

def evaluate_model(model_path: str, config: UnifiedConfig, device: str = 'cpu'):
    """评估模型性能"""
    
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # 创建评估器
    evaluator = ClassificationEvaluator(model, config, device)
    
    # 创建验证数据集
    from data_processor_fixed import create_datasets
    _, val_dataset = create_datasets(config)
    
    # 评估
    results = evaluator.evaluate_dataset(val_dataset, batch_size=8)
    
    # 打印结果
    evaluator.print_evaluation_results(results)
    
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from config.model_config import get_config
    
    config = get_config()
    
    # 评估当前模型
    if os.path.exists('outputs/best_model'):
        print("🧪 评估当前训练的模型...")
        results = evaluate_model('outputs/best_model', config)
    else:
        print("❌ 未找到训练好的模型，请先训练模型")
