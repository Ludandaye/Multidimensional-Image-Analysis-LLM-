#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据训练数据绘制训练结果图表
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import json
import os
from datetime import datetime
import re

# 设置美观的图表样式
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

def extract_training_data():
    """从训练日志中提取训练数据"""
    # 从检查点获取训练历史
    checkpoint_path = 'outputs/checkpoints/checkpoint.pkl'
    if os.path.exists(checkpoint_path):
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ 从检查点加载数据，轮次: {checkpoint.get('epoch', 'unknown')}")
    
    # 手动提取训练数据（基于你提供的日志）
    training_data = {
        'epochs': list(range(1, 21)),
        'train_loss': [3.1404, 0.8799, 0.1898, 0.0717, 0.0359, 0.0239, 0.0187, 0.0155, 0.0137, 0.0124, 
                       0.0116, 0.0110, 0.0106, 0.0103, 0.0100, 0.0098, 0.0097, 0.0096, 0.0095, 0.0095],
        'val_loss': [1.6078, 0.2925, 0.0806, 0.0417, 0.0252, 0.0187, 0.0158, 0.0138, 0.0125, 0.0119,
                     0.0115, 0.0110, 0.0105, 0.0105, 0.0103, 0.0102, 0.0101, 0.0101, 0.0100, 0.0100],
        'train_ppl': [23.11, 2.41, 1.21, 1.07, 1.04, 1.02, 1.02, 1.02, 1.01, 1.01,
                      1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01],
        'val_ppl': [4.99, 1.34, 1.08, 1.04, 1.03, 1.02, 1.02, 1.01, 1.01, 1.01,
                    1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01],
        'learning_rate': [5.00e-05, 5.00e-05, 4.97e-05, 4.88e-05, 4.73e-05, 4.52e-05, 4.27e-05, 3.97e-05,
                          3.63e-05, 3.27e-05, 2.89e-05, 2.50e-05, 2.11e-05, 1.73e-05, 1.37e-05, 1.03e-05,
                          7.32e-06, 4.77e-06, 2.72e-06, 1.22e-06]
    }
    
    return training_data

def create_training_plots(data):
    """创建训练结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('GPT Model Training Results Analysis', fontsize=24, fontweight='bold', y=0.95)
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(data['epochs'], data['train_loss'], color=colors[0], marker='o', 
             label='Training Loss', linewidth=3, markersize=8, alpha=0.8)
    ax1.plot(data['epochs'], data['val_loss'], color=colors[1], marker='s', 
             label='Validation Loss', linewidth=3, markersize=8, alpha=0.8)
    ax1.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss Value', fontsize=14, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, framealpha=0.9)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Training and Validation Perplexity
    ax2 = axes[0, 1]
    ax2.plot(data['epochs'], data['train_ppl'], color=colors[2], marker='o', 
             label='Training PPL', linewidth=3, markersize=8, alpha=0.8)
    ax2.plot(data['epochs'], data['val_ppl'], color=colors[3], marker='s', 
             label='Validation PPL', linewidth=3, markersize=8, alpha=0.8)
    ax2.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Perplexity (PPL)', fontsize=14, fontweight='bold')
    ax2.set_title('Training and Validation Perplexity', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, framealpha=0.9)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Learning Rate Schedule
    ax3 = axes[1, 0]
    ax3.plot(data['epochs'], data['learning_rate'], color='#6A4C93', marker='o', 
             linewidth=3, markersize=8, alpha=0.8)
    ax3.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
    ax3.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=16, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.4)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Loss Improvement Analysis
    ax4 = axes[1, 1]
    initial_loss = data['val_loss'][0]
    improvement = [(initial_loss - loss) / initial_loss * 100 for loss in data['val_loss']]
    ax4.plot(data['epochs'], improvement, color='#1982C4', marker='o', 
             linewidth=3, markersize=8, alpha=0.8)
    ax4.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Loss Improvement (%)', fontsize=14, fontweight='bold')
    ax4.set_title('Validation Loss Improvement', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.4)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add final improvement annotation
    final_improvement = improvement[-1]
    ax4.annotate(f'Final Improvement: {final_improvement:.1f}%', 
                xy=(data['epochs'][-1], final_improvement),
                xytext=(data['epochs'][-1]-4, final_improvement+8),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=3, alpha=0.8),
                fontsize=12, color='#E74C3C', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout(pad=3.0)
    return fig

def create_performance_summary(data):
    """创建性能总结图表"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Training Performance Summary', fontsize=24, fontweight='bold', y=0.95)
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Final Performance Metrics
    ax1 = axes[0]
    metrics = ['Train Loss', 'Val Loss', 'Train PPL', 'Val PPL']
    final_values = [data['train_loss'][-1], data['val_loss'][-1], 
                   data['train_ppl'][-1], data['val_ppl'][-1]]
    
    bars = ax1.bar(metrics, final_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Final Training Metrics', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 移除顶部和右侧边框
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Convergence Speed Analysis
    ax2 = axes[1]
    # 找到损失降到0.1以下的轮次
    convergence_epoch = None
    for i, loss in enumerate(data['val_loss']):
        if loss < 0.1:
            convergence_epoch = i + 1
            break
    
    if convergence_epoch:
        wedges, texts, autotexts = ax2.pie([convergence_epoch, 20-convergence_epoch], 
               labels=[f'Convergence\nEpoch {convergence_epoch}', f'Stabilization\nEpochs {convergence_epoch+1}-20'],
               colors=['#2ECC71', '#3498DB'], autopct='%1.0f%%', startangle=90,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title(f'Convergence Speed Analysis\n(Converged at Epoch {convergence_epoch})', 
                     fontsize=16, fontweight='bold')
        
        # 美化饼图文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, 'Not converged in 20 epochs', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14, fontweight='bold')
        ax2.set_title('Convergence Speed Analysis', fontsize=16, fontweight='bold')
    
    # 3. Training Efficiency Analysis
    ax3 = axes[2]
    # 计算每轮的损失改善
    improvements = []
    for i in range(1, len(data['val_loss'])):
        improvement = (data['val_loss'][i-1] - data['val_loss'][i]) / data['val_loss'][i-1] * 100
        improvements.append(improvement)
    
    ax3.plot(range(2, 21), improvements, color='#E67E22', marker='o', 
             linewidth=3, markersize=8, alpha=0.8)
    ax3.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Improvement per Epoch (%)', fontsize=14, fontweight='bold')
    ax3.set_title('Training Efficiency Analysis', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.4)
    
    # 添加平均改善率标注
    avg_improvement = np.mean(improvements)
    ax3.axhline(y=avg_improvement, color='#E74C3C', linestyle='--', alpha=0.8, linewidth=2,
                label=f'Average Improvement: {avg_improvement:.1f}%')
    ax3.legend(fontsize=11, framealpha=0.9)
    
    # 移除顶部和右侧边框
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=3.0)
    return fig

def save_plots():
    """保存所有图表"""
    data = extract_training_data()
    
    # 创建输出目录
    os.makedirs('training_plots', exist_ok=True)
    
    # 1. 训练过程图表
    fig1 = create_training_plots(data)
    fig1.savefig('training_plots/training_process.png', dpi=300, bbox_inches='tight')
    print("✅ 训练过程图表已保存: training_plots/training_process.png")
    
    # 2. 性能总结图表
    fig2 = create_performance_summary(data)
    fig2.savefig('training_plots/performance_summary.png', dpi=300, bbox_inches='tight')
    print("✅ 性能总结图表已保存: training_plots/performance_summary.png")
    
    # 3. 保存数据到JSON文件
    with open('training_plots/training_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("✅ 训练数据已保存: training_plots/training_data.json")
    
    # 4. 生成训练报告
    generate_training_report(data)
    
    plt.show()

def generate_training_report(data):
    """生成训练报告"""
    report = f"""
# GPT模型训练报告

## 训练概览
- **总训练轮数**: {len(data['epochs'])}
- **最终训练损失**: {data['train_loss'][-1]:.4f}
- **最终验证损失**: {data['val_loss'][-1]:.4f}
- **最终训练困惑度**: {data['train_ppl'][-1]:.2f}
- **最终验证困惑度**: {data['val_ppl'][-1]:.2f}

## 性能分析
- **初始验证损失**: {data['val_loss'][0]:.4f}
- **最终验证损失**: {data['val_loss'][-1]:.4f}
- **总改善**: {((data['val_loss'][0] - data['val_loss'][-1]) / data['val_loss'][0] * 100):.2f}%

## 收敛分析
- **收敛轮次**: 约第{next((i+1 for i, loss in enumerate(data['val_loss']) if loss < 0.1), '未收敛')}轮
- **稳定性能**: 困惑度稳定在1.01左右
- **无过拟合**: 训练和验证损失基本一致

## 训练效率
- **学习率调度**: 余弦退火，从{data['learning_rate'][0]:.2e}降到{data['learning_rate'][-1]:.2e}
- **平均每轮改善**: {np.mean([(data['val_loss'][i-1] - data['val_loss'][i]) / data['val_loss'][i-1] * 100 for i in range(1, len(data['val_loss']))]):.1f}%

## 结论
模型训练非常成功，验证损失从{data['val_loss'][0]:.4f}降到{data['val_loss'][-1]:.4f}，困惑度从{data['val_ppl'][0]:.2f}降到{data['val_ppl'][-1]:.2f}，达到了理想的训练效果。
"""
    
    with open('training_plots/training_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✅ 训练报告已保存: training_plots/training_report.md")

if __name__ == "__main__":
    print("📊 Starting to generate training result charts...")
    save_plots()
    print("\n🎉 All charts and reports generated successfully!")
    print("📁 Output directory: training_plots/")
    print("📈 Chart files:")
    print("   - training_process.png (Training Process Charts)")
    print("   - performance_summary.png (Performance Summary Charts)")
    print("   - training_data.json (Training Data)")
    print("   - training_report.md (Training Report)")
