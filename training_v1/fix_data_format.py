#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正数据格式，解决监督目标对齐问题
将 ... </IMG> <CLS> <CLS> <EOS> 修正为 ... </IMG> <CLS> <CLS_y> <EOS>
"""

import json
import os
from config.model_config import get_config

def fix_data_format(input_path: str, output_path: str):
    """修正数据格式"""
    config = get_config()
    
    print(f"🔧 开始修正数据格式...")
    print(f"📥 输入文件: {input_path}")
    print(f"📤 输出文件: {output_path}")
    
    fixed_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_no, line in enumerate(f_in, 1):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line)
                total_count += 1
                
                # 获取原始tokens和标签
                tokens_str = item['tokens']
                label = item['label']
                tokens = tokens_str.split()
                
                # 检查并修正格式
                if len(tokens) >= 3:
                    # 查找 </IMG> <CLS> <CLS> <EOS> 模式
                    img_end_token = config.special_tokens.img_end_token
                    cls_token = config.special_tokens.cls_token
                    eos_token = config.special_tokens.eos_token
                    expected_cls_token = f"<CLS_{label}>"
                    
                    # 找到</IMG>的位置
                    img_end_pos = -1
                    for i, token in enumerate(tokens):
                        if token == img_end_token:
                            img_end_pos = i
                            break
                    
                    if img_end_pos != -1:
                        # 检查是否是错误格式: </IMG> <CLS> <CLS> <EOS>
                        if (img_end_pos + 3 < len(tokens) and
                            tokens[img_end_pos + 1] == cls_token and
                            tokens[img_end_pos + 2] == cls_token and
                            tokens[img_end_pos + 3] == eos_token):
                            
                            # 修正为: </IMG> <CLS> <CLS_y> <EOS>
                            tokens[img_end_pos + 2] = expected_cls_token
                            fixed_count += 1
                            
                            # 更新item
                            item['tokens'] = ' '.join(tokens)
                            
                            if fixed_count <= 5:  # 显示前5个修正示例
                                print(f"✅ 修正样本{line_no}: {cls_token} {cls_token} → {cls_token} {expected_cls_token}")
                
                # 写入修正后的数据
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                if total_count % 100 == 0:
                    print(f"   处理进度: {total_count}条, 修正: {fixed_count}条")
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ 跳过无效JSON行 {line_no}: {e}")
    
    print(f"\n✅ 数据修正完成!")
    print(f"📊 总处理: {total_count}条")
    print(f"🔧 修正: {fixed_count}条")
    print(f"📁 输出文件: {output_path}")

def validate_fixed_data(data_path: str):
    """验证修正后的数据格式"""
    config = get_config()
    
    print(f"\n🔍 验证修正后的数据格式...")
    
    correct_format = 0
    total_checked = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            if total_checked >= 100:  # 只检查前100个
                break
                
            try:
                item = json.loads(line)
                tokens = item['tokens'].split()
                label = item['label']
                expected_cls_token = f"<CLS_{label}>"
                
                # 检查格式: ... </IMG> <CLS> <CLS_y> <EOS>
                if len(tokens) >= 4:
                    img_end_token = config.special_tokens.img_end_token
                    cls_token = config.special_tokens.cls_token
                    eos_token = config.special_tokens.eos_token
                    
                    # 从后往前检查
                    if (tokens[-1] == eos_token and
                        tokens[-2] == expected_cls_token and
                        tokens[-3] == cls_token and
                        img_end_token in tokens):
                        correct_format += 1
                
                total_checked += 1
                
            except json.JSONDecodeError:
                continue
    
    accuracy = correct_format / total_checked if total_checked > 0 else 0
    print(f"📊 格式验证结果: {correct_format}/{total_checked} = {accuracy:.1%}")
    
    if accuracy > 0.95:
        print("✅ 数据格式修正成功！")
    else:
        print("⚠️ 仍有部分数据格式不正确")

if __name__ == "__main__":
    # 修正数据格式
    input_file = "generated_sequences_super_enhanced/sequences_labels_fixed.jsonl"
    output_file = "generated_sequences_super_enhanced/sequences_labels_fixed_v2.jsonl"
    
    fix_data_format(input_file, output_file)
    validate_fixed_data(output_file)
    
    print(f"\n💡 使用修正后的数据文件进行训练:")
    print(f"   --data_path {output_file}")
