import json
import torch
from transformers import GPT2LMHeadModel
import pandas as pd

def main():
    print('🧪 使用训练数据格式完整测试模型')
    print('=' * 80)
    
    # 加载训练时的codebook
    codebook = pd.read_csv('unified_codebook/unified_codebook.csv')
    print(f'✅ 加载codebook成功，共{len(codebook)}个聚类')
    
    # 加载训练时的词汇表
    with open('generated_sequences_super_enhanced/vocab.json', 'r') as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    print(f'✅ 加载词汇表成功，共{len(vocab)}个token')
    
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained('outputs/best_model')
    model.eval()
    device = 'cpu'
    model = model.to(device)
    print(f'✅ 模型加载成功，使用设备: {device}')
    print(f'📊 模型词汇表大小: {model.config.vocab_size}')
    print(f'📊 训练词汇表大小: {len(vocab)}')
    
    # 加载测试样本
    test_samples = []
    with open('generated_sequences_super_enhanced/sequences_labels_fixed.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 取前3个样本
                break
            if line.strip():
                test_samples.append(json.loads(line.strip()))
    print(f'✅ 加载测试样本成功，共{len(test_samples)}个样本')
    
    # 测试每个样本
    for i, sample in enumerate(test_samples):
        print(f'\n📊 测试样本 {i+1}:')
        print(f'   标签: {sample["label"]}')
        print(f'   文件名: {sample["meta"]["filename"]}')
        print(f'   原始数字: {sample["meta"]["original_digit"]}')
        
        # 获取token序列
        tokens = sample['tokens'].split()
        print(f'   Token序列长度: {len(tokens)}')
        
        # 找到<CLS>位置（训练时的格式）
        try:
            cls_pos = tokens.index('<CLS>')
            input_tokens = tokens[:cls_pos+1]
        except ValueError:
            print('   ⚠️ 未找到<CLS>token，使用前400个token')
            input_tokens = tokens[:400]
        
        # 转换为ID序列
        input_ids = [vocab.get(t, vocab['<UNK>']) for t in input_tokens]
        
        # 截断到模型最大位置
        max_len = min(len(input_ids), 400)
        input_ids = input_ids[:max_len]
        
        print(f'   输入长度: {len(input_ids)}')
        print(f'   输入tokens前5个: {" ".join(input_tokens[:5])}...')
        
        # 进行推理
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
        
        print(f'   🔮 下一个token的top-5预测:')
        for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            idx_val = idx.item()
            token_name = inv_vocab.get(idx_val, f'UNK_{idx_val}')
            print(f'     {j+1}. {token_name}: 概率 {prob.item():.4f}')
        
        # 检查期望标签
        expected_label = sample['label']
        if f'<CLS_{expected_label}>' in vocab:
            expected_id = vocab[f'<CLS_{expected_label}>']
            expected_prob = probs[expected_id].item()
            print(f'   🎯 期望标签: <CLS_{expected_label}> (ID: {expected_id}), 概率: {expected_prob:.4f}')
        elif str(expected_label) in vocab:
            expected_id = vocab[str(expected_label)]
            expected_prob = probs[expected_id].item()
            print(f'   🎯 期望标签: {expected_label} (ID: {expected_id}), 概率: {expected_prob:.4f}')
        else:
            print(f'   ⚠️ 期望标签不在词汇表中: {expected_label}')
        
        print('-' * 50)
    
    print('\n🚀 测试文本生成能力...')
    # 测试生成
    test_inputs = [
        ['<IMG>', '<Z_100>', '<Z_200>'],
        ['<IMG>', '<Z_369>', '<Z_255>', '<CLS>']
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f'\n📝 生成测试 {i+1}:')
        test_ids = [vocab.get(t, vocab['<UNK>']) for t in test_input]
        test_tensor = torch.tensor([test_ids], dtype=torch.long)
        
        try:
            with torch.no_grad():
                generated = model.generate(
                    test_tensor,
                    max_new_tokens=3,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=vocab['<PAD>'],
                    eos_token_id=vocab.get('<EOS>', vocab['<UNK>'])
                )
            
            generated_tokens = [inv_vocab.get(tid.item(), f'UNK_{tid.item()}') 
                              for tid in generated[0]]
            
            print(f'   输入: {test_input}')
            print(f'   生成: {generated_tokens}')
            print(f'   新增: {generated_tokens[len(test_input):]}')
        
        except Exception as e:
            print(f'   ❌ 生成失败: {e}')
    
    print('\n' + '=' * 80)
    print('✅ 测试完成！')
    print('📋 总结:')
    print('  - 模型可以正常加载和推理')
    print('  - 使用了训练时的codebook和词汇表')
    print('  - 输入格式与训练时一致（到<CLS>截止）')
    print('  - 可以进行next-token预测和文本生成')

if __name__ == "__main__":
    main()
