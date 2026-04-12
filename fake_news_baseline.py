"""
Fake News Detection - ModernBERT Baseline
基线模型：预训练 ModernBERT + 平均池化 + MLP 二分类
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time

# ==================== 路径配置 ====================

# 预训练 RoBERTa 模型路径（服务器路径）
ROBERTA_PATH = r"/home/zjh/project/Administrator/utils/roberta-base-token"
MODEL_PATH = r"/home/zjh/project/Administrator/utils/roberta-base-model"

# 数据路径
DATA_PATH = r"C:\Users\Administrator\Desktop\AINEW\fakeexe\fakedata\FakeNewsNet-master\fakeandreal.csv"

# 输出目录
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\AINEW\fakeexe\fakedata\FakeNewsNet-master\baseline_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 设备 ====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================== 数据准备 ====================

print("\n" + "="*60)
print("1. 数据准备")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"原始数据: {len(df)} 条")

# 标签映射：fake=1, real=0
df['label_binary'] = (df['label'] == 'fake').astype(int)

# 处理缺失文本
df['text'] = df['text'].fillna('')
df['title'] = df['title'].fillna('')
df['full_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)

# 过滤空文本
df = df[df['full_text'].str.strip().str.len() > 10].reset_index(drop=True)
print(f"过滤后数据: {len(df)} 条")

# 类别分布
fake_count = (df['label_binary'] == 1).sum()
real_count = (df['label_binary'] == 0).sum()
print(f"Fake: {fake_count} ({fake_count/len(df)*100:.1f}%)")
print(f"Real: {real_count} ({real_count/len(df)*100:.1f}%)")

# 划分训练/验证/测试
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label_binary'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_binary'])

print(f"\n训练集: {len(train_df)}")
print(f"验证集: {len(val_df)}")
print(f"测试集: {len(test_df)}")

# ==================== Tokenizer ====================

print("\n" + "="*60)
print("2. 加载 ModernBERT")
print("="*60)

tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_PATH)
print(f"Tokenizer 加载完成: {ROBERTA_PATH}")

# ==================== 数据集 ====================

class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data =dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['full_text'])[:10000]  # 截断到 10000 字符
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(row['label_binary'], dtype=torch.long)
        }

# ==================== 模型定义 ====================

class ModernBERTBaseline(nn.Module):
    """
    基线模型：预训练 ModernBERT + 平均池化 + MLP 二分类
    
    架构：
    1. ModernBERT 编码文本
    2. 平均池化所有 token 的隐藏状态
    3. MLP 分类器 (fake/real)
    """
    def __init__(self, model_path, hidden_dim=768, dropout=0.3):
        super(ModernBERTBaseline, self).__init__()
        
        # 预训练 RoBERTa（冻结，只提取特征）
        self.bert = RobertaModel.from_pretrained(model_path)
        
        # 冻结 BERT 参数
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 分类头：平均池化特征 + MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        # 统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n模型创建完成:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数 (分类头): {trainable_params:,}")
        print(f"  BERT 参数: {total_params - trainable_params:,} (冻结)")
        
    def forward(self, input_ids, attention_mask):
        # ModernBERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取最后一层隐藏状态: [batch, seq_len, hidden_dim]
        last_hidden_state = outputs.last_hidden_state
        
        # 平均池化（考虑 attention mask）
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask  # [batch, hidden_dim]
        
        # 分类
        logits = self.classifier(pooled_output)
        return logits

# ==================== 创建模型 ====================

print("\n创建基线模型...")
model = ModernBERTBaseline(MODEL_PATH, hidden_dim=768, dropout=0.3)
model = model.to(device)

# ==================== 训练配置 ====================

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3  # 较高学习率，因为只有分类头训练
PATIENCE = 5

train_dataset = FakeNewsDataset(train_df, tokenizer)
val_dataset = FakeNewsDataset(val_df, tokenizer)
test_dataset = FakeNewsDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=False)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 只优化分类头参数
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=0.01
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)

# ==================== 训练函数 ====================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    preds, labels = [], []
    
    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(label.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, probs, labels = [], [], []
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        
        logits = model(input_ids, attention_mask)
        prob = F.softmax(logits, dim=1)[:, 1]  # fake 的概率
        
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        probs.extend(prob.cpu().numpy())
        labels.extend(label.cpu().numpy())
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_binary = f1_score(labels, preds, average='binary')
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    
    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_binary': f1_binary,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }

# ==================== 训练主循环 ====================

print("\n" + "="*60)
print("3. 开始训练")
print("="*60)

best_f1 = 0
best_epoch = 0
patience_counter = 0
history = []

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # 训练
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
    
    # 验证
    val_metrics = evaluate(model, val_loader)
    
    epoch_time = time.time() - epoch_start
    
    print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s)")
    print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"  Val   - Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1_macro']:.4f}, AUC: {val_metrics['auc']:.4f}")
    
    history.append({
        'epoch': epoch+1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        **{k: v for k, v in val_metrics.items() if k != 'confusion_matrix'}
    })
    
    # 保存最佳模型
    if val_metrics['f1_macro'] > best_f1:
        best_f1 = val_metrics['f1_macro']
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'best_f1': best_f1,
            'val_metrics': val_metrics
        }, os.path.join(OUTPUT_DIR, 'best_baseline_model.pth'))
        print(f"  ✅ 保存最佳模型 (F1: {best_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  ❌ 无提升 ({patience_counter}/{PATIENCE})")
        
        if patience_counter >= PATIENCE:
            print(f"\n早停触发！最佳 F1: {best_f1:.4f} @ Epoch {best_epoch}")
            break
    
    scheduler.step(val_metrics['f1_macro'])

# ==================== 测试集评估 ====================

print("\n" + "="*60)
print("4. 测试集评估")
print("="*60)

checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_baseline_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
test_metrics = evaluate(model, test_loader)

print(f"\n测试集结果:")
print(f"  Accuracy:   {test_metrics['acc']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")
print(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
print(f"  F1 (Binary): {test_metrics['f1_binary']:.4f}")
print(f"  AUC:       {test_metrics['auc']:.4f}")

cm = np.array(test_metrics['confusion_matrix'])
print(f"\n混淆矩阵 (行=真实, 列=预测):")
print(f"              Pred Fake  Pred Real")
print(f"  Actual Fake    {cm[1,1]:5d}     {cm[1,0]:5d}")
print(f"  Actual Real    {cm[0,1]:5d}     {cm[0,0]:5d}")

# ==================== 保存结果 ====================

results = {
    'best_epoch': best_epoch,
    'best_val_f1': best_f1,
    'test_metrics': test_metrics,
    'history': history,
    'config': {
        'model_path': MODEL_PATH,
        'tokenizer_path': ROBERTA_PATH,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr': LR,
        'pooling': 'mean_pooling',
        'hidden_dim': 768,
        'dropout': 0.3
    }
}

with open(os.path.join(OUTPUT_DIR, 'baseline_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存至: {OUTPUT_DIR}")
print("="*60)
