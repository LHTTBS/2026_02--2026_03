#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
modle_third_result_plot.py

针对 nineteen_training_history.csv 的训练日志可视化脚本。
模仿 pnglast.py 的图表风格，但适配 nineteen.csv 的实际字段。
生成三组图表：
  1. 2x3 六子图布局 (training_analysis_6plots_nineteen.png)
  2. 2x2 四子图布局 (training_analysis_4plots_nineteen.png)
  3. 关键指标汇总图 (training_summary_nineteen.png)

字段说明：
  epoch, train_loss, val_loss, lr,
  stance_acc, intent_f1, harmfulness_acc, fairness_acc, psych_f1
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# -------------------- 全局样式设置 --------------------
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300
})
sns.set_style("whitegrid")

# -------------------- 读取数据 --------------------
df = pd.read_csv('20260210\ineteen_training_history.csv')
print(f"数据加载成功：{len(df)} 个 epoch")

# 为了方便引用，定义列名常量（与csv列完全一致）
EPOCH = 'epoch'
TRAIN_LOSS = 'train_loss'
VAL_LOSS = 'val_loss'
LR = 'lr'
STANCE_ACC = 'stance_acc'
HARMFULNESS_ACC = 'harmfulness_acc'
FAIRNESS_ACC = 'fairness_acc'
INTENT_F1 = 'intent_f1'
PSYCH_F1 = 'psych_f1'

# ==================== 1. 六子图布局 (2x3) ====================
print("\n生成 6 子图布局...")
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
fig1.suptitle('Training Analysis - Nineteen (6 Plots)', fontsize=16, fontweight='bold', y=0.98)

# ---- 子图1：训练/验证损失 ----
ax1 = axes1[0, 0]
ax1.plot(df[EPOCH], df[TRAIN_LOSS], 'b-', linewidth=2, label='Train Loss',
         marker='o', markersize=3, markevery=5)
ax1.plot(df[EPOCH], df[VAL_LOSS], 'r-', linewidth=2, label='Val Loss',
         marker='s', markersize=3, markevery=5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# ---- 子图2：学习率衰减 ----
ax2 = axes1[0, 1]
ax2.plot(df[EPOCH], df[LR], 'g-', linewidth=2, marker='^', markersize=3, markevery=5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Decay', fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# ---- 子图3：主要任务准确率 ----
ax3 = axes1[0, 2]
acc_tasks = [STANCE_ACC, HARMFULNESS_ACC, FAIRNESS_ACC]
acc_labels = ['Stance', 'Harmfulness', 'Fairness']
colors = ['blue', 'red', 'green']
markers = ['o', 's', '^']
for col, label, color, marker in zip(acc_tasks, acc_labels, colors, markers):
    ax3.plot(df[EPOCH], df[col], color=color, linewidth=1.5, label=label,
             marker=marker, markersize=3, markevery=5)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')
ax3.set_title('Main Tasks Accuracy', fontweight='bold')
ax3.set_ylim(0.6, 0.85)
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# ---- 子图4：意图分类 F1 (整体) ----
ax4 = axes1[1, 0]
ax4.plot(df[EPOCH], df[INTENT_F1], 'purple', linewidth=2, label='Intent Macro F1',
         marker='D', markersize=3, markevery=5)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('F1 Score')
ax4.set_title('Intent Classification (Overall F1)', fontweight='bold')
ax4.set_ylim(0.0, 0.75)
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# ---- 子图5：心理意图 F1 (唯一可用的意图类别) ----
ax5 = axes1[1, 1]
ax5.plot(df[EPOCH], df[PSYCH_F1], 'darkred', linewidth=2, label='Psychological F1',
         marker='v', markersize=3, markevery=5)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('F1 Score')
ax5.set_title('Psychological Intent F1', fontweight='bold')
ax5.set_ylim(0.0, 0.55)
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.3)

# ---- 子图6：后10个epoch平均准确率（主要任务） ----
ax6 = axes1[1, 2]
df_late = df.tail(10)
tasks_late = [STANCE_ACC, HARMFULNESS_ACC, FAIRNESS_ACC]
task_names = ['Stance', 'Harmfulness', 'Fairness']
mean_values = [df_late[col].mean() for col in tasks_late]
bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
x_pos = np.arange(len(task_names))
bars = ax6.bar(x_pos, mean_values, color=bar_colors, edgecolor='black', linewidth=0.5)
ax6.set_xlabel('Task')
ax6.set_ylabel('Avg Accuracy (Last 10 epochs)')
ax6.set_title('Late-stage Performance', fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(task_names)
ax6.set_ylim(0.6, 0.85)
ax6.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mean_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_analysis_6plots_nineteen.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== 2. 四子图布局 (2x2) ====================
print("\n生成 4 子图布局...")
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Training Analysis - Nineteen (4 Plots)', fontsize=16, fontweight='bold', y=0.98)

# ---- 子图1：损失 & 学习率（双Y轴） ----
ax1_4 = axes2[0, 0]
ax1_4_twin = ax1_4.twinx()
line1 = ax1_4.plot(df[EPOCH], df[TRAIN_LOSS], 'b-', linewidth=2, label='Train Loss')
line2 = ax1_4.plot(df[EPOCH], df[VAL_LOSS], 'r-', linewidth=2, label='Val Loss')
ax1_4.set_xlabel('Epoch')
ax1_4.set_ylabel('Loss', color='black')
ax1_4.tick_params(axis='y', labelcolor='black')
ax1_4.set_title('Loss & Learning Rate', fontweight='bold')
ax1_4.grid(True, alpha=0.3)
line3 = ax1_4_twin.plot(df[EPOCH], df[LR], 'g--', linewidth=1.5, label='LR')
ax1_4_twin.set_ylabel('Learning Rate', color='green')
ax1_4_twin.tick_params(axis='y', labelcolor='green')
ax1_4_twin.set_yscale('log')
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1_4.legend(lines, labels, loc='upper right')

# ---- 子图2：所有任务准确率（含意图F1作为参考） ----
ax2_4 = axes2[0, 1]
acc_all = [STANCE_ACC, HARMFULNESS_ACC, FAIRNESS_ACC, INTENT_F1]
acc_all_labels = ['Stance Acc', 'Harmfulness Acc', 'Fairness Acc', 'Intent F1']
acc_colors = ['blue', 'red', 'green', 'purple']
acc_markers = ['o', 's', '^', 'D']
for col, label, color, marker in zip(acc_all, acc_all_labels, acc_colors, acc_markers):
    ax2_4.plot(df[EPOCH], df[col], color=color, linewidth=1.5, label=label,
               marker=marker, markersize=3, markevery=5)
ax2_4.set_xlabel('Epoch')
ax2_4.set_ylabel('Score')
ax2_4.set_title('All Tasks Accuracy / Intent F1', fontweight='bold')
ax2_4.set_ylim(0.0, 0.85)
ax2_4.legend(loc='lower right')
ax2_4.grid(True, alpha=0.3)

# ---- 子图3：意图相关 F1 指标 ----
ax3_4 = axes2[1, 0]
f1_metrics = [INTENT_F1, PSYCH_F1]
f1_labels = ['Intent Macro F1', 'Psychological F1']
f1_colors = ['orange', 'darkred']
for col, label, color in zip(f1_metrics, f1_labels, f1_colors):
    ax3_4.plot(df[EPOCH], df[col], color=color, linewidth=1.5, label=label)
ax3_4.set_xlabel('Epoch')
ax3_4.set_ylabel('F1 Score')
ax3_4.set_title('Intent F1 Scores', fontweight='bold')
ax3_4.set_ylim(0.0, 0.75)
ax3_4.legend(loc='lower right')
ax3_4.grid(True, alpha=0.3)

# ---- 子图4：指标相关性热力图 ----
ax4_4 = axes2[1, 1]
corr_metrics = [VAL_LOSS, STANCE_ACC, HARMFULNESS_ACC, FAIRNESS_ACC, INTENT_F1, PSYCH_F1]
corr_labels = ['Val Loss', 'Stance Acc', 'Harm Acc', 'Fair Acc', 'Intent F1', 'Psych F1']
corr_matrix = df[corr_metrics].corr()
im = ax4_4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax4_4.set_xticks(np.arange(len(corr_metrics)))
ax4_4.set_yticks(np.arange(len(corr_metrics)))
ax4_4.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=9)
ax4_4.set_yticklabels(corr_labels, fontsize=9)
ax4_4.set_title('Metrics Correlation', fontweight='bold')
# 添加数值标签
for i in range(len(corr_metrics)):
    for j in range(len(corr_metrics)):
        value = corr_matrix.iloc[i, j]
        color = "white" if abs(value) > 0.6 else "black"
        ax4_4.text(j, i, f'{value:.2f}', ha="center", va="center",
                   color=color, fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_analysis_4plots_nineteen.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== 3. 关键指标汇总图 ====================
print("\n生成关键指标汇总图...")
fig3 = plt.figure(figsize=(15, 10))
fig3.suptitle('Training Key Metrics Summary - Nineteen', fontsize=16, fontweight='bold', y=0.98)
gs = fig3.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.25, wspace=0.3)

# ---- 子图1：损失与过拟合 ----
ax1_sum = fig3.add_subplot(gs[0, 0])
ax1_sum.plot(df[EPOCH], df[TRAIN_LOSS], 'b-', linewidth=2, label='Train Loss')
ax1_sum.plot(df[EPOCH], df[VAL_LOSS], 'r-', linewidth=2, label='Val Loss')
ax1_sum.fill_between(df[EPOCH], df[TRAIN_LOSS], df[VAL_LOSS],
                      color='gray', alpha=0.2, label='Overfitting Gap')
ax1_sum.set_xlabel('Epoch')
ax1_sum.set_ylabel('Loss')
ax1_sum.set_title('Loss & Overfitting', fontweight='bold')
ax1_sum.legend(loc='upper right')
ax1_sum.grid(True, alpha=0.3)

# ---- 子图2：各任务最佳性能 ----
ax2_sum = fig3.add_subplot(gs[0, 1])
best_dict = {
    'Stance': df[STANCE_ACC].max(),
    'Harmfulness': df[HARMFULNESS_ACC].max(),
    'Fairness': df[FAIRNESS_ACC].max(),
    'Intent F1': df[INTENT_F1].max(),
    'Psych F1': df[PSYCH_F1].max()
}
x_pos = np.arange(len(best_dict))
colors_best = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
bars = ax2_sum.bar(x_pos, list(best_dict.values()), color=colors_best, edgecolor='black', linewidth=0.5)
ax2_sum.set_xlabel('Task')
ax2_sum.set_ylabel('Best Score')
ax2_sum.set_title('Best Performance per Task', fontweight='bold')
ax2_sum.set_xticks(x_pos)
ax2_sum.set_xticklabels(best_dict.keys(), rotation=15)
ax2_sum.set_ylim(0, 1.0)
ax2_sum.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, best_dict.values()):
    height = bar.get_height()
    ax2_sum.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# ---- 子图3：不同训练阶段性能（以Stance和Harmfulness为例） ----
ax3_sum = fig3.add_subplot(gs[0, 2])
early_idx = len(df) // 3
mid_idx = 2 * len(df) // 3
stages = ['Early', 'Mid', 'Late']
stance_stage = [
    df[STANCE_ACC].iloc[:early_idx].mean(),
    df[STANCE_ACC].iloc[early_idx:mid_idx].mean(),
    df[STANCE_ACC].iloc[mid_idx:].mean()
]
harm_stage = [
    df[HARMFULNESS_ACC].iloc[:early_idx].mean(),
    df[HARMFULNESS_ACC].iloc[early_idx:mid_idx].mean(),
    df[HARMFULNESS_ACC].iloc[mid_idx:].mean()
]
x = np.arange(len(stages))
width = 0.35
bars1 = ax3_sum.bar(x - width/2, stance_stage, width, label='Stance', color='#1f77b4')
bars2 = ax3_sum.bar(x + width/2, harm_stage, width, label='Harmfulness', color='#ff7f0e')
ax3_sum.set_xlabel('Training Stage')
ax3_sum.set_ylabel('Avg Accuracy')
ax3_sum.set_title('Performance by Training Stage', fontweight='bold')
ax3_sum.set_xticks(x)
ax3_sum.set_xticklabels(stages)
ax3_sum.legend()
ax3_sum.grid(True, alpha=0.3, axis='y')

# ---- 子图4：关键指标汇总表格 ----
ax4_sum = fig3.add_subplot(gs[1, :])
ax4_sum.axis('off')
# 收集汇总信息
best_val_loss_epoch = df.loc[df[VAL_LOSS].idxmin(), EPOCH]
best_train_loss_epoch = df.loc[df[TRAIN_LOSS].idxmin(), EPOCH]
best_stance_epoch = df.loc[df[STANCE_ACC].idxmax(), EPOCH]
best_intent_epoch = df.loc[df[INTENT_F1].idxmax(), EPOCH]
overfit_ratio = df[VAL_LOSS].iloc[-1] / df[VAL_LOSS].iloc[0]
lr_decay_factor = df[LR].iloc[0] / df[LR].iloc[-1]

summary_data = [
    ['Best Val Loss', f"{df[VAL_LOSS].min():.4f}", f"Epoch {int(best_val_loss_epoch)}"],
    ['Best Train Loss', f"{df[TRAIN_LOSS].min():.4f}", f"Epoch {int(best_train_loss_epoch)}"],
    ['Best Stance Acc', f"{df[STANCE_ACC].max():.4f}", f"Epoch {int(best_stance_epoch)}"],
    ['Best Intent F1', f"{df[INTENT_F1].max():.4f}", f"Epoch {int(best_intent_epoch)}"],
    ['Best Psych F1', f"{df[PSYCH_F1].max():.4f}", f"Epoch {int(df.loc[df[PSYCH_F1].idxmax(), EPOCH])}"],
    ['Overfitting', f"{overfit_ratio:.2f}x", "Val loss change"],
    ['LR Decay', f"{lr_decay_factor:.0f}x", f"{df[LR].iloc[0]:.1e} → {df[LR].iloc[-1]:.1e}"]
]
table = ax4_sum.table(cellText=summary_data,
                      colLabels=['Metric', 'Value', 'Note'],
                      colColours=['lightgray'] * 3,
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax4_sum.set_title('Key Metrics Summary', fontweight='bold', fontsize=12, pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_summary_nineteen.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== 终端输出汇总 ====================
print("\n" + "="*70)
print("图表生成成功！")
print("="*70)
print("1. 6子图布局 : training_analysis_6plots_nineteen.png")
print("2. 4子图布局 : training_analysis_4plots_nineteen.png")
print("3. 关键指标汇总: training_summary_nineteen.png")
print("\n关键发现（基于nineteen_training_history.csv）:")
print(f"  - 验证损失最小值: {df[VAL_LOSS].min():.4f} (epoch {int(best_val_loss_epoch)})")
print(f"  - 训练损失最小值: {df[TRAIN_LOSS].min():.4f} (epoch {int(best_train_loss_epoch)})")
print(f"  - 立场准确率最佳: {df[STANCE_ACC].max():.4f} (epoch {int(best_stance_epoch)})")
print(f"  - 意图F1最佳   : {df[INTENT_F1].max():.4f} (epoch {int(best_intent_epoch)})")
print(f"  - 心理意图F1最佳: {df[PSYCH_F1].max():.4f}")
print(f"  - 最终过拟合比 : {overfit_ratio:.2f}x (val loss 终值/初值)")
print("="*70)