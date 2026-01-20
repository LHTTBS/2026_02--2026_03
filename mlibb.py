# test_plot_simple.py
import os
import matplotlib
matplotlib.use('Agg')  # 强制使用Agg后端
import matplotlib.pyplot as plt
import numpy as np

print("开始测试Matplotlib...")

# 创建简单的数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 4, 2, 5, 3, 6, 4, 7, 5, 8]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'b-', linewidth=2, label='测试数据')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_title('Matplotlib简单测试')
ax.legend()
ax.grid(True, alpha=0.3)

# 保存图形
output_path = 'test_output.png'
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close(fig)  # 关闭图形释放内存

print(f"图表已保存到: {output_path}")
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path)
    print(f"✓ 文件大小: {file_size:,} 字节")
else:
    print("✗ 错误: 文件未创建")