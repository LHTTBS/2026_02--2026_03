# # # # test_basic.py
# # # print("测试开始...")
# # # import matplotlib
# # # print("1. Matplotlib导入成功")
# # # matplotlib.use('Agg')
# # # print("2. 设置后端成功")
# # # import matplotlib.pyplot as plt
# # # print("3. Pyplot导入成功")

# # # # 创建一个非常简单的图
# # # fig = plt.figure()
# # # print("4. 创建图形成功")
# # # plt.plot([1, 2, 3], [1, 4, 9])
# # # print("5. 绘制数据成功")
# # # plt.savefig('test_basic.png')
# # # print("6. 保存文件成功")
# # # plt.close()
# # # print("7. 关闭图形成功")
# # # print("测试完成!")


# # # # test_savefig_detailed.py


# # import os
# # import sys

# # print("测试开始...")

# # # 1. 检查当前工作目录
# # current_dir = os.getcwd()
# # print(f"当前工作目录: {current_dir}")

# # # 2. 检查目录权限
# # test_dir = "."
# # if os.path.exists(test_dir):
# #     print(f"目录 '{test_dir}' 存在")
# #     try:
# #         # 尝试创建一个测试文件来检查权限
# #         test_file = os.path.join(test_dir, "test_permission.txt")
# #         with open(test_file, "w") as f:
# #             f.write("test")
# #         os.remove(test_file)
# #         print(f"目录 '{test_dir}' 有写入权限")
# #     except Exception as e:
# #         print(f"目录 '{test_dir}' 没有写入权限: {e}")
# # else:
# #     print(f"目录 '{test_dir}' 不存在")

# # # 3. 检查磁盘空间
# # import shutil
# # try:
# #     total, used, free = shutil.disk_usage(current_dir)
# #     print(f"磁盘空间 - 总共: {total // (2**30)} GB, 已用: {used // (2**30)} GB, 可用: {free // (2**30)} GB")
# # except:
# #     print("无法获取磁盘空间信息")

# # # 4. 导入matplotlib
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt

# # # 5. 创建简单图形
# # print("创建简单图形...")
# # fig, ax = plt.subplots()
# # ax.plot([1, 2, 3], [1, 4, 2])

# # # 6. 尝试不同路径保存
# # test_paths = [
# #     "test_output.png",  # 当前目录
# #     "./test_output.png",  # 当前目录
# #     os.path.join(current_dir, "test_output.png"),  # 绝对路径
# #     "C:/Users/Administrator/Desktop/test_output.png"  # 桌面
# # ]

# # for path in test_paths:
# #     print(f"\n尝试保存到: {path}")
# #     try:
# #         # 确保目录存在
# #         dir_path = os.path.dirname(path)
# #         if dir_path and not os.path.exists(dir_path):
# #             print(f"  目录不存在，尝试创建: {dir_path}")
# #             os.makedirs(dir_path, exist_ok=True)
        
# #         # 保存文件
# #         plt.savefig(path, dpi=100)
# #         print(f"  ✓ 保存成功")
        
# #         # 检查文件
# #         if os.path.exists(path):
# #             size = os.path.getsize(path)
# #             print(f"  ✓ 文件存在，大小: {size} 字节")
# #         else:
# #             print(f"  ✗ 文件不存在")
            
# #     except Exception as e:
# #         print(f"  ✗ 保存失败: {e}")

# # # 7. 尝试不同的保存格式
# # print("\n尝试不同格式保存...")
# # try:
# #     # 尝试保存为SVG格式（文本格式，可能更简单）
# #     plt.savefig('test_output.svg')
# #     print("  ✓ SVG格式保存成功")
# # except Exception as e:
# #     print(f"  ✗ SVG格式保存失败: {e}")

# # try:
# #     # 尝试保存为PDF格式
# #     plt.savefig('test_output.pdf')
# #     print("  ✓ PDF格式保存成功")
# # except Exception as e:
# #     print(f"  ✗ PDF格式保存失败: {e}")

# # plt.close()
# # print("\n测试完成!")

# # check_pillow.py
# try:
#     import PIL
#     from PIL import Image
#     print(f"✓ Pillow已安装，版本: {PIL.__version__}")
    
#     # 测试PIL是否能正常工作
#     img = Image.new('RGB', (100, 100), color='red')
#     img.save('test_pillow.png')
#     print("✓ Pillow可以正常创建和保存图像")
# except ImportError:
#     print("✗ Pillow未安装")
# except Exception as e:
#     print(f"✗ Pillow存在问题: {e}")

# test_pil_save.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io

print("测试使用PIL保存Matplotlib图像...")

# 创建图形
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('测试图')

# 方法1: 使用canvas将图像保存到内存，然后用PIL保存
try:
    # 将图像渲染到内存
    canvas = fig.canvas
    canvas.draw()
    
    # 获取图像数据
    width, height = canvas.get_width_height()
    image_data = canvas.tostring_rgb()
    
    # 使用PIL创建图像并保存
    pil_image = Image.frombytes('RGB', (width, height), image_data)
    pil_image.save('test_pil_method1.png')
    print("✓ 方法1成功: 使用canvas + PIL保存")
except Exception as e:
    print(f"✗ 方法1失败: {e}")

# 方法2: 使用buffer
try:
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # 使用PIL打开并保存
    img = Image.open(buf)
    img.save('test_pil_method2.png')
    print("✓ 方法2成功: 使用BytesIO + PIL保存")
except Exception as e:
    print(f"✗ 方法2失败: {e}")

plt.close()
print("测试完成")