# test_matplotlib.py
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("matplotlib导入成功")
except Exception as e:
    print(f"导入matplotlib时出错: {e}")
    import sys
    print(f"Python路径: {sys.path}")
    # test_matplotlib_backend.py
import matplotlib
print(f"Matplotlib版本: {matplotlib.__version__}")
print(f"当前后端: {matplotlib.get_backend()}")
print(f"可用后端: {matplotlib.rcsetup.all_backends}")

# 尝试设置不同后端
backends_to_try = ['Agg', 'PS', 'PDF', 'SVG', 'Cairo']

for backend in backends_to_try:
    try:
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt
        print(f"✓ 后端 '{backend}' 可用")
        
        # 尝试创建一个简单的图
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.savefig(f'test_{backend}.png')
        plt.close(fig)
        print(f"  成功保存图表为 test_{backend}.png")
        
    except Exception as e:
        print(f"✗ 后端 '{backend}' 不可用: {str(e)[:100]}...")