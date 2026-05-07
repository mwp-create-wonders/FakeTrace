import numpy as np

# 1. 定义你的 .npz 文件路径
# 请将 'path/to/your/file.jpg.npz' 替换为你的实际文件路径
file_path = 'output_folder/Sp_D_CND_A_pla0005_pla0023_0281.jpg.npz' 

try:
    # 2. 使用 np.load() 加载文件
    with np.load(file_path) as data:
        
        # 3. 打印文件包含的所有键
        all_keys = data.files
        print(f"文件 '{file_path}' 中包含的数组名称 (keys): {all_keys}\n")
        
        # 4. 遍历所有键，并加载对应的数组
        for key in all_keys:
            # 加载当前键对应的数组
            array_data = data[key]
            
            # 打印分隔符，让输出更清晰
            print("-" * 40)
            print(f"读取数组: '{key}'")
            print("-" * 40)
            
            # 打印数组的详细信息
            print(f"  - 形状 (shape): {array_data.shape}")
            print(f"  - 数据类型 (dtype): {array_data.dtype}")
            print(f"  - 数组内容 (value):")
            
            # 为了防止打印过大的数组导致刷屏，我们做一个简单的判断
            # 如果是多维数组（比如图像掩码），就只打印一些基本统计信息
            # 如果是标量或者小数组，就直接打印内容
            if array_data.ndim > 1: # ndim 代表数组的维度数量
                print(f"    (多维数组，不完全显示)")
                print(f"    - 最小值 (min): {np.min(array_data)}")
                print(f"    - 最大值 (max): {np.max(array_data)}")
                print(f"    - 平均值 (mean): {np.mean(array_data)}")
            else:
                # 直接打印数组内容
                print(f"    {array_data}")
            
            # 在每个数组信息后加一个换行，方便阅读
            print("\n")

except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。")
except Exception as e:
    print(f"读取文件时发生错误: {e}")