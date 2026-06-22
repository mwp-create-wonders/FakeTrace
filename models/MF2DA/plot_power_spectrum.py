import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def calculate_average_power_spectrum(folder_path, num_samples, output_path='average_power_spectrum.png'):
    """
    从指定文件夹中随机抽取指定数量的图片，计算它们的平均功率谱，并绘制保存结果。

    参数:
    folder_path (str): 包含图片的文件夹路径。
    num_samples (int): 要随机抽取的图片数量。
    output_path (str): 保存最终平均功率谱图的路径。
    """
    # 步骤 1: 扫描文件夹，获取所有支持的图片文件路径
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    try:
        all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if os.path.splitext(f)[1].lower() in supported_formats]
    except FileNotFoundError:
        print(f"错误：文件夹不存在，请检查路径：{folder_path}")
        return

    if not all_images:
        print(f"错误：在文件夹 '{folder_path}' 中没有找到支持的图片文件。")
        return

    # 步骤 2: 随机抽样
    if len(all_images) < num_samples:
        print(f"警告：文件夹中图片数量 ({len(all_images)}) 少于要求抽样的数量 ({num_samples})。将使用所有图片。")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, num_samples)

    print(f"将从 {len(all_images)} 张图片中处理 {len(selected_images)} 张随机样本。")

    # 用于累加功率谱的变量
    accumulated_power_spectrum = None
    first_image_shape = None
    processed_count = 0

    # 步骤 3 & 4: 遍历选中的图片，计算并累加功率谱
    for image_path in selected_images:
        # 以灰度模式读取图片
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"警告：无法读取图片 {image_path}，已跳过。")
            continue

        # 如果是第一张有效图片，初始化累加器
        if accumulated_power_spectrum is None:
            first_image_shape = img.shape
            # 初始化一个与图像尺寸相同的全零浮点数数组
            accumulated_power_spectrum = np.zeros(first_image_shape, dtype=np.float64)

        # 确保所有图片尺寸一致，若不一致则跳过
        if img.shape != first_image_shape:
            print(f"警告：图片 {image_path} 的尺寸 {img.shape} 与第一张图片 {first_image_shape} 不符，已跳过。")
            continue

        # 计算傅里叶变换
        f_transform = np.fft.fft2(img)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # 计算功率谱（幅度的平方），并累加
        # 注意：这里我们累加的是真实的功率谱，而不是对数尺度下的值
        power_spectrum = np.abs(f_transform_shifted)**2
        accumulated_power_spectrum += power_spectrum
        processed_count += 1

    if processed_count == 0:
        print("错误：没有任何图片被成功处理。")
        return
    else:
        print("processed_count:",processed_count)
    
    # 计算平均功率谱
    average_power_spectrum = accumulated_power_spectrum / processed_count

    # 使用对数尺度进行可视化，以增强低幅度频率分量的可见性
    # 加 1 是为了避免 log(0)
    log_avg_power_spectrum = np.log10(average_power_spectrum + 1)

# Step 5: Plotting and Saving
    plt.style.use('classic')
    plt.figure(figsize=(8, 8))
    # Using a colormap like 'viridis' or 'inferno' can reveal more details
    plt.imshow(log_avg_power_spectrum, cmap='inferno')
    plt.colorbar(label='Log Power Spectrum Intensity (log10 scale)')
    plt.title(f'Average Power Spectrum based on {processed_count} images')
    plt.axis('off')

    # Save the image
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nAverage power spectrum plot saved successfully to: {output_path}")
    except Exception as e:
        print(f"Failed to save the plot: {e}")

    # Display the plot
    plt.show()


if __name__ == '__main__':
    # --- 使用说明 ---
    # 1. 修改下面的 'your_image_folder' 为您存放图片的文件夹路径。
    # 2. 修改 'num_files_to_sample'为您希望随机抽取的图片数量。
    # 3. 运行此脚本。

    # 要处理的图片文件夹路径
    # image_folder = '/workspace/user-data/models/EXDA/dataset/ExImage/test/CycleGAN'  # <--- 在这里修改您的文件夹路径

    image_folder = '/workspace/user-data/models/EXDA/dataset/Transmisson/Facebook_ExImage/CycleGAN'
    

    # 希望随机抽取的图片数量
    num_files_to_sample = 100  # <--- 在这里修改您希望抽样的数量

    # 输出结果的文件名
    # output_file = 'CycleGAN_power_spectrum.png'
    output_file = 'CycleGAN_transmission_facebook_power_spectrum.png'
    

    # 调用函数
    calculate_average_power_spectrum(image_folder, num_files_to_sample, output_file)