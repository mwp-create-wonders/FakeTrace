from PIL import Image
import os

# 目标文件夹路径（可根据实际路径修改，绝对路径/相对路径都可以）
folder_path = "autodl-tmp/CocoGlide/mask"
# 目标尺寸：512*512
target_size = (512, 512)

# 遍历文件夹下的所有文件
for file_name in os.listdir(folder_path):
    # 拼接文件完整路径
    file_path = os.path.join(folder_path, file_name)
    # 跳过文件夹（只处理文件）
    if os.path.isdir(file_path):
        continue
    try:
        # 打开图片文件
        with Image.open(file_path) as img:
            # 缩放图片（LANCZOS是最高质量的缩放算法，适合图片保真）
            # Image.Resampling.LANCZOS 适配Pillow9.1+，低版本用 Image.LANCZOS
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            # 覆盖保存原图片（保持原格式）
            resized_img.save(file_path)
            print(f"处理完成：{file_name}")
    except Exception as e:
        # 跳过非图片文件/损坏的图片，打印错误信息不中断程序
        print(f"跳过非图片/损坏文件 {file_name}，错误：{str(e)[:50]}")

print("所有图片批量缩放完成！")