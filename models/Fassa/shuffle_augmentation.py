import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# ===================== 配置路径 =====================
TAMPERED_DIR = "autodl-tmp/data/tampered"
MASKS_DIR = "autodl-tmp/data/masks"
OUT_TAMPERED = "autodl-tmp/data/tampered_shuffle"
OUT_MASKS = "autodl-tmp/data/masks_shuffle"

# 自动创建输出文件夹
os.makedirs(OUT_TAMPERED, exist_ok=True)
os.makedirs(OUT_MASKS, exist_ok=True)

# ===================== 核心函数：切分 + 强制打乱 =====================
def shuffle_4_patches(img: Image.Image) -> Image.Image:
    """
    512x512 → 4个256x256 patch
    随机重排，但**绝对不会使用原始顺序 [0,1,2,3]**
    """
    w, h = img.size
    assert w == 512 and h == 512, "必须是512x512尺寸"
    patch_size = 256

    # 切割成4个patch（固定顺序：左上、右上、左下、右下）
    patches = [
        img.crop((0, 0, patch_size, patch_size)),                # 0 左上
        img.crop((patch_size, 0, w, patch_size)),               # 1 右上
        img.crop((0, patch_size, patch_size, h)),               # 2 左下
        img.crop((patch_size, patch_size, w, h))                 # 3 右下
    ]

    # ============== 关键修改：保证打乱后≠原始顺序 ==============
    original_order = [0, 1, 2, 3]
    order = original_order.copy()
    
    # 循环打乱，直到得到一个和原始顺序不同的结果
    while order == original_order:
        random.shuffle(order)

    # 按随机顺序重排
    shuffled_patches = [patches[i] for i in order]

    # 重新拼接成512x512
    new_img = Image.new(img.mode, (w, h))
    new_img.paste(shuffled_patches[0], (0, 0))
    new_img.paste(shuffled_patches[1], (patch_size, 0))
    new_img.paste(shuffled_patches[2], (0, patch_size))
    new_img.paste(shuffled_patches[3], (patch_size, patch_size))

    return new_img

# ===================== 批量处理配对图片 =====================
img_files = sorted([f for f in os.listdir(TAMPERED_DIR) if f.endswith(".png")])

for img_file in tqdm(img_files, desc="处理图片中"):
    base_name = os.path.splitext(img_file)[0]
    mask_file = f"{base_name}_mask.png"

    img_path = os.path.join(TAMPERED_DIR, img_file)
    mask_path = os.path.join(MASKS_DIR, mask_file)
    
    if not os.path.exists(mask_path):
        print(f"跳过：{img_file} 无对应掩码")
        continue

    # 打开图片
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # 同一对图片+掩码使用相同随机种子（保证顺序完全一致）
    random.seed(base_name)

    # 处理（都使用相同的打乱顺序）
    img_shuffled = shuffle_4_patches(img)
    mask_shuffled = shuffle_4_patches(mask)

    # 保存
    img_shuffled.save(os.path.join(OUT_TAMPERED, img_file))
    mask_shuffled.save(os.path.join(OUT_MASKS, mask_file))

print("\n全部处理完成！")
print(f"打乱后原图：{OUT_TAMPERED}")
print(f"打乱后掩码：{OUT_MASKS}")