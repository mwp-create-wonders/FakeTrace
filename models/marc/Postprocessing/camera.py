import os
import json
import random
import argparse
from io import BytesIO
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_image_files(folder: str) -> Dict[str, str]:
    """
    返回: {filename: absolute_path}
    这里要求三个文件夹中同名文件一一对应。
    """
    files = {}
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_EXTS:
            files[name] = os.path.join(folder, name)
    return files


def pil_open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(img: Image.Image, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".png":
        img.save(save_path, format="PNG")
    elif ext in [".jpg", ".jpeg"]:
        img.save(save_path, format="JPEG", quality=95)
    elif ext == ".webp":
        img.save(save_path, format="WEBP", quality=95)
    elif ext == ".bmp":
        img.save(save_path, format="BMP")
    else:
        # 默认保存为 PNG
        save_path = os.path.splitext(save_path)[0] + ".png"
        img.save(save_path, format="PNG")


# =========================
# 单一处理操作
# =========================

def op_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def op_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def op_saturation(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)


def op_color_temperature(img: Image.Image, r_scale: float, g_scale: float, b_scale: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr[..., 0] *= r_scale
    arr[..., 1] *= g_scale
    arr[..., 2] *= b_scale
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def op_gamma(img: Image.Image, gamma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def op_sharpen(img: Image.Image, alpha: float, blur_radius: float) -> Image.Image:
    # Unsharp masking
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr_img = np.asarray(img).astype(np.float32)
    arr_blur = np.asarray(blurred).astype(np.float32)
    out = arr_img + alpha * (arr_img - arr_blur)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def op_gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def op_jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    # 通过内存编码-解码模拟 JPEG 压缩
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def op_resize_recover(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    down = img.resize((new_w, new_h), Image.BICUBIC)
    up = down.resize((w, h), Image.BICUBIC)
    return up


# =========================
# 随机选择一种操作，并生成参数
# =========================

def sample_one_operation(rng: random.Random) -> Tuple[str, Dict]:
    """
    每张图只随机选择一个操作。
    返回:
        op_name: str
        params: dict
    """
    op_name = rng.choice([
        "brightness",
        "contrast",
        "saturation",
        "color_temperature",
        "gamma",
        "sharpen",
        "gaussian_blur",
        "jpeg_compress",
        "resize_recover",
    ])

    if op_name == "brightness":
        params = {"factor": round(rng.uniform(0.90, 1.10), 4)}

    elif op_name == "contrast":
        params = {"factor": round(rng.uniform(0.90, 1.10), 4)}

    elif op_name == "saturation":
        params = {"factor": round(rng.uniform(0.92, 1.08), 4)}

    elif op_name == "color_temperature":
        warm = rng.choice([True, False])
        if warm:
            params = {
                "r_scale": round(rng.uniform(1.02, 1.08), 4),
                "g_scale": round(rng.uniform(0.98, 1.02), 4),
                "b_scale": round(rng.uniform(0.92, 0.98), 4),
                "mode": "warm"
            }
        else:
            params = {
                "r_scale": round(rng.uniform(0.92, 0.98), 4),
                "g_scale": round(rng.uniform(0.98, 1.02), 4),
                "b_scale": round(rng.uniform(1.02, 1.08), 4),
                "mode": "cool"
            }

    elif op_name == "gamma":
        params = {"gamma": round(rng.uniform(0.90, 1.10), 4)}

    elif op_name == "sharpen":
        params = {
            "alpha": round(rng.uniform(0.3, 0.8), 4),
            "blur_radius": round(rng.uniform(0.6, 1.2), 4)
        }

    elif op_name == "gaussian_blur":
        params = {"radius": round(rng.uniform(0.4, 1.2), 4)}

    elif op_name == "jpeg_compress":
        params = {"quality": rng.randint(75, 95)}

    elif op_name == "resize_recover":
        params = {"scale": round(rng.uniform(0.70, 0.95), 4)}

    else:
        raise ValueError(f"Unknown operation: {op_name}")

    return op_name, params


def apply_operation(img: Image.Image, op_name: str, params: Dict) -> Image.Image:
    if op_name == "brightness":
        return op_brightness(img, **params)
    elif op_name == "contrast":
        return op_contrast(img, **params)
    elif op_name == "saturation":
        return op_saturation(img, **params)
    elif op_name == "color_temperature":
        # mode 只是记录，不参与函数参数
        p = {k: v for k, v in params.items() if k in ["r_scale", "g_scale", "b_scale"]}
        return op_color_temperature(img, **p)
    elif op_name == "gamma":
        return op_gamma(img, **params)
    elif op_name == "sharpen":
        return op_sharpen(img, **params)
    elif op_name == "gaussian_blur":
        return op_gaussian_blur(img, **params)
    elif op_name == "jpeg_compress":
        return op_jpeg_compress(img, **params)
    elif op_name == "resize_recover":
        return op_resize_recover(img, **params)
    else:
        raise ValueError(f"Unsupported operation: {op_name}")


def process_triplets(
    real_dir: str,
    vae_dir: str,
    ar_dir: str,
    output_root: str,
    real_subdir: str,
    vae_subdir: str,
    ar_subdir: str,
    json_path: str,
    seed: int = 2026,
):
    rng = random.Random(seed)

    real_files = list_image_files(real_dir)
    vae_files = list_image_files(vae_dir)
    ar_files = list_image_files(ar_dir)

    common_names = sorted(set(real_files.keys()) & set(vae_files.keys()) & set(ar_files.keys()))
    if len(common_names) == 0:
        raise RuntimeError("三个文件夹中没有找到同名图像，无法进行对应处理。")

    print(f"Found {len(common_names)} matched image triplets.")

    out_real_dir = os.path.join(output_root, real_subdir)
    out_vae_dir = os.path.join(output_root, vae_subdir)
    out_ar_dir = os.path.join(output_root, ar_subdir)

    ensure_dir(out_real_dir)
    ensure_dir(out_vae_dir)
    ensure_dir(out_ar_dir)

    records: List[Dict] = []

    for idx, filename in enumerate(common_names, 1):
        real_path = real_files[filename]
        vae_path = vae_files[filename]
        ar_path = ar_files[filename]

        try:
            img_real = pil_open_rgb(real_path)
            img_vae = pil_open_rgb(vae_path)
            img_ar = pil_open_rgb(ar_path)

            # 关键：同名的三张图采用完全相同的操作和参数
            op_name, params = sample_one_operation(rng)

            out_real = apply_operation(img_real, op_name, params)
            out_vae = apply_operation(img_vae, op_name, params)
            out_ar = apply_operation(img_ar, op_name, params)

            save_image(out_real, os.path.join(out_real_dir, filename))
            save_image(out_vae, os.path.join(out_vae_dir, filename))
            save_image(out_ar, os.path.join(out_ar_dir, filename))

            records.append({
                "filename": filename,
                "operation": op_name,
                "params": params
            })

            if idx % 200 == 0 or idx == len(common_names):
                print(f"[{idx}/{len(common_names)}] processed")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            records.append({
                "filename": filename,
                "operation": "ERROR",
                "params": {},
                "error": str(e)
            })

    ensure_dir(os.path.dirname(json_path) if os.path.dirname(json_path) else ".")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "num_triplets": len(common_names),
            "real_dir": real_dir,
            "vae_dir": vae_dir,
            "ar_dir": ar_dir,
            "output_root": output_root,
            "records": records
        }, f, ensure_ascii=False, indent=2)

    print(f"Done. JSON saved to: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="对 REAL / VAE / AR 三个文件夹中同名图像进行同步单操作后处理，并保存 JSON 记录。"
    )
    parser.add_argument("--real_dir", type=str, required=True, help="原始 REAL 图像文件夹")
    parser.add_argument("--vae_dir", type=str, required=True, help="VAE 重建图像文件夹")
    parser.add_argument("--ar_dir", type=str, required=True, help="AR 重建图像文件夹")
    parser.add_argument("--output_root", type=str, required=True, help="输出大文件夹")
    parser.add_argument("--real_subdir", type=str, default="real_processed", help="输出大文件夹下 REAL 子文件夹名")
    parser.add_argument("--vae_subdir", type=str, default="vae_processed", help="输出大文件夹下 VAE 子文件夹名")
    parser.add_argument("--ar_subdir", type=str, default="ar_processed", help="输出大文件夹下 AR 子文件夹名")
    parser.add_argument("--json_path", type=str, required=True, help="保存文件名与操作记录的 JSON 路径")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，保证可复现")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_triplets(
        real_dir=args.real_dir,
        vae_dir=args.vae_dir,
        ar_dir=args.ar_dir,
        output_root=args.output_root,
        real_subdir=args.real_subdir,
        vae_subdir=args.vae_subdir,
        ar_subdir=args.ar_subdir,
        json_path=args.json_path,
        seed=args.seed,
    )