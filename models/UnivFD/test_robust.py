import argparse
import csv
import os
import random
from collections import OrderedDict
from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile, ImageFilter
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader

from models import get_model
from data.datasets import collect_images_from_dirs

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 0

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}
STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_dir_list(s):
    if s is None:
        return None
    if isinstance(s, list):
        return s
    s = s.strip()
    if s == "":
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def parse_int_list(s):
    if s is None:
        return None
    if isinstance(s, list):
        return [int(x) for x in s]
    s = s.strip()
    if s == "":
        return None
    return [int(item.strip()) for item in s.split(",") if item.strip()]


def parse_float_list(s):
    if s is None:
        return None
    if isinstance(s, list):
        return [float(x) for x in s]
    s = str(s).strip()
    if s == "":
        return None
    return [float(item.strip()) for item in s.split(",") if item.strip()]


def find_best_threshold(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    best_acc = 0.0
    best_thres = 0.5
    for thres in np.unique(y_pred):
        pred_bin = (y_pred >= thres).astype(np.int32)
        acc = (pred_bin == y_true).mean()
        if acc >= best_acc:
            best_acc = acc
            best_thres = float(thres)
    return best_thres


def calculate_acc(y_true, y_pred, thres):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_bin = (y_pred >= thres).astype(np.int32)

    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)

    r_acc = accuracy_score(y_true[real_mask], y_bin[real_mask]) if real_mask.sum() > 0 else 0.0
    f_acc = accuracy_score(y_true[fake_mask], y_bin[fake_mask]) if fake_mask.sum() > 0 else 0.0
    acc = accuracy_score(y_true, y_bin)
    return r_acc, f_acc, acc


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=int(quality))
    out.seek(0)
    img = Image.open(out).convert("RGB")
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def apply_gaussian_blur_scipy(img, sigma):
    """Original gaussian blur implementation kept for compatibility."""
    img = np.array(img).copy()
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
    return Image.fromarray(img)


def apply_gaussian_blur_pil(img, radius):
    """PIL GaussianBlur. The radius is easy to interpret in image robustness tests."""
    return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def apply_resize_recover(img, scale, interpolation="bicubic"):
    """
    Downsample the image by `scale` and then recover it to the original size.
    This simulates resolution degradation while keeping the final input size unchanged.
    """
    scale = float(scale)
    if scale <= 0 or scale > 1:
        raise ValueError(f"resize_scale should be in (0, 1], got {scale}")

    w, h = img.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    resample = interp_map.get(interpolation.lower(), Image.BICUBIC)

    img = img.resize((new_w, new_h), resample=resample)
    img = img.resize((w, h), resample=resample)
    return img


def build_robust_tag(opt):
    if opt.robust_mode == "clean":
        return "clean"
    if opt.robust_mode == "blur":
        return f"blur_r{opt.blur_radius}"
    if opt.robust_mode == "resize":
        return f"resize_s{opt.resize_scale}"
    if opt.robust_mode == "jpeg":
        return f"jpeg_q{opt.jpeg_quality}"
    if opt.robust_mode == "blur_resize":
        return f"blur_r{opt.blur_radius}_resize_s{opt.resize_scale}"
    if opt.robust_mode == "resize_blur":
        return f"resize_s{opt.resize_scale}_blur_r{opt.blur_radius}"
    return opt.robust_mode


class EvalRealFakeDataset(Dataset):
    def __init__(
        self,
        real_dirs,
        fake_dirs,
        arch,
        real_num_per_dir=None,
        fake_num_per_dir=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        image_size=224,
        robust_mode="clean",
        blur_radius=None,
        resize_scale=None,
        resize_interpolation="bicubic",
    ):
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.robust_mode = robust_mode
        self.blur_radius = blur_radius
        self.resize_scale = resize_scale
        self.resize_interpolation = resize_interpolation

        if self.robust_mode not in ["clean", "jpeg", "blur", "resize", "blur_resize", "resize_blur"]:
            raise ValueError(
                "robust_mode must be one of: clean, jpeg, blur, resize, blur_resize, resize_blur"
            )

        # Backward compatibility: if users still pass --gaussian_sigma without --robust_mode,
        # the script will behave as the original blur test.
        if self.robust_mode == "clean" and self.gaussian_sigma is not None:
            self.robust_mode = "blur"
            self.blur_radius = self.gaussian_sigma

        if self.robust_mode == "jpeg" and self.jpeg_quality is None:
            raise ValueError("--robust_mode jpeg requires --jpeg_quality")
        if self.robust_mode in ["blur", "blur_resize", "resize_blur"] and self.blur_radius is None:
            raise ValueError(f"--robust_mode {self.robust_mode} requires --blur_radius")
        if self.robust_mode in ["resize", "blur_resize", "resize_blur"] and self.resize_scale is None:
            raise ValueError(f"--robust_mode {self.robust_mode} requires --resize_scale")

        real_list = collect_images_from_dirs(real_dirs, real_num_per_dir)
        fake_list = collect_images_from_dirs(fake_dirs, fake_num_per_dir)

        if len(real_list) == 0:
            raise ValueError("real_list is empty.")
        if len(fake_list) == 0:
            raise ValueError("fake_list is empty.")

        self.total_list = real_list + fake_list
        self.labels_dict = {p: 0 for p in real_list}
        self.labels_dict.update({p: 1 for p in fake_list})

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

        print(f"[Dataset] real: {len(real_list)}, fake: {len(fake_list)}, total: {len(self.total_list)}")
        print(
            f"[Robustness] mode={self.robust_mode}, "
            f"blur_radius={self.blur_radius}, resize_scale={self.resize_scale}, "
            f"jpeg_quality={self.jpeg_quality}, resize_interpolation={self.resize_interpolation}"
        )

    def apply_robustness(self, img):
        if self.robust_mode == "clean":
            return img

        if self.robust_mode == "jpeg":
            return png2jpg(img, self.jpeg_quality)

        if self.robust_mode == "blur":
            return apply_gaussian_blur_pil(img, self.blur_radius)

        if self.robust_mode == "resize":
            return apply_resize_recover(img, self.resize_scale, self.resize_interpolation)

        if self.robust_mode == "blur_resize":
            img = apply_gaussian_blur_pil(img, self.blur_radius)
            img = apply_resize_recover(img, self.resize_scale, self.resize_interpolation)
            return img

        if self.robust_mode == "resize_blur":
            img = apply_resize_recover(img, self.resize_scale, self.resize_interpolation)
            img = apply_gaussian_blur_pil(img, self.blur_radius)
            return img

        raise ValueError(f"Unsupported robust_mode: {self.robust_mode}")

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        for _ in range(10):
            img_path = self.total_list[idx]
            label = self.labels_dict[img_path]
            try:
                img = Image.open(img_path).convert("RGB")
                img = self.apply_robustness(img)
                img = self.transform(img)
                return img, label
            except Exception as e:
                print(f"[Warning] Failed to read image: {img_path}, error: {e}")
                idx = np.random.randint(0, len(self.total_list))

        raise RuntimeError("Too many failed image reads.")


@torch.no_grad()
def validate(model, loader, device=None, find_thres=False):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    y_true, y_pred = [], []

    print("Length of dataloader: %d" % len(loader))

    for img, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(img)

        if logits.ndim == 1:
            prob = torch.sigmoid(logits)
        elif logits.ndim == 2 and logits.shape[1] == 1:
            prob = torch.sigmoid(logits[:, 0])
        elif logits.ndim == 2 and logits.shape[1] == 2:
            prob = torch.softmax(logits, dim=1)[:, 1]
        else:
            raise ValueError(f"Unsupported model output shape: {logits.shape}")

        y_pred.extend(prob.detach().cpu().numpy().tolist())
        y_true.extend(label.detach().cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auroc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)

    if not find_thres:
        return auroc, ap, r_acc0, f_acc0, acc0

    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return auroc, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


def load_checkpoint_to_model(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu')

    if isinstance(state_dict, dict) and 'model' in state_dict:
        inner = state_dict['model']

        filtered_state_dict = OrderedDict()
        if isinstance(inner, dict) and 'fc.weight' in inner:
            filtered_state_dict['weight'] = inner['fc.weight']
        if isinstance(inner, dict) and 'fc.bias' in inner:
            filtered_state_dict['bias'] = inner['fc.bias']

        if len(filtered_state_dict) > 0 and hasattr(model, 'fc'):
            model.fc.load_state_dict(filtered_state_dict, strict=False)
            print("Loaded fc layer from checkpoint.")
            return model

        missing, unexpected = model.load_state_dict(inner, strict=False)
        print("Loaded checkpoint from state_dict['model'].")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        return model

    if isinstance(state_dict, dict):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint from raw state_dict.")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        return model

    raise ValueError("Unsupported checkpoint format.")


def build_dataset_configs(opt):
    dataset_configs = []

    if opt.parent_folder is not None and len(opt.subfolders) > 0:
        for subfolder in opt.subfolders:
            fake_num_per_dir = opt.fake_num_per_dir

            # parent_folder + subfolders 模式下，每个 subfolder 会单独构成一个 fake_dirs
            # 因此如果只传了一个 fake_num_per_dir，例如 1000，则对每个 subfolder 都使用 1000
            if fake_num_per_dir is not None:
                if len(fake_num_per_dir) == 1:
                    fake_num_per_dir = [fake_num_per_dir[0]]
                else:
                    raise ValueError(
                        "在 --parent_folder + --subfolders 模式下，"
                        "--fake_num_per_dir 建议只传一个数，例如 --fake_num_per_dir 1000"
                    )

            dataset_configs.append({
                'key': subfolder,
                'fake_dirs': [os.path.join(opt.parent_folder, subfolder)],
                'fake_num_per_dir': fake_num_per_dir
            })

    elif opt.fake_dirs is not None and len(opt.fake_dirs) > 0:
        dataset_configs.append({
            'key': opt.key if opt.key is not None else 'custom_test',
            'fake_dirs': opt.fake_dirs,
            'fake_num_per_dir': opt.fake_num_per_dir
        })
    else:
        raise ValueError("必须提供 --fake_dirs，或提供 --parent_folder 与 --subfolders")

    return dataset_configs


def build_robust_jobs(opt):
    """
    Build one or multiple robustness jobs.
    If --run_resize_blur_suite is enabled, this creates a sequence of resize/blur tests.
    Otherwise, only the current argument setting is evaluated once.
    """
    if not opt.run_resize_blur_suite:
        return [{
            "robust_mode": opt.robust_mode,
            "blur_radius": opt.blur_radius,
            "resize_scale": opt.resize_scale,
            "jpeg_quality": opt.jpeg_quality,
            "tag": build_robust_tag(opt),
        }]

    jobs = []
    if opt.include_clean_in_suite:
        jobs.append({
            "robust_mode": "clean",
            "blur_radius": None,
            "resize_scale": None,
            "jpeg_quality": None,
            "tag": "clean",
        })

    blur_radii = parse_float_list(opt.blur_radii)
    resize_scales = parse_float_list(opt.resize_scales)

    if blur_radii:
        for r in blur_radii:
            jobs.append({
                "robust_mode": "blur",
                "blur_radius": r,
                "resize_scale": None,
                "jpeg_quality": None,
                "tag": f"blur_r{r}",
            })

    if resize_scales:
        for s in resize_scales:
            jobs.append({
                "robust_mode": "resize",
                "blur_radius": None,
                "resize_scale": s,
                "jpeg_quality": None,
                "tag": f"resize_s{s}",
            })

    return jobs


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--real_dirs', type=str, required=True)
    parser.add_argument('--real_num_per_dir', type=str, default=None)

    parser.add_argument('--fake_dirs', type=str, default=None)
    parser.add_argument('--fake_num_per_dir', type=str, default=1000)

    parser.add_argument('--parent_folder', type=str, default=None)
    parser.add_argument('--subfolders', type=str, nargs='+', default=[])

    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--result_folder', type=str, default='result')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    # Original parameters kept for compatibility.
    parser.add_argument('--jpeg_quality', type=int, default=None)
    parser.add_argument('--gaussian_sigma', type=float, default=None,
                        help='Backward-compatible blur parameter. Prefer --robust_mode blur --blur_radius.')

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--key', type=str, default=None)

    # New robustness parameters.
    parser.add_argument('--robust_mode', type=str, default='clean',
                        choices=['clean', 'jpeg', 'blur', 'resize', 'blur_resize', 'resize_blur'],
                        help='Test-time robustness perturbation mode.')
    parser.add_argument('--blur_radius', type=float, default=None,
                        help='Gaussian blur radius for --robust_mode blur / blur_resize / resize_blur.')
    parser.add_argument('--resize_scale', type=float, default=None,
                        help='Downsample ratio for resize-recover robustness test, e.g., 0.9, 0.75, 0.5, 0.25.')
    parser.add_argument('--resize_interpolation', type=str, default='bicubic',
                        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'])

    # Optional built-in suite for one-shot resize and blur experiments.
    parser.add_argument('--run_resize_blur_suite', action='store_true',
                        help='Run clean + multiple blur radii + multiple resize scales in one command.')
    parser.add_argument('--include_clean_in_suite', action='store_true')
    parser.add_argument('--blur_radii', type=str, default='0.5,1.0,2.0')
    parser.add_argument('--resize_scales', type=str, default='0.9,0.75,0.5,0.25')

    parser.add_argument('--append_csv', action='store_true',
                    help='Append results to existing results.csv instead of overwriting it.')

    opt = parser.parse_args()

    set_seed()

    opt.real_dirs = parse_dir_list(opt.real_dirs)
    opt.fake_dirs = parse_dir_list(opt.fake_dirs)
    opt.real_num_per_dir = parse_int_list(opt.real_num_per_dir)
    opt.fake_num_per_dir = parse_int_list(opt.fake_num_per_dir)

    if opt.real_dirs is None or len(opt.real_dirs) == 0:
        raise ValueError("real_dirs 不能为空")

    if opt.real_num_per_dir is not None and len(opt.real_dirs) != len(opt.real_num_per_dir):
        raise ValueError("real_dirs 和 real_num_per_dir 长度不一致")

    if opt.fake_dirs is not None and opt.fake_num_per_dir is not None:
        if len(opt.fake_dirs) != len(opt.fake_num_per_dir):
            raise ValueError("fake_dirs 和 fake_num_per_dir 长度不一致")

    os.makedirs(opt.result_folder, exist_ok=True)

    model = get_model(opt.arch)
    model = load_checkpoint_to_model(model, opt.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataset_configs = build_dataset_configs(opt)
    robust_jobs = build_robust_jobs(opt)

    csv_path = os.path.join(opt.result_folder, 'results.csv')
    
    csv_header = [
        'Dataset', 'Robust_Mode', 'Robust_Tag',
        'Blur_Radius', 'Resize_Scale', 'JPEG_Quality', 'Resize_Interpolation',
        'AUROC', 'AP',
        'R_Acc_0.5', 'F_Acc_0.5', 'Acc_0.5',
        'Best_Threshold', 'R_Acc_Best', 'F_Acc_Best', 'Acc_Best'
    ]
    
    need_write_header = True
    
    if opt.append_csv and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        need_write_header = False
    
    if (not opt.append_csv) or need_write_header:
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header)

    all_results = []

    for job in robust_jobs:
        print("\n" + "=" * 80)
        print(f"Running robustness job: {job['tag']}")
        print("=" * 80)

        for cfg in dataset_configs:
            set_seed()

            dataset = EvalRealFakeDataset(
                real_dirs=opt.real_dirs,
                fake_dirs=cfg['fake_dirs'],
                arch=opt.arch,
                real_num_per_dir=opt.real_num_per_dir,
                fake_num_per_dir=cfg['fake_num_per_dir'],
                jpeg_quality=job['jpeg_quality'],
                gaussian_sigma=opt.gaussian_sigma,
                image_size=opt.image_size,
                robust_mode=job['robust_mode'],
                blur_radius=job['blur_radius'],
                resize_scale=job['resize_scale'],
                resize_interpolation=opt.resize_interpolation,
            )

            loader = DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False
            )

            auroc, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(
                model, loader, device=device, find_thres=True
            )

            result = [
                cfg['key'],
                job['robust_mode'],
                job['tag'],
                '' if job['blur_radius'] is None else job['blur_radius'],
                '' if job['resize_scale'] is None else job['resize_scale'],
                '' if job['jpeg_quality'] is None else job['jpeg_quality'],
                opt.resize_interpolation,
                round(auroc * 100, 6),
                round(ap * 100, 6),
                round(r_acc0 * 100, 6),
                round(f_acc0 * 100, 6),
                round(acc0 * 100, 6),
                round(best_thres, 6),
                round(r_acc1 * 100, 6),
                round(f_acc1 * 100, 6),
                round(acc1 * 100, 6),
            ]
            all_results.append(result)

            print(
                f"完成: {cfg['key']} | {job['tag']}, "
                f"AUROC: {auroc * 100:.2f}%, "
                f"AP: {ap * 100:.2f}%, "
                f"Acc@0.5: {acc0 * 100:.2f}%, "
                f"Acc@Best: {acc1 * 100:.2f}%"
            )

    with open(csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(all_results)

    print(f"\n所有测试完成，结果已保存到: {csv_path}")
    print(f"共测试了 {len(all_results)} 个结果项")


if __name__ == '__main__':
    main()
