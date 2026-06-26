import os
import cv2
import numpy as np
import pywt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# -------------------------- 配置 --------------------------
DATASET_PATH = "autodl-tmp/CocoGlide/tampered"
SAVE_ROOT_PATH = "autodl-tmp/cocov2_tensor"
WAVELET = "db1"
TARGET_SIZE = (1024, 1024)
FILL_VALUE = 0

# -------------------------- 向量化核心函数 --------------------------

def binary_encoding(matrix):
    center = matrix[1:-1, 1:-1]

    neighbors = [
        matrix[0:-2, 1:-1],
        matrix[0:-2, 2:],
        matrix[1:-1, 2:],
        matrix[2:, 2:],
        matrix[2:, 1:-1],
        matrix[2:, 0:-2],
        matrix[1:-1, 0:-2],
        matrix[0:-2, 0:-2],
    ]

    bits = [(n <= center).astype(np.uint8) for n in neighbors]
    bits = np.stack(bits, axis=-1)

    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    return np.sum(bits * weights, axis=-1).astype(np.uint8)


def lbp_to_8channels(matrix):
    center = matrix[1:-1, 1:-1]

    neighbors = [
        matrix[0:-2, 1:-1],
        matrix[0:-2, 2:],
        matrix[1:-1, 2:],
        matrix[2:, 2:],
        matrix[2:, 1:-1],
        matrix[2:, 0:-2],
        matrix[1:-1, 0:-2],
        matrix[0:-2, 0:-2],
    ]

    channels = [(n <= center).astype(np.uint8) for n in neighbors]
    return np.stack(channels, axis=-1)


def restore_size(matrix, target_size, fill_value=FILL_VALUE):
    if matrix.ndim == 2:
        h, w = matrix.shape
        out = np.full(target_size, fill_value, dtype=matrix.dtype)
        out[1:-1, 1:-1] = matrix
        return out

    elif matrix.ndim == 3:
        h, w, c = matrix.shape
        out = np.full((*target_size, c), fill_value, dtype=matrix.dtype)
        out[1:-1, 1:-1, :] = matrix
        return out

    else:
        raise ValueError(matrix.shape)


def normalize(x):
    mn, mx = x.min(), x.max()
    if mx - mn == 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

# -------------------------- 单张图处理 --------------------------

def process_single_image(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(img_path)

    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32)

    # ---------- 1级小波 ----------
    LL1, (LH1, HL1, HH1) = pywt.dwt2(img, WAVELET)
    A = np.stack([LL1, LH1, HL1, HH1], axis=-1)

    # ---------- B ----------
    B = []
    for i in range(4):
        enc = binary_encoding(A[..., i])
        B.append(restore_size(enc, (512, 512)))
    B = np.stack(B, axis=-1).astype(np.float32)

    C = A * B

    # ---------- HVD LBP ----------
    hvd = []
    for sub in [LH1, HL1, HH1]:
        lbp = lbp_to_8channels(sub)
        hvd.append(restore_size(lbp, (512, 512)))
    HVD_24 = np.concatenate(hvd, axis=-1).astype(np.float32)

    # ---------- 2级小波 ----------
    D_list = []
    for i in range(4):
        LL2, (LH2, HL2, HH2) = pywt.dwt2(A[..., i], WAVELET)
        D_list.extend([LL2, LH2, HL2, HH2])

    D = np.stack(D_list, axis=-1)

    # ---------- E ----------
    E_list = []
    for i in range(16):
        enc = binary_encoding(D[..., i])
        E_list.append(restore_size(enc, (256, 256)))

    E = np.stack(E_list, axis=-1).astype(np.float32)
    F = D * E

    # ---------- resize ----------
    E_up = cv2.resize(E, (512, 512))
    F_up = cv2.resize(F, (512, 512))

    # ---------- concat ----------
    out = np.concatenate([B, C, E_up, F_up, HVD_24], axis=-1)

    return normalize(out).astype(np.float16)

# -------------------------- worker（多进程） --------------------------

def init_worker(save_root):
    global SAVE_ROOT
    SAVE_ROOT = save_root


def worker(img_path):
    try:
        tensor = process_single_image(img_path)

        name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(SAVE_ROOT, name + ".npz")

        np.savez_compressed(save_path, tensor=tensor)

        return 1
    except Exception as e:
        print("error:", img_path, e)
        return 0

# -------------------------- 主函数（并行） --------------------------

def process_dataset():

    os.makedirs(SAVE_ROOT_PATH, exist_ok=True)

    img_paths = [
        os.path.join(DATASET_PATH, f)
        for f in os.listdir(DATASET_PATH)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif"))
    ]

    workers = os.cpu_count() or 1
    print(f"Using {workers} CPU cores")

    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(SAVE_ROOT_PATH,)
    ) as pool:

        results = list(tqdm(
            pool.imap(worker, img_paths),
            total=len(img_paths)
        ))

    print("done:", sum(results), "/", len(results))


# -------------------------- main --------------------------

if __name__ == "__main__":
    process_dataset()