import random
import shutil
from pathlib import Path
from typing import Dict, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def random_extract_images(
    src_root: str,
    dst_root: str,
    num_images: int,
    seed: int = None,
    strict: bool = False,
) -> Dict[str, List[str]]:
    """
    Randomly extract images from each subfolder under src_root and copy them to
    subfolders with the same names under dst_root.

    Args:
        src_root: Source parent folder that contains multiple subfolders.
        dst_root: Destination parent folder.
        num_images: Number of images to extract from each subfolder.
        seed: Optional random seed for reproducible sampling.
        strict: If True, raise an error when a subfolder has fewer than
            num_images images. If False, copy all available images instead.

    Returns:
        A mapping from subfolder name to the list of copied image paths.
    """
    if num_images <= 0:
        raise ValueError("num_images must be greater than 0.")

    src_root_path = Path(src_root)
    dst_root_path = Path(dst_root)

    if not src_root_path.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src_root_path}")
    if not src_root_path.is_dir():
        raise NotADirectoryError(f"Source path is not a folder: {src_root_path}")

    rng = random.Random(seed)
    results: Dict[str, List[str]] = {}

    subfolders = sorted([path for path in src_root_path.iterdir() if path.is_dir()])
    if not subfolders:
        raise ValueError(f"No subfolders found under: {src_root_path}")

    for subfolder in subfolders:
        image_files = sorted(
            [
                path
                for path in subfolder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

        if not image_files:
            print(f"Skip empty image folder: {subfolder}")
            results[subfolder.name] = []
            continue

        if strict and len(image_files) < num_images:
            raise ValueError(
                f"Folder '{subfolder}' only contains {len(image_files)} images, "
                f"which is fewer than the requested {num_images}."
            )

        sample_count = min(num_images, len(image_files))
        selected_files = rng.sample(image_files, sample_count)

        dst_subfolder = dst_root_path / subfolder.name
        dst_subfolder.mkdir(parents=True, exist_ok=True)

        copied_files: List[str] = []
        for image_path in selected_files:
            dst_path = dst_subfolder / image_path.name
            shutil.copy2(image_path, dst_path)
            copied_files.append(str(dst_path))

        results[subfolder.name] = copied_files
        print(
            f"{subfolder.name}: copied {len(copied_files)} / "
            f"{len(image_files)} images to {dst_subfolder}"
        )

    return results


if __name__ == "__main__":
    src_root = r"D:/ARForensics"
    dst_root = r"D:/ARForensics-tiny"
    num_images = 500

    random_extract_images(
        src_root=src_root,
        dst_root=dst_root,
        num_images=num_images,
        seed=42,
        strict=False,
    )
