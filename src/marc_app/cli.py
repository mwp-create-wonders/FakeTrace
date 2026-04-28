import argparse
import csv
import json
from pathlib import Path

from .config import load_config
from .inference_engine import MARCInferenceEngine, collect_image_paths
from .paths import DEFAULT_CONFIG_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="MARC real/fake image detector CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image path. Can be passed multiple times.",
    )
    input_group.add_argument("--image-dir", help="Directory containing images.")

    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Config JSON path.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, cuda, cuda:0.")
    parser.add_argument("--recursive", action="store_true", help="Scan image-dir recursively.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument("--image-size", type=int, default=None, help="Override input image size.")
    parser.add_argument("--threshold", type=float, default=None, help="Override real/fake threshold.")
    parser.add_argument("--save-json", default="", help="Optional JSON output path.")
    parser.add_argument("--save-csv", default="", help="Optional CSV output path.")
    return parser.parse_args()


def build_overrides(args) -> dict:
    overrides = {}
    for key in ("checkpoint", "device", "batch_size", "image_size", "threshold"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    return overrides


def print_results(results, threshold):
    print("\nMARC CLI inference")
    print(f"Threshold: real_probability >= {threshold:.3f} => real")
    print("-" * 92)
    print(f"{'prediction':10s} {'real_prob':>10s} {'fake_prob':>10s}  file")
    print("-" * 92)
    for item in results:
        print(
            f"{item.prediction:10s} "
            f"{item.real_probability:10.4f} "
            f"{item.fake_probability:10.4f}  "
            f"{item.filename}"
        )
    print("-" * 92)
    print(f"Total: {len(results)} image(s)\n")


def save_outputs(results, save_json, save_csv):
    rows = [item.to_dict() for item in results]
    if save_json:
        json_path = Path(save_json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {json_path}")

    if save_csv:
        csv_path = Path(save_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "path",
                    "filename",
                    "real_probability",
                    "fake_probability",
                    "prediction",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {csv_path}")


def main():
    args = parse_args()
    source_paths = args.image if args.image else [args.image_dir]
    paths = collect_image_paths(source_paths, recursive=args.recursive)
    config = load_config(args.config, build_overrides(args))
    engine = MARCInferenceEngine(config)
    results = engine.predict_paths(paths)
    print(f"Device: {engine.device}")
    print(f"Checkpoint: {config.checkpoint}")
    print_results(results, config.threshold)
    save_outputs(results, args.save_json, args.save_csv)


if __name__ == "__main__":
    main()
