# FakeTrace CLI

Runtime entry points live in `app.py` and `web_app.py`. The app code is in `src/faketrace_app`, and model implementations are grouped in `models/`.

## CLI

### Image detection

```bash
python app.py --image path/to/image.jpg
```

Use the bundled checkpoint by default:

```bash
python app.py --image path/to/image.jpg --device cpu
```

Run on a folder:

```bash
python app.py --image-dir path/to/images --recursive --batch-size 4
```

Save machine-readable output:

```bash
python app.py --image-dir path/to/images --save-json results.json --save-csv results.csv
```

### Audio experiments

ATADD-style audio training, evaluation, and batch export are available from the unified `app.py audio ...` entrypoint. `audio_app.py` remains as a compatibility wrapper.

Check that the config, manifest, audio decoding, resampling, and padding/cropping path works:

```bash
python app.py audio audio-healthcheck ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --manifest data/manifests/val.csv ^
  --output-dir output/audio_healthcheck
```

Train from CSV manifests:

```bash
python app.py audio audio-train ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --train-manifest data/manifests/train.csv ^
  --val-manifest data/manifests/val.csv ^
  --output-dir output/audio_ast ^
  --device cuda
```

Evaluate a checkpoint:

```bash
python app.py audio audio-eval ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --checkpoint output/audio_ast/best.pt ^
  --manifest data/manifests/val.csv ^
  --output-dir output/audio_ast_eval ^
  --device cuda
```

Export `predict.csv`, optional probabilities, and `submission.zip`:

```bash
python app.py audio audio-predict ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --checkpoint output/audio_ast/best.pt ^
  --audio-dir data/eval_audio ^
  --output-dir output/audio_submission ^
  --fake-threshold 0.5 ^
  --save-probs
```

Scan fake-probability thresholds on a labeled manifest:

```bash
python app.py audio audio-threshold-scan ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --checkpoint output/audio_ast/best.pt ^
  --manifest data/manifests/val.csv ^
  --output-dir output/audio_thresholds ^
  --metric track2_macro_f1
```

Use the best threshold from a scan summary:

```bash
python app.py audio audio-predict ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --checkpoint output/audio_ast/best.pt ^
  --audio-dir data/eval_audio ^
  --output-dir output/audio_submission ^
  --threshold-summary output/audio_thresholds/threshold_scan_summary.json ^
  --save-probs
```

Manifest CSV files should contain at least `audio_path,label` for training and evaluation.
Optional columns `type` and `source_name` enable per-type metrics such as `track2_macro_f1`.

Install dependencies first:

```bash
pip install -r requirements.txt
```

## Web App

Start the local web app:

```bash
python web_app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Runtime Structure

- `configs/default.json`: runtime checkpoint, device, image size, threshold, and model settings
- `src/faketrace_app/inference_engine.py`: shared model loading and prediction logic
- `src/faketrace_app/cli.py`: command line entry point used by `app.py`
- `src/faketrace_app/web.py`: FastAPI app used by `web_app.py`
- `src/faketrace_app/audio_cli.py`: audio experiment subcommands used by the unified CLI and the legacy `audio_app.py` wrapper
- `models/marc/`: MARC detection implementation
- `models/trufor/`: TruFor localization implementation

Notes:

- Default checkpoint: `models/marc/pretrained/model_best.pth`
- Default threshold: `real_probability >= 0.5` means `real`; otherwise `fake`
- Use `--device auto`, `--device cpu`, or CUDA devices such as `--device cuda:0`
