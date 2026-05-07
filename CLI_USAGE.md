# FakeTrace CLI

The project keeps algorithm implementations in `MARC/` and `Trufor/`.
Runtime entry points live outside them and call the app through `src/faketrace_app`.

## CLI

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
- `MARC/`: MARC detection implementation
- `Trufor/`: TruFor localization implementation

Notes:

- Default checkpoint: `MARC/pretrained/model_best.pth`
- Default threshold: `real_probability >= 0.5` means `real`; otherwise `fake`
- Use `--device auto`, `--device cpu`, or CUDA devices such as `--device cuda:0`
