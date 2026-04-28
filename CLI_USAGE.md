# MARC Local App

The project keeps `Training/` as the development and training area. Runtime
entry points live outside it and call the model through `src/marc_app`.

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
- `src/marc_app/inference_engine.py`: shared model loading and prediction logic
- `src/marc_app/cli.py`: command line entry point used by `app.py`
- `src/marc_app/web.py`: FastAPI app used by `web_app.py`
- `Training/`: preserved training and model-development files

Notes:

- Default checkpoint: `Training/pretrained/model_best.pth`
- Default threshold: `real_probability >= 0.5` means `real`; otherwise `fake`
- Use `--device auto`, `--device cpu`, or CUDA devices such as `--device cuda:0`
