# Features

Place future product capabilities here.

Suggested pattern for each module:

- `features/<feature_name>/service.py`: feature business logic
- `features/<feature_name>/schemas.py`: request and response models
- `api/routes/<feature_name>.py`: HTTP routes for the feature
- `ui/assets/<feature_name>/...`: dedicated front-end assets when needed

Current modules:

- `features/detector`: existing image real/fake detection
- `features/audio`: ATADD-style audio real/fake detection

Audio experiment helpers migrated from ATADD:

- `features/audio/config.py`: YAML experiment config loader
- `features/audio/dataset.py`: manifest-based audio dataset
- `features/audio/metrics.py`: accuracy, macro F1, and type-aware F1 metrics
- `features/audio/experiment.py`: train/eval runners
- `features/audio/predict_export.py`: batch prediction, `predict.csv`, and `submission.zip`
- `features/audio/threshold_scan.py`: fake-probability threshold scan for calibration
- `features/audio/healthcheck.py`: manifest and preprocessing smoke test
