# Features

Place future product capabilities here.

Suggested pattern for each module:

- `features/<feature_name>/service.py`: feature business logic
- `features/<feature_name>/schemas.py`: request and response models
- `api/routes/<feature_name>.py`: HTTP routes for the feature
- `ui/assets/<feature_name>/...`: dedicated front-end assets when needed

Current module:

- `features/detector`: existing image real/fake detection
