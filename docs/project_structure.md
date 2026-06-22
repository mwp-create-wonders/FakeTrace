# Project Structure

This repository now follows a standard split between product code, configs, docs, scripts, and model assets.

## Root Layout

```text
FakeTrace/
├─ app.py                  # Image CLI wrapper
├─ audio_app.py            # Audio experiment CLI wrapper
├─ web_app.py              # Local web server wrapper
├─ configs/                # Runtime and experiment configs
├─ docs/                   # Project documentation
├─ legacy/                 # Historical code kept for reference
├─ models/                 # Third-party model code and checkpoints
├─ scripts/                # Helper scripts and PowerShell runners
└─ src/
   └─ faketrace_app/       # Main Python application package
```

## Application Package

```text
src/faketrace_app/
├─ api/                    # FastAPI app and HTTP routes
├─ core/                   # Shared config and path utilities
├─ features/               # Feature-specific business logic
├─ ui/                     # Front-end assets
├─ audio_cli.py            # Audio CLI implementation
├─ cli.py                  # Image CLI implementation
├─ config.py               # Compatibility export for core.config
├─ inference_engine.py     # Compatibility export for detector service
├─ paths.py                # Compatibility export for core.paths
└─ web.py                  # FastAPI app export
```

## Notes

- Root entry files are intentionally thin wrappers so the real implementation stays under `src/`.
- `legacy/` contains historical code that is not part of the main runtime path.
- Static assets should live under `src/faketrace_app/ui/assets/` instead of the repository root.
