# FakeTrace

FakeTrace is now organized as a single app package, `faketrace_app`, with external model implementations grouped under a shared `models/` directory.

## Structure

```text
FakeTrace/
├─ app.py                         # CLI entry
├─ web_app.py                     # Web entry
├─ configs/
│  └─ default.json                # Runtime defaults
├─ models/
│  ├─ CAT-Net/                    # CAT-Net localization
│  ├─ Fassa/                      # Fassa localization
│  ├─ effunetpp/
│  ├─ marc/                       # MARC detector implementation and weights
│  └─ trufor/                     # TruFor localization 
implementation and weights
├─ src/
│  └─ faketrace_app/
│     ├─ api/                     # FastAPI routes and dependency wiring
│     ├─ core/                    # Shared config and path helpers
│     ├─ features/
│     │  ├─ detector/             # Detection feature
│     │  └─ trufor/               # Localization feature
│     ├─ ui/                      # Web assets
│     ├─ cli.py                   # CLI workflow
│     ├─ config.py                # Public config exports
│     ├─ inference_engine.py      # Public detector exports
│     ├─ paths.py                 # Public path exports
│     └─ web.py                   # Web app exports
└─ requirements.txt
```

## Development Notes

- New app code should go under `src/faketrace_app/`.
- New model or third-party capability folders should go under `models/`.
- HTTP routes should follow `src/faketrace_app/api/routes/<feature>.py`.
- Feature logic should follow `src/faketrace_app/features/<feature>/service.py`.

