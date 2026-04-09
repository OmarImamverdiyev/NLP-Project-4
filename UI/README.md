# UI Dashboard

This folder contains a lightweight Streamlit dashboard for the project.

Goals:

- inspect saved Task 1 and Task 2 results
- preview reproducible training and evaluation commands
- test saved sentiment and QA models manually
- reuse locally cached model bundles so the UI does not redownload heavy checkpoints every time

## Install

```bash
pip install -r UI/requirements.txt
```

## Run

```bash
streamlit run UI/app.py
```

## Notes

- The dashboard is lazy by design: it only loads a model when you click a testing action.
- Future runs from `evaluate_sentiment140_bert.py` and `qa_system/train.py` will save `run_manifest.json` metadata for easier UI discovery.
- Transformers models cached for UI reuse are stored under `UI/artifacts/model_store/`.
