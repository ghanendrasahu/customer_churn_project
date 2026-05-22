# Telco Customer Churn Prediction

End-to-end ML project with:
- model training pipeline (`src/churn_analysis.py`)
- FastAPI inference service (`app/api.py`)
- Streamlit dashboard (`app/app.py`)

## Project Structure

```text
customer_churn_project/
  app/
    api.py
    app.py
  data/
    telco_churn.csv          # generated dataset
  models/
    model_artifacts.pkl      # generated model artifacts
  src/
    generate_dataset.py
    churn_analysis.py
  requirements.txt
  README.md
```

## Quick Start

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Generate data
python -m src.generate_dataset

# 3) Train model (creates models/model_artifacts.pkl)
python -m src.churn_analysis

# 4) Run API
uvicorn app.api:app --reload

# 5) In another terminal, run dashboard
streamlit run app/app.py
```

## Notes

- `data/` and `models/` contain generated artifacts and are ignored by Git.
- `outputs/` is optional generated visualization output and is ignored by Git.

## Future Improvements

- Refactor FastAPI into a production-style structure (`routers/`, `schemas/`, `services/`).
- Add `/health` and `/reload` endpoints with model/version metadata.
- Implement heuristic fallback when trained artifacts are unavailable.
- Add prediction logging (`JSONL` or SQLite) for monitoring and traceability.
- Add unit and integration tests (`pytest`) for training + API routes.
- Containerize with Docker and add one-command local startup.
- Add GitHub Actions CI (lint, test, build checks on every push/PR).
- Add post-training evaluation report (metrics + plots).
- Add SHAP-based explainability for richer prediction insights.
- Add basic drift checks for incoming feature distributions.
- Add API auth/rate limiting for safer external use.
- Add deployment guide and live demo link (Render/Railway/AWS).
