# AI Stock Market Analyzer — Local 2023 demo

## Folder layout

```
stock_market_analyzer/
  app.py
  requirements.txt
  models/
    best_stock_forecaster.pth
    gru_preprocessing.joblib
    finbert/   # config.json, tokenizer, weights, etc.
  data/
    stock_features.csv
    test_predictions_2023.csv
    clean_stock_news.csv
```

## Install and run

Use your existing PyTorch environment. Avoid reinstalling PyTorch if your CUDA build works.

```
pip install -r requirements.txt
ollama list
streamlit run app.py
```

If `ollama list` does not show `mistral:7b-instruct-v0.3-q4_K_M`, edit `OLLAMA_MODEL` in `app.py` to your exact installed tag. Ensure Ollama's local service is running.

## What the app does

- Rebuilds the original 14 features from stock_features.csv, using saved RobustScaler and 30-session windows.
- Loads the original 12,633-parameter GRU and performs genuine inference.
- Uses only previous sessions to forecast the chosen 2023 target date.
- Shows the last 10 prior saved predictions and outcomes; verifies reconstructed forecast against the saved prediction file.
- Extracts relevant excerpts from news dated strictly **before the last input session** (conservative because news timestamps are date-only).
- Runs FinBERT on up to 512 tokens per article and supplies selected evidence to Mistral through Ollama.
- Displays actual target-day price **only for evaluation**; it is excluded from Mistral's prompt.

## Important caveats

1. Historical 2023 demo, not a live forecasting service.
2. The historical pipeline removes rows without a next-day target, matching the original training sequence construction; future live inference would require a separate latest-row path.
3. `volatility_20d` units depend on original feature engineering. Verify whether it is a decimal or percentage before presenting the number. The UI uses a heuristic and warns if values are unusually large.
4. FinBERT scores reflect selected excerpts, not a full-document sentiment assessment. A ticker label alone does not prove an article is relevant to the company.
5. News date alone does not prove publication time. We exclude same-day news at the input cutoff; historical source availability is not fully verified.
6. Reconstructed inference should agree with `test_predictions_2023.csv` to within numerical tolerance. If not, check feature engineering, preprocessing, model weights and original data ordering.
7. Do not treat 10 sessions of directional accuracy as reliable evidence of model performance. Report full held-out test metrics separately.
8. `joblib.load` and `torch.load` should only load trusted local artifacts.
