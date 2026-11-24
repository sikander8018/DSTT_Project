# G16 — Intelligent Stock Market Analyzer (Term Project)

Complete submission repository containing **Part 1** (Data Selection, Preprocessing & Exploratory Data Analysis) and **Part 2** (Time-Series Forecasting, Model Comparison & Streamlit + FinBERT + Mistral 7B Analyzer).

## Repository Structure

```text
Final/
├── Part 1/
│   ├── final_submission.ipynb                  # Complete Part 1 notebook (with outputs)
│   ├── 01_stock_selection.ipynb
│   ├── 02_stock_preprocessing_features.ipynb
│   ├── 03_article_preprossing.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── data/raw/
│   │   ├── filtered_stock_prices.csv
│   │   └── filtered_financial_news.csv         # >100 MB (.gitignored)
│   └── outputs/
│       ├── stock_features.csv
│       ├── clean_stock_prices.csv
│       ├── daily_news_counts.csv
│       ├── quality_report.csv
│       ├── eda_insights.md
│       ├── clean_financial_news.csv            # >100 MB (.gitignored)
│       └── figures/                            # fig_01 .. fig_15
└── Part 2/
    ├── app.py                                  # Streamlit 2023 historical analyzer
    ├── requirements.txt                        # Streamlit UI dependencies
    ├── data/
    │   ├── stock_features.csv
    │   ├── test_predictions_2023.csv
    │   └── clean_stock_news.csv                # >100 MB (.gitignored)
    ├── models/
    │   ├── best_stock_forecaster.pth           # Saved 12,633-param GRU weights
    │   ├── gru_preprocessing.joblib            # Saved RobustScaler & config
    │   └── finbert/                            # Local FinBERT tokenizer & weights
    └── model_training/
        ├── 05_stock_price_forecasting.ipynb    # Complete Part 2 notebook (with outputs)
        ├── requirements_part2.txt              # Modeling dependencies (Python 3.11)
        └── outputs/
            ├── model_comparison_metrics.csv
            ├── per_ticker_test_metrics_2023.csv
            ├── gru_complexity_results.csv
            └── figures/                        # fig_01 .. fig_06
```

---

## Reproducing Part 2 Modeling (`05_stock_price_forecasting.ipynb`)

1. Create and activate a Python 3.11 environment, then install modeling dependencies:
   ```bash
   pip install -r "Part 2/model_training/requirements_part2.txt"
   ```
2. Open and run `Part 2/model_training/05_stock_price_forecasting.ipynb`.
   - `Part 2/data/stock_features.csv` (and `Part 1/outputs/stock_features.csv`) are included directly in the repository, so the notebook runs without needing the >100 MB news CSVs.

---

## Running the Streamlit Analyzer (`Part 2/app.py`)

### 1. Install UI Dependencies
Use your existing PyTorch environment (avoid overwriting a working CUDA build):
```bash
cd "Part 2"
pip install -r requirements.txt
```

### 2. Large Files Excluded from GitHub (>100 MB)
GitHub enforces a 100 MB per-file limit, so the following files are listed in `.gitignore` (they are included in the ZIP submission if distributed directly):

- **FinBERT weights (`Part 2/models/finbert/model.safetensors`, 438 MB):**
  Download `ProsusAI/finbert` locally into `Part 2/models/finbert/`:
  ```bash
  python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; m='ProsusAI/finbert'; AutoTokenizer.from_pretrained(m).save_pretrained('Part 2/models/finbert'); AutoModelForSequenceClassification.from_pretrained(m).save_pretrained('Part 2/models/finbert')"
  ```
- **Cleaned News CSV (`Part 2/data/clean_stock_news.csv`, 388 MB) & Part 1 News CSVs (`Part 1/data/raw/filtered_financial_news.csv`, `Part 1/outputs/clean_financial_news.csv`):**
  Regenerate from the raw FNSPID / Benzinga financial news dataset by running `Part 1/03_article_preprossing.ipynb` (or place the provided `clean_stock_news.csv` into `Part 2/data/`).

### 3. Local Mistral 7B via Ollama
`Part 2/app.py` uses a local Ollama server (`http://127.0.0.1:11434/api/chat`) with `mistral:7b-instruct-v0.3-q4_K_M` to generate the narrative analysis:
1. Install Ollama from [ollama.com](https://ollama.com) and ensure the Ollama service is running.
2. Pull the model:
   ```bash
   ollama pull mistral:7b-instruct-v0.3-q4_K_M
   ollama list
   ```
   *(If you use a different Mistral tag, update `OLLAMA_MODEL` in `Part 2/app.py`.)*

### 4. Launch the App
```bash
cd "Part 2"
streamlit run app.py
```
