# Group 16: Intelligent Stock Market Analyzer

## Project Part 1: Dataset Acquisition, Preprocessing, Feature Engineering, and EDA

Course: Data Science Tools and Techniques
Group: G16
Submission: Assignment 2 (Term Project Part 1)

Team Members and Roles:
- Tayyab Akhtar: Dataset acquisition, stock selection logic, initial extraction, and data dictionary.
- Hamza: Stock price cleaning, financial feature engineering, risk indicator calculation, and article preprocessing.
- Sikander: Exploratory data analysis, statistical time-series diagnostics, stationarity testing, volatility analysis, and report synthesis.

---

## Project Overview

The goal of this project is to build an intelligent market regime and risk analyzer. The system combines quantitative stock price data with financial news articles to analyze market behavior, risk levels, and sector dynamics.

This first submission covers the complete first half of the machine learning pipeline:
1. Selecting our asset universe from raw market data.
2. Cleaning stock prices and financial news articles.
3. Engineering technical indicators and risk metrics.
4. Conducting exploratory data analysis with statistical tests and visualizations.
5. Extracting key data-backed insights for future predictive modeling.

---

## Selected Stocks and Dataset Source

Dataset Source:
We use the FNSPID (Financial News and Stock Price Integration Dataset) sourced from Hugging Face. The study covers nine full years from January 2, 2015 to December 28, 2023, consisting of 2,263 contiguous trading sessions.

Stock Universe:
Instead of modeling every stock, we selected five industry leaders from five different sectors, plus one broad market index ETF:
- SPY (SPDR S&P 500 ETF Trust): Broad market benchmark.
- NVDA (NVIDIA): Information Technology sector.
- WFC (Wells Fargo): Financials and banking sector.
- MRK (Merck and Co.): Healthcare and pharmaceuticals sector.
- OXY (Occidental Petroleum): Energy and oil sector.
- COST (Costco Wholesale): Consumer Staples sector.

Selection Criterion:
Each selected company had at least 100 published news articles in every single year from 2015 to 2023 in the raw dataset. This guaranteed consistent news coverage across the full nine-year timeline.

---

## Notebook Structure and Workflow

Primary Master Submission Notebook:
final_submission.ipynb
- Contains the complete end-to-end implementation from start to finish in one single file.
- Integrates all five assignment requirements: dataset acquisition justification, data quality audits, feature engineering, full EDA with all 15 visualizations, and empirical insights.
- Fully pre-executed with all charts, correlation heatmaps, ADF and KPSS stationarity test tables, and diagnostic metrics embedded directly inside the file.
- Uses portable relative path detection so it executes cleanly on any machine without path conflicts.

Supporting Modular Notebooks:
For transparency, our individual modular workflow notebooks are also preserved:
- Notebook 01: Stock Selection (01_stock_selection.ipynb) filters the FNSPID corpus and finalizes the five sector stocks and SPY benchmark.
- Notebook 02: Stock Price Preprocessing and Features (02_stock_preprocessing_features.ipynb) handles price cleaning, returns, moving averages, rolling volatility, drawdowns, and rolling Beta.
- Notebook 03: Financial News Preprocessing (03_article_preprossing.ipynb) audits news completeness, verifies midnight timestamps, and implements the leak-free calendar alignment.
- Notebook 04: Exploratory Data Analysis (04_exploratory_data_analysis.ipynb) performs the full visual and econometric time-series analysis.

---

## Key Observations and Findings

1. Stock Prices Are Non-Stationary, but Returns Are Stationary:
Statistical tests (ADF and KPSS) prove that raw stock prices and log prices are non-stationary with strong trends. However, daily percentage returns are strictly stationary. This means future linear models like ARIMA must difference the series, while LSTM networks should be trained on stationary returns and scaled features.

2. Volatility Clusters Strongly over Time:
Daily returns show almost zero linear autocorrelation, confirming weak-form market efficiency. But squared returns show strong positive correlation lasting over 30 trading days. This proves strong volatility clustering: calm periods stay calm, while high-risk turbulent periods stay turbulent. Gated recurrent networks like LSTM are well suited to capture these non-linear volatility regimes.

3. Heavy Tails and Severe Crash Risk:
All six assets have heavy tails with excess kurtosis ranging from 8.5 to 44.4 (a normal bell curve has 0.0). Historical drawdowns reached -88.65% for OXY and -70.17% for WFC. Real financial risks are asymmetric and require risk-focused loss functions.

4. Sector Regimes Decouple During Crises:
In the March 2020 COVID shock, all stocks crashed together as correlations spiked toward 1.0. But in 2022, rising interest rates hurt tech (NVDA fell roughly 50%) while commodity inflation lifted energy (OXY gained 115%). Using SPY returns and rolling Beta provides essential macro context that single-stock models lack.

5. Leak-Free News Alignment:
Because 99.85% of news timestamps are midnight UTC labels, news is mapped to the subsequent trading day. This prevents the model from peeking into information before it was available.

6. Dual-Track Handling of Days Without News:
News coverage ranges from 63% for NVDA to 92% for MRK. Dropping days without news would remove 37% of Nvidia price history and destroy time-series continuity. We keep all 2,263 trading days for price modeling and record news count as zero for non-news days. Furthermore, days with over 10 articles show twice the price volatility of zero-news days, proving news volume itself is an informative market attention signal.

---

## Required Libraries

The project is built using Python 3.11. The required external packages are:
- pandas (data manipulation and time-series alignment)
- numpy (numerical operations and array transformations)
- scipy (statistical calculations and kurtosis measurements)
- matplotlib (visualization generation)
- seaborn (statistical distribution plots and heatmaps)
- statsmodels (ADF test, KPSS test, ACF, PACF, and seasonal decomposition)
- jupyter and ipykernel (interactive notebook execution)
- python-dotenv (environment variable management)
- tqdm (progress tracking for iterative tasks)

Install all requirements in one step:
pip install pandas numpy scipy matplotlib seaborn statsmodels jupyter ipykernel python-dotenv tqdm

---

## Setup and Execution Instructions

Running Locally:
1. Open a terminal and navigate to the project directory.
2. Ensure Python 3.11 is installed.
3. Create and activate a virtual environment:
   python -m venv .venv
   Windows Command Prompt: .venv\Scripts\activate.bat
   Windows PowerShell: .venv\Scripts\Activate.ps1
4. Install dependencies:
   pip install pandas numpy scipy matplotlib seaborn statsmodels jupyter ipykernel python-dotenv tqdm
5. Launch Jupyter Notebook:
   jupyter notebook
6. Open and run the master submission notebook:
   final_submission.ipynb
   (Alternatively, the supporting modular notebooks 01 through 04 can be executed in numerical sequence).

Running in Google Colab:
1. Upload the project folder to Google Drive or open final_submission.ipynb directly in Colab.
2. In the first code cell of Colab, run:
   !pip install pandas numpy scipy matplotlib seaborn statsmodels python-dotenv tqdm
3. Run the notebook from top to bottom by selecting Runtime, then Run all.

