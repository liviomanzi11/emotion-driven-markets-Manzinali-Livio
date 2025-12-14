<h1 align="center"> Emotion-Driven Markets:<br>Social Media Sentiment and NASDAQ Movements</h1>

<p align="center">
  <em>Data Science & Advanced Programming – HEC Lausanne (Fall 2025)</em><br>
  <strong>Author:</strong> Livio Manzinali  
</p>

## Abstract

This project investigates whether **Twitter sentiment** can enhance the **prediction of short-term stock returns** for 6 major NASDAQ companies (AAPL, GOOG, GOOGL, AMZN, TSLA, MSFT). By combining **Natural Language Processing (NLP)** with **machine learning**, it compares the predictive power of **technical indicators** versus **sentiment features** using multiple models (Logistic Regression, Random Forest, XGBoost, LSTM) evaluated through backtesting strategies.

Using the **FinBERT** transformer model for financial sentiment extraction and merging these results with historical market data from **Yahoo Finance**, this project isolates sentiment to measure its predictive contribution against pure technical analysis within a behavioral finance framework.

## 1. Context and Motivation

The behavior of financial markets is increasingly influenced by collective sentiment expressed online. Platforms like **Twitter** serve as real-time barometers of investor mood, where reactions to corporate news, political announcements, or macroeconomic trends can spread instantly.  

This project applies principles of **behavioral finance** and **machine learning** to examine whether aggregated sentiment from Twitter can improve the predictive modeling of short-term stock returns for individual companies. Rather than describing correlations, it aims to evaluate the **predictive contribution** of sentiment compared to traditional market indicators through rigorous backtesting.

## 2. Research Objective

The main goal is to determine whether **social media sentiment** provides **predictive power** when forecasting daily stock returns for 6 NASDAQ companies.

Specifically, the project compares multiple modeling approaches:
- **Technical-only models:** Logistic Regression, Random Forest, XGBoost, LSTM trained exclusively on technical indicators
- **Sentiment-only models:** Logistic Regression, Random Forest, XGBoost, LSTM trained exclusively on sentiment features
- **Buy & Hold baseline** for benchmark comparison

All models are evaluated through backtesting with realistic equity curves and performance metrics (Total Return, Sharpe Ratio, Maximum Drawdown).

## 3. Methodology Overview

The project follows a reproducible workflow combining NLP, financial feature engineering, supervised learning, and strategy backtesting.

1. **Data Collection & Preparation**  
   Historical stock data (2015–2019) for 6 companies retrieved via `yfinance`.  
   Financial tweets from **Kaggle datasets** related to each company.  
   Merging tweet timestamps with daily stock data by company and date.

2. **Sentiment Analysis**  
   Application of **FinBERT** to classify tweets as positive, neutral, or negative.  
   Daily aggregation of sentiment metrics per company (polarity, weighted sentiment, tweet volume, engagement).

3. **Feature Engineering**  
   Creation of **15 technical features** (returns, MA, volatility, RSI, MACD) per company.  
   Creation of **15 sentiment features** (polarity, impact-weighted scores, deltas, volatility).  
   Train/test split with temporal validation (2015-2018 train, 2019 test).

4. **Model Training**  
   Train **8 models per company:** 4 technical-only models + 4 sentiment-only models.  
   Models: Logistic Regression, Random Forest, XGBoost, LSTM.  
   Each classical model trained with balanced class weights and StandardScaler (LR only).  
   LSTM uses 45-day sequences with bidirectional architecture.  

5. **Backtesting & Evaluation**  
   Long-only strategy: invest when model predicts probability > 0.5, else cash.  
   Metrics: Total Return, Sharpe Ratio, Maximum Drawdown.  
   Generate daily equity curves for visualization and comparison.

## 4. Repository Structure

```text
emotion-driven-markets/
│
├── main.py                 # Main entry point - Run this!
├── environment.yml         # Conda environment dependencies
├── AI_USAGE.md             # AI tools usage documentation
├── PROPOSAL.md
├── README.md
├── LICENSE
│
├── data/
│   ├── raw/                # Raw datasets (place CSV files here)
│   │   ├── Company.csv
│   │   ├── Company_Tweet.csv
│   │   └── Tweet.csv
│   └── processed/          # Auto-generated outputs
│       ├── company_stock_data.csv
│       ├── tweet_sentiment.csv
│       ├── merged_sentiment_stock_company.csv
│       ├── features_company/     # Feature-engineered datasets by ticker
│       ├── models/
│       │   ├── classical_company/  # Trained classical models by ticker
│       │   └── lstm_company/       # Trained LSTM models by ticker
│       └── samples/              # Sample outputs for testing
│
├── src/
│   ├── pipelines/
│   │   ├── yahoo_finance_company_data.py
│   │   ├── sentiment_pipeline_company.py
│   │   ├── merge_sentiment_company_daily.py
│   │   └── feature_engineering_company.py
│   ├── models/
│   │   ├── classical_models_company.py
│   │   └── lstm_models_company.py
│   ├── strategies/
│   │   └── backtest_company.py
│   └── visualization/
│       └── figures_company.py
│
├── results/
│   ├── classical_company/       # Classical model classification reports by ticker
│   ├── lstm_company/            # LSTM training history by ticker
│   ├── equity_curves/           # Daily equity curve CSV files (all strategies)
│   └── figures_company/         # Visualization outputs
│       └── strategy/            # Backtest comparison charts and results
│
└── tests/
    └── 
```

## 5. Installation and Data Setup

### **5.1 Clone Repository**
```bash
git clone https://github.com/liviomanzi11/emotion-driven-markets-Manzinali-Livio.git
cd emotion-driven-markets-Manzinali-Livio
```

### **5.2 Create Conda Environment**
```bash
conda env create -f environment.yml
conda activate emotion-driven-markets
```

**Note:** The environment includes all required packages:
- Python 3.11
- pandas, numpy, scikit-learn, xgboost
- tensorflow, pytorch, transformers
- yfinance, matplotlib, seaborn
- tqdm, joblib
## 5.2 Download Required Datasets

**Google Drive (RECOMMENDED):**  
https://drive.google.com/drive/folders/1MfgTLtyA0fM9Xrliyw2eE4cFIff44bkD?usp=sharing

**Required downloads:**

1. **Raw data** → place in `data/raw/`:
   - `Tweet.csv`
   - `Company_Tweet.csv`
   - `Company.csv`
   
   **Alternative:** Raw data can also be downloaded from Kaggle:  
   https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020

2. **Preprocessed sentiment** (MANDATORY) → place in `data/processed/`:
   - `tweet_sentiment.csv` (FinBERT sentiment scores for all tweets)

**Why is `tweet_sentiment.csv` required?**  
Running FinBERT sentiment analysis on all tweets takes **5-6 hours**. The preprocessed file is provided to save time. Download it from Google Drive and place it in `data/processed/`.

## 5.3 File Placement

After downloading from Google Drive, your structure must be:

```
emotion-driven-markets/
├── data/
│   ├── raw/
│   │   ├── Tweet.csv
│   │   ├── Company_Tweet.csv
│   │   └── Company.csv
│   └── processed/
│       └── tweet_sentiment.csv  # MANDATORY download from Drive
```

All other files in `data/processed/` are **automatically generated** when you run the pipeline.

## 5.4 Run the Complete Pipeline

```bash
python main.py
```

This executes all steps automatically:
1. Load preprocessed sentiment data from `data/processed/tweet_sentiment.csv`
2. Merge sentiment + stock data by company 
3. Feature engineering (15 technical + 15 sentiment features per company)
4. Train classical models (LR, RF, XGB) - separately for technical and sentiment
5. Train LSTM models - separately for technical and sentiment
6. Backtest all strategies (9 strategies × 6 companies = 54 backtests)
7. Generate visualization charts and equity curves

**Expected Runtime:** ~10-15 minutes (with preprocessed `tweet_sentiment.csv`)

## 5.5 Reproducibility

All random processes use fixed seeds to ensure reproducible results:
- Train/test temporal split (2015-2018 train, 2019 test)
- Model initialization (`random_state=42` for LR, RF, XGB; `seed=42` for LSTM)
- Feature ordering enforced via `feature_order.pkl` files
- LSTM sequence creation with consistent scaling

Running `python main.py` twice will produce identical models and backtest results.

## 6. References

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*.
- Bollen, J., Mao, H., & Zeng, X. (2011). *Twitter mood predicts the stock market.*
- Sul, H., Dennis, A., & Yuan, L. (2016). *Trading on Twitter: Using Social Media Sentiment to Predict Stock Returns*.
- Malo, P. et al. (2013). *Good debt or bad debt? Detecting semantic orientations in economic texts.*
- Zhang, W. (2017). *Deep learning for financial market prediction.*
- Kaggle datasets used for tweets and financial sentiment.
- Yahoo Finance API via `yfinance`.
- Course: **Data Science and Advanced Programming (DSAP)** – HEC Lausanne, University of Lausanne (2025).  

> **Note:**
This repository is part of the MSc in Finance – *Data Science and Advanced Programming* course at HEC Lausanne (Fall 2025).
Licensed under the MIT License.

<p align="center">
  © 2025 Livio Manzinali – HEC Lausanne  
</p>
