   <h1 align="center"> Emotion-Driven Markets:<br>Machine Learning for Stock Prediction Using Technical Indicators and Twitter Sentiment</h1>

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
      Generate daily equity curves and charts for visualization and comparison.

   ## 4. Repository Structure

   ```text
   emotion-driven-markets/
   │
   ├── main.py                 # Main entry point - Run this!
   ├── requirements.txt        # Python dependencies (pip)
   ├── environment.yml         # Conda environment (alternative)
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
   ```

   ## 5. Installation and Data Setup

   ### **5.1 Clone Repository**
   ```bash
   git clone https://github.com/liviomanzi11/emotion-driven-markets-Manzinali-Livio.git
   cd emotion-driven-markets-Manzinali-Livio
   ```

   ### **5.2 Install Dependencies**

   **Option A: Using pip (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS/Linux
   # venv\Scripts\activate      # Windows
   
   pip install -r requirements.txt
   ```

   **Option B: Using Conda**
   ```bash
   conda env create -f environment.yml
   conda activate emotion-driven-markets
   ```

   **Note:** Both options install the same packages:
   - Python 3.11.5
   - pandas 2.1.1, numpy 1.26.0, scikit-learn 1.3.1, xgboost 2.0.0
   - tensorflow 2.13.0, torch 2.0.1, transformers 4.33.2
   - yfinance 0.2.28, matplotlib 3.8.0, seaborn 0.13.0
   - tqdm 4.66.1, joblib 1.3.2, pytest 7.4.2

   ## 5.3 Download Required Datasets

   **Google Drive (RECOMMENDED):**  
   https://drive.google.com/drive/folders/1MfgTLtyA0fM9Xrliyw2eE4cFIff44bkD?usp=sharing
   
   **Required downloads (MANDATORY):**

   1. **Raw data** from Google Drive → place in `data/raw/`:
      - `Tweet.csv`
      - `Company_Tweet.csv`
      - `Company.csv`
      
      **Alternative:** Raw data can also be downloaded from Kaggle:  
      https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020

      The Kaggle download will provide a ZIP file. You must unzip it then place in `data/raw/`:
      - `Tweet.csv`
      - `Company_Tweet.csv`
      - `Company.csv`

   2. **Preprocessed sentiment** from Google Drive → place in `data/processed/`:
      - `tweet_sentiment.csv` (FinBERT sentiment scores for all tweets)

   **Why is `tweet_sentiment.csv` required?**  
   Running FinBERT sentiment analysis on all tweets takes **5-6 hours**. The preprocessed file is provided to save time. Download it from Google Drive and place it in `data/processed/`.

   3. **Stock market data** from Google Drive → place in `data/processed/`:
      - `company_stock_data.csv` (Historical OHLCV data for 6 companies: AAPL, GOOG, GOOGL, AMZN, TSLA, MSFT)

   **Why is `company_stock_data.csv` required?**  
   Yahoo Finance API can return micro-variations in precision on each download, breaking reproducibility. The preprocessed file uses the exact data from the original analysis, ensuring **100% identical results**. Download it from Google Drive and place it in `data/processed/`.

   **Note:** If `company_stock_data.csv` is missing, the pipeline will automatically download from Yahoo Finance with 2-decimal rounding as a fallback. However, for perfect reproducibility, use the provided file.

   ## 5.4 File Placement

   After downloading from Google Drive, your data structure must be:

   ```
   emotion-driven-markets/
   ├── data/
   │   ├── raw/
   │   │   ├── Tweet.csv
   │   │   ├── Company_Tweet.csv
   │   │   └── Company.csv
   │   └── processed/
   │       ├── tweet_sentiment.csv
   │       └── company_stock_data.csv
   ```

   All other files in `data/processed/` and all other folders are **automatically generated** when you run the pipeline.

   ## 5.5 Run the Complete Pipeline

   ```bash
   python main.py
   ```

   The interactive menu offers two execution modes:
   - **[1] Full pipeline execution:** Run all 8 steps automatically without interruption
   - **[2] Step-by-step execution:** Press ENTER to proceed to each next step, 's' to skip a step, or 'q' to quit

   When running the full pipeline (option 1), it executes all steps automatically:
   1. Load preprocessed sentiment data from `data/processed/tweet_sentiment.csv`
   2. Load stock data from `data/processed/company_stock_data.csv` (or download from Yahoo if missing)
   3. Merge sentiment + stock data by company 
   4. Feature engineering (15 technical + 15 sentiment features per company)
   5. Train classical models (LR, RF, XGB) - separately for technical and sentiment
   6. Train LSTM models - separately for technical and sentiment
   7. Backtest all strategies (9 strategies × 6 companies = 54 backtests)
   8. Generate visualization charts and equity curves

   **Expected Runtime:** ~25-30 minutes (with preprocessed `tweet_sentiment.csv` and `company_stock_data.csv`)

   ## 5.6 Reproducibility

   All random processes use fixed seeds to ensure reproducible results:
   - **Yahoo Finance data:** Cached file (`company_stock_data.csv`) ensures identical stock data across runs
   - **FinBERT sentiment:** Preprocessed file (`tweet_sentiment.csv`) ensures identical sentiment scores
   - Train/test temporal split (2015-2018 train, 2019 test)
   - Model initialization (`random_state=42` for LR, RF, XGB; `seed=42` for LSTM)
   - Feature ordering enforced via `feature_order.pkl` files
   - LSTM: CPU-only execution, seeded initializers, shuffle=False
   - Random Forest: single-threaded (`n_jobs=1`), no subsampling
   - XGBoost: `tree_method='exact'`, single-threaded

   Running `python main.py` multiple times with the provided `tweet_sentiment.csv` and `company_stock_data.csv` will produce **identical** models and backtest results.

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
