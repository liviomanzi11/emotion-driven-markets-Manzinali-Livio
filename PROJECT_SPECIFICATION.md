# Project Specification: Social Media Sentiment and NASDAQ Movements

## 0. Introduction

In recent years, social media has become a central space for expressing opinions that influence public sentiment and investor behavior. Platforms such as **Twitter** play a crucial role in shaping market expectations, particularly for large technology companies that dominate the **NASDAQ Composite Index**. 

Understanding whether these online emotions can help explain or predict market dynamics has become a relevant question in both **behavioral finance** and **applied data science**.

This project aims to bridge the gap between textual sentiment analysis and quantitative market modeling. By combining **Natural Language Processing (NLP)** and **machine learning**, it investigates whether social media sentiment provides predictive insights into short-term financial fluctuations.


## 1. Project Overview

The objective of this project is to **predict short-term NASDAQ movements** using a combination of **Twitter sentiment** and **technical market indicators**. The analysis focuses on financial tweets about major NASDAQ-listed firms such as Apple, Tesla, Amazon, and Google, combined with historical market data retrieved from Yahoo Finance.  

Unlike previous correlation-based studies, this project treats the question as a **supervised learning problem**, comparing models trained with and without sentiment-related variables. The goal is to determine whether sentiment extracted from social media improves predictive accuracy relative to a purely quantitative baseline.


## 2. Research Question / Problem Statement

The main research question is:

> **Does social media sentiment improve the ability of machine learning models to predict short-term movements in the NASDAQ Composite Index?**

This raises three sub-questions:
- Can sentiment-based features derived from Twitter enhance market prediction compared to technical indicators alone?  
- Do shifts in sentiment precede or follow market changes?  
- Which modeling approach best captures the relationship between sentiment and short-term returns?  


## 3. Datasets

The analysis combines textual and financial data from multiple open sources:

- **Financial Sentiment Analysis (Kaggle)** — Labeled financial sentences used to evaluate and calibrate FinBERT’s performance.  
- **Tweets about Top NASDAQ Companies (2015–2020)** — A dataset containing tweets about major firms such as Apple, Tesla, and Amazon, used to compute daily sentiment indicators.  
- **Yahoo Finance (^IXIC)** — Historical NASDAQ Composite data (prices, returns, volatility) obtained through the `yfinance` API.  

All datasets will be cleaned, normalized, and aligned by trading date to create a unified feature table containing both textual and quantitative information.


## 4. Methodology

The project follows a structured workflow combining NLP-based sentiment analysis, financial feature engineering, and predictive modeling.  

### Step 1 – Data Preparation  
Tweets will be cleaned to remove duplicates, URLs, and special characters, and their timestamps will be aligned with market trading days. Historical NASDAQ data will be retrieved via `yfinance`, and technical indicators such as moving averages, RSI, and volatility will be computed using the `ta` library.  

### Step 2 – Sentiment Analysis (NLP)  
The **FinBERT** transformer model will be applied to classify tweets as *positive*, *neutral*, or *negative*. Daily averages and polarity ratios will then be computed to produce sentiment indicators representing the prevailing market tone.

### Step 3 – Predictive Modeling  
The dataset will include two feature sets:
- **Baseline model:** technical indicators only;  
- **Hybrid model:** combination of sentiment and technical indicators.  

Models such as **Logistic Regression**, **Random Forest**, and **XGBoost** will be trained to predict the next-day market direction. Temporal validation will be used to simulate real predictive performance (train: 2015–2018, test: 2019–2020).

### Step 4 – Evaluation  
Model performance will be assessed using **accuracy**, **F1-score**, and **ROC-AUC** metrics. Feature importance and visualization techniques will help interpret which factors—sentiment or technical—drive the model’s predictions.


## 5. Tools and Libraries

- **Python 3.10+**  
- **Pandas, NumPy, Matplotlib, Seaborn** – Data manipulation and visualization  
- **Transformers (FinBERT)** – Financial sentiment analysis  
- **yfinance, ta** – Market data and technical indicators  
- **Scikit-learn, XGBoost** – Machine learning and model evaluation  
- **VS Code / Jupyter Notebook** – Development environment  


## 6. Expected Outcomes

The project aims to assess whether sentiment-based features add measurable predictive value to models of short-term NASDAQ performance. It is expected that a hybrid model combining both sentiment and technical features will outperform a purely quantitative baseline. 

Even if the predictive gains are limited, the study will provide valuable insights into how digital sentiment and investor emotions interact with financial market dynamics.


## 7. Repository Structure

```text
emotion-driven-markets/
│
├── data/                # Raw and processed datasets
├── notebooks/           # Exploratory and model analysis notebooks
├── src/                 # Source code (data processing, NLP, ML)
│   ├── preprocessing.py
│   ├── sentiment_extraction.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── results/             # Figures, reports, and model outputs
├── tests/               # Unit tests for preprocessing and modeling
├── PROJECT_SPECIFICATION.md   # Detailed project description
├── PROPOSAL.md                # Project proposal document
└── README.md                  # Project overview and documentation
```

## 8. Summary

This project combines Natural Language Processing, financial feature engineering, and machine learning to investigate whether online sentiment provides useful signals for market prediction. 

By integrating textual data from social media with quantitative indicators, it seeks to evaluate the predictive potential of collective investor sentiment and its role in short-term NASDAQ dynamics.
