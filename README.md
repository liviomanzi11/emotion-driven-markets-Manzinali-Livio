<h1 align="center"> Emotion-Driven Markets:<br>Social Media Sentiment and NASDAQ Movements</h1>

<p align="center">
  <em>Data Science & Advanced Programming – HEC Lausanne (Fall 2025)</em><br>
  <strong>Author:</strong> Livio Manzinali  
</p>

## Abstract

This project investigates whether **Twitter sentiment** can enhance the **prediction of short-term movements** in the **NASDAQ Composite Index**.  By combining **Natural Language Processing (NLP)** with **machine learning**, it explores whether the tone of online discussions about major NASDAQ-listed companies — such as Apple, Tesla, and Amazon — provides useful signals for forecasting daily market direction.  

Using the **FinBERT** transformer model for financial sentiment extraction and merging these results with historical market data from **Yahoo Finance**, this project integrates behavioral finance concepts into a predictive data science framework.

## 1. Context and Motivation

The behavior of financial markets is increasingly influenced by collective sentiment expressed online. Platforms like **Twitter** serve as real-time barometers of investor mood, where reactions to corporate news, political announcements, or macroeconomic trends can spread instantly.  

This project applies principles of **behavioral finance** and **machine learning** to examine whether aggregated sentiment from Twitter can improve the predictive modeling of short-term changes in the NASDAQ index. Rather than describing correlations, it aims to evaluate the **predictive contribution** of sentiment compared to traditional market indicators.

## 2. Research Objective

The main goal is to determine whether **social media sentiment** provides **additional predictive power** when forecasting daily NASDAQ movements.  

Specifically, the project compares two modeling approaches:
- A **baseline model** using only technical indicators (returns, moving averages, volatility).  
- A **hybrid model** that integrates both technical and sentiment-based features.

Machine learning models such as **Logistic Regression**, **Random Forest**, and **XGBoost** will be trained and evaluated on historical data to test this hypothesis.

## 3. Methodology Overview

The project follows a reproducible workflow combining NLP, financial feature engineering, and supervised learning.

1. **Data Collection & Preparation**  
   Historical NASDAQ data (2015–2020) retrieved via `yfinance`.  
   Financial tweets from **Kaggle datasets** related to major NASDAQ-listed companies.  
   Cleaning and time alignment of tweets with trading days.  

2. **Sentiment Analysis**  
   Application of **FinBERT** to classify tweets as positive, neutral, or negative.  
   Daily aggregation of sentiment metrics (average score, sentiment proportions).  

3. **Feature Engineering & Modeling**  
   Merge sentiment and market indicators into a unified dataset.  
   Train **Logistic Regression**, **Random Forest**, and **XGBoost** to predict next-day market direction.  
   Compare predictive performance with and without sentiment features.

4. **Evaluation & Interpretation**  
   Temporal validation (training 2015–2018, testing 2019–2020).  
   Performance measured using classification accuracy and feature importance analysis.  
   Visualization of sentiment versus market movements.

## 4. Repository Structure

```text
emotion-driven-markets/
│
├── data/                # Raw and processed datasets
├── notebooks/           # Exploratory and analysis notebooks
├── src/                 # Source code (data processing, NLP, ML)
│   ├── preprocessing.py
│   ├── sentiment_extraction.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── results/             # Figures, reports, and model outputs
├── tests/               # Unit tests for preprocessing and model pipelines
├── PROJECT_SPECIFICATION.md   # Detailed project outline
├── PROPOSAL.md                # Project proposal document
└── README.md                  # Overview and documentation
```

## 5. Installation and Usage

### Requirements
- Python 3.10+  
- Dependencies listed in `requirements.txt`  
- Recommended IDE: **Visual Studio Code**

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/emotion-driven-markets.git
cd emotion-driven-markets

# Install dependencies
pip install -r requirements.txt
```

### Running the Project
```bash
# Run preprocessing and sentiment analysis
python src/preprocessing.py
python src/sentiment_extraction.py

# Feature engineering and model training
python src/feature_engineering.py
python src/model_training.py

# Evaluate models and visualize results
python src/evaluation.py
```

Model results, plots, and summary statistics will be saved in the `/results` directory.

## 6. References and Acknowledgements

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.*  
- Bollen, J., Mao, H., & Zeng, X. (2011). *Twitter mood predicts the stock market.* *Journal of Computational Science.*  
- Gentzkow, M., Kelly, B., & Taddy, M. (2019). *Text as Data.* *Journal of Economic Literature.*  
- Yahoo Finance API via `yfinance` Python library.  
- Kaggle Datasets: *Financial Sentiment Analysis* and *Tweets about Top NASDAQ Companies (2015–2020).*  
- Course: **Data Science and Advanced Programming (DSAP)** – HEC Lausanne, University of Lausanne (2025).  
- Supervisors: *Prof. Simon Scheidegger* and *Dr. Anna Smirnova.*

> **Note:**
This repository is part of the MSc in Finance – *Data Science and Advanced Programming* course at HEC Lausanne (Fall 2025).
Licensed under the MIT License.

<p align="center">
  © 2025 Livio Manzinali – HEC Lausanne  
</p>
