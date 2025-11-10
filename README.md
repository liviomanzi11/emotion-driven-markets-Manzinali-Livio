# Emotion-Driven Markets: Social Media Sentiment and NASDAQ Movements

## Abstract

This project investigates the relationship between **social media sentiment** and **short-term fluctuations** in the **NASDAQ Composite Index**.  
Using natural language processing (NLP) and financial data analysis, the study examines whether shifts in public tone regarding major NASDAQ-listed companies—such as Apple, Tesla, and Amazon—correlate with or precede market movements.  
By applying the **FinBERT** transformer model for sentiment classification on historical tweets and merging these results with market data from **Yahoo Finance**, the project contributes to understanding how digital sentiment interacts with financial markets.


## 1. Context and Motivation

In modern financial markets, investor psychology and public opinion play a critical role in shaping price dynamics.  
Platforms such as **Twitter** serve as real-time reflections of collective sentiment, and viral reactions or influential statements can instantly affect asset valuations.  
This project builds on behavioral finance theory by quantifying whether aggregated online sentiment can explain or anticipate short-term NASDAQ movements.  
Beyond potential predictive insights, it aims to shed light on how efficiently markets integrate public information from social media.


## 2. Research Objective and Questions

The main objective is to determine whether **social media sentiment** is **statistically associated** with **NASDAQ market performance**.

**Research Questions:**
1. Does aggregated sentiment about major NASDAQ companies correlate with daily market returns?  
2. Are sentiment shifts leading or lagging indicators of market movements?  
3. Can sentiment-based features enhance short-term predictive financial models?


## 3. Methodology Overview

The project follows a structured workflow combining NLP and quantitative finance:

1. **Data Collection & Preparation**  
   - Twitter datasets (2015–2020) for major NASDAQ firms (Apple, Tesla, Amazon, Google).  
   - NASDAQ index data retrieved via the `yfinance` library.  
   - Cleaning: remove URLs, mentions, and non-alphanumeric characters; align timestamps with trading days.

2. **Sentiment Analysis**  
   - Sentiment classification using **FinBERT**, a transformer model pre-trained on financial language.  
   - Daily aggregation of sentiment scores (positive, neutral, negative).

3. **Financial Analysis**  
   - Correlation analysis between sentiment and NASDAQ returns.  
   - Exploratory regression to test predictive potential.

4. **Visualization and Interpretation**  
   - Compare time series of sentiment and market performance.  
   - Identify periods where sentiment trends preceded major market moves.


## 4. Repository Structure

```text
emotion-driven-markets/
│
├── data/                # Raw and processed datasets
├── notebooks/           # Exploratory and analysis notebooks
├── src/                 # Source code (data processing, NLP, visualization)
│   ├── preprocessing.py
│   ├── sentiment_analysis.py
│   ├── merge_finance_data.py
│   └── correlations.py
├── reports/             # Generated figures and the final project report
├── tests/               # Unit tests for data and model validation
├── PROJECT_SPECIFICATION.md   # Detailed project description
├── CONTRIBUTING.md            # Development guidelines
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

### Running the Analysis
```bash
# Run preprocessing and sentiment analysis
python src/preprocessing.py
python src/sentiment_analysis.py

# Merge sentiment and market data
python src/merge_finance_data.py

# Run correlation or regression analysis
python src/correlations.py
```

Results (plots and summary statistics) will be saved in the `/reports` directory.


## 6. References and Acknowledgements

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.*  
- Yahoo Finance API via `yfinance` Python library.  
- Kaggle Datasets: *Financial Sentiment Analysis*, *Tweets about Top NASDAQ Companies (2015–2020).*  
- Course: **Data Science and Advanced Programming (DSAP)** – HEC Lausanne, University of Lausanne (2025).  
- Supervisors: *Prof. Simon Scheidegger* and *Dr. Anna Smirnova.*


> **Note:**  
> This repository is part of the MSc in Finance – *Data Science and Advanced Programming* course at HEC Lausanne (Fall 2025).  
> Licensed under the MIT License.
