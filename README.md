<h1 align="center"> Emotion-Driven Markets:<br>Social Media Sentiment and NASDAQ Movements</h1>

<p align="center">
  <em>Data Science & Advanced Programming – HEC Lausanne (Fall 2025)</em><br>
  <strong>Author:</strong> Livio Manzinali  
</p>

## Abstract

This project explores the link between **social media sentiment** and **short-term fluctuations** in the **NASDAQ Composite Index**.  
Using advanced **Natural Language Processing (NLP)** and **financial data analysis**, it investigates whether variations in the public tone surrounding major NASDAQ-listed companies — such as Apple, Tesla, and Amazon — correspond to or anticipate movements in the market index.  

By applying the **FinBERT** transformer model for financial sentiment analysis and combining the results with market data from **Yahoo Finance**, this research provides insights into the relationship between **digital emotions** and **market dynamics**, contributing to the broader field of behavioral finance.

## 1. Context and Motivation

Investor psychology and collective sentiment play a critical role in shaping financial markets.  
Platforms such as **Twitter** have become real-time reflections of public opinion, where viral posts and influential statements can immediately influence stock valuations.  

This project extends behavioral finance research by **quantifying the emotional pulse of investors online** and assessing whether aggregated sentiment can predict or accompany short-term changes in the NASDAQ index.  

Beyond predictive potential, it also aims to evaluate how efficiently financial markets absorb and react to information circulating on social media platforms.

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
