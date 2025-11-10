# Project Specification: Social Media Sentiment and NASDAQ Movements

## 0. Introduction

In recent years, social media has become a powerful reflection of public opinion and market sentiment.  
Investors, journalists, and even automated trading systems continuously react to online discussions, particularly those concerning major technology companies.  
Understanding how such digital emotions interact with financial markets is crucial for both behavioral finance research and practical investment strategies.

## 1. Project Overview

The purpose of this project is to analyze how **sentiment expressed on social media** about major **NASDAQ-listed companies** relates to **short-term market movements**.  
The main objective is to determine whether shifts in the collective tone of discussions — positive, neutral, or negative — **correspond to or anticipate variations in the NASDAQ index**.

In the digital era, investors are increasingly influenced by online information flows. Tweets, public statements, or viral reactions from influential figures can immediately impact stock prices. Understanding this dynamic provides insight into both **market behavior** and **investor psychology**.

## 2. Research Question / Problem Statement

The central research question guiding this project is:

> **Can social media sentiment about major NASDAQ companies help explain or predict short-term movements in the NASDAQ Composite Index?**

This leads to several sub-questions:
- Do changes in public sentiment precede or follow market fluctuations?
- Which companies or sectors show the strongest sentiment-market relationships?
- Can sentiment data improve short-term predictive models of market behavior?

## 3. Datasets

The analysis will combine textual and financial data from open sources:

- **Financial Sentiment Analysis (Kaggle)** — Labeled financial sentences from news and tweets, used for evaluating and fine-tuning FinBERT.  
- **Tweets about Top NASDAQ Companies (2015–2020)** — A dataset of tweets related to major companies such as Apple, Tesla, Amazon, and Google, used to compute daily sentiment.  
- **Yahoo Finance (^IXIC)** — NASDAQ Composite index data for the same period, retrieved via the `yfinance` Python library.

These datasets together allow for a **multi-modal approach**, linking language (sentiment) with quantitative market indicators.

## 4. Methodology

### Step 1 – Data Preparation
- Load and clean tweet data (remove duplicates, URLs, mentions, and special characters).  
- Convert timestamps and align tweets with NASDAQ trading days.  
- Aggregate tweets by date to compute a daily sentiment overview.

### Step 2 – Sentiment Analysis (NLP)
- Use **FinBERT**, a pre-trained transformer model specialized in financial language, to classify each tweet as positive, neutral, or negative.  
- Compute daily average sentiment scores and proportions of each sentiment class.

### Step 3 – Market Comparison
- Retrieve NASDAQ index data using the `yfinance` library.  
- Merge daily sentiment metrics with NASDAQ returns by date.  
- Analyze correlations between daily sentiment changes and market returns or volatility.  
- Optionally, test predictive relationships using simple regression or classification models.

### Step 4 – Visualization & Interpretation
- Plot the evolution of sentiment versus NASDAQ performance.  
- Highlight periods where strong negative sentiment preceded significant market drops.  
- Summarize findings and interpret them in the context of **behavioral finance** and **market efficiency**.

## 5. Tools and Libraries

- **Python 3.10+**  
- **Pandas, NumPy, Matplotlib, Seaborn** – Data manipulation and visualization  
- **Transformers (FinBERT)** – Sentiment classification  
- **yfinance** – Financial data retrieval  
- **Scikit-learn** – Regression and classification analysis  
- **Visual Studio Code** – Main development environment and documentation

## 6. Expected Outcomes

The project aims to evaluate whether **social media sentiment correlates with or predicts short-term NASDAQ movements**.  
Even if no strong predictive relationship is found, the research will provide valuable insights into:

- How collective online emotions evolve alongside financial trends;  
- Whether markets efficiently integrate public sentiment;  
- The broader role of **digital sentiment** in modern investor behavior.

## 7. Repository Structure

project_root/
│
├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter notebooks for analysis
├── src/ # Source code (data processing, NLP, visualization)
├── reports/ # Generated figures and final report
└── README.md # Project overview and instructions


## 8. Summary

This project combines **Natural Language Processing (FinBERT)**, **financial data analysis**, and **behavioral finance concepts** to study the connection between online emotions and market behavior.  
It uses open datasets and a reproducible Python-based workflow to demonstrate both **technical and analytical skills** relevant to applied data science in finance.
