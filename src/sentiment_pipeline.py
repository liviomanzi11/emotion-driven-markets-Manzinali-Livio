import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

TWEET_CSV = RAW / "Tweet.csv"
MAP_CSV = RAW / "Company_Tweet.csv"

# Only keep files you actually need
OUT_TWEET_LABELED = PROC / "tweet_sentiment_labeled.csv"
OUT_COMPANY = PROC / "company_sentiment_daily.csv"

def load_and_merge():
    tweets = pd.read_csv(TWEET_CSV, engine="python", on_bad_lines="skip")
    mapping = pd.read_csv(MAP_CSV)
    tweets = tweets.dropna(subset=["body", "post_date"])
    df = tweets.merge(mapping, on="tweet_id", how="left")
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_finbert(device):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.to(device)
    model.eval()
    return tokenizer, model

def predict_sentiment(texts, tokenizer, model, device, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT inference"):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
            scores = torch.softmax(logits, dim=1).cpu().numpy()

        for s in scores:
            results.append({
                "positive": float(s[0]),
                "neutral": float(s[1]),
                "negative": float(s[2])
            })

    return pd.DataFrame(results)

def aggregate(df):
    company = (
        df.dropna(subset=["ticker_symbol"])
          .groupby(["ticker_symbol", "date"])[["positive", "neutral", "negative"]]
          .mean()
          .reset_index()
    )
    return company

def main():
    df = load_and_merge()
    df = df.sample(n=100, random_state=42).reset_index(drop=True)

    df["clean_text"] = df["body"].apply(clean_text)
    df = df[df["clean_text"] != ""]

    df["date"] = pd.to_datetime(df["post_date"], unit="s", utc=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    df = df[(df["date"] >= pd.to_datetime("2015-01-01").date()) &
            (df["date"] <= pd.to_datetime("2020-12-31").date())]

    device = choose_device()
    tokenizer, model = load_finbert(device)

    preds = predict_sentiment(df["clean_text"].tolist(), tokenizer, model, device)
    preds = preds.reset_index(drop=True)
    df = df.reset_index(drop=True)

    assert len(df) == len(preds)

    df = pd.concat([df, preds], axis=1)
    df["sentiment"] = df[["positive", "neutral", "negative"]].idxmax(axis=1)
    df["polarity"] = df["sentiment"].map({"positive": 1, "negative": -1}).fillna(0)

    df.to_csv(OUT_TWEET_LABELED, index=False)

    company_daily = aggregate(df)
    company_daily.to_csv(OUT_COMPANY, index=False)

    print("\nFiles generated successfully:\n")
    print(f"- {OUT_TWEET_LABELED.name}")
    print(f"- {OUT_COMPANY.name}")

if __name__ == "__main__":
    main()
