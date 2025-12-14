import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path Configuration
BASE = Path(__file__).resolve().parents[2]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# SAMPLE folder
SAMPLES = PROC / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)

# Define input files
TWEET_CSV = RAW / "Tweet.csv"
MAP_CSV = RAW / "Company_Tweet.csv"

# Define output file: tweet-level sentiment with FinBERT results
OUT_TWEET_SENTIMENT = PROC / "tweet_sentiment.csv"

# SAMPLE output
OUT_TWEET_SENTIMENT_SAMPLE = SAMPLES / "tweet_sentiment_SAMPLE.csv"

# SAMPLE SIZE for automatic sample-mode
SAMPLE_SIZE = 200


def load_and_merge():
    """Load tweets and merge with company mapping."""
    tweets = pd.read_csv(TWEET_CSV, engine="python", on_bad_lines="skip")
    mapping = pd.read_csv(MAP_CSV)
    
    tweets = tweets.dropna(subset=["body", "post_date"])
    
    # Merge to get ticker_symbol for each tweet
    df = tweets.merge(mapping, on="tweet_id", how="inner")
    return df


def clean_text(text):
    """Remove URLs, mentions, hashtags, and special characters from tweet text."""
    if not isinstance(text, str):
        return ""
    # URLs and mentions add noise without semantic value for financial sentiment
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def choose_device():
    """Select GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_finbert(device):
    """Load FinBERT tokenizer and model for financial sentiment analysis."""
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_sentiment(texts, tokenizer, model, device, batch_size=32):
    """
    Predict sentiment probabilities for a list of texts using FinBERT.
    Returns a DataFrame with columns: positive, neutral, negative.
    """
    all_probs = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT inference"):
        batch = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        
        all_probs.append(probs)
    
    all_probs = np.vstack(all_probs)
    return pd.DataFrame(all_probs, columns=["positive", "neutral", "negative"])


def determine_sentiment_label(row):
    """Convert probabilities to sentiment label (positive/neutral/negative)."""
    max_col = row[["positive", "neutral", "negative"]].idxmax()
    return max_col


def determine_polarity(row):
    """Convert sentiment to polarity score: positive=1.0, neutral=0.0, negative=-1.0."""
    if row["sentiment"] == "positive":
        return 1.0
    elif row["sentiment"] == "negative":
        return -1.0
    else:
        return 0.0


def main():
    """
    Main pipeline for tweet-level sentiment analysis with FinBERT.
    
    If tweet_sentiment.csv exists:
        - Run SAMPLE mode (200 tweets) for demonstration
        - Save sample output separately
    
    If tweet_sentiment.csv does NOT exist:
        - Run FULL FinBERT inference on all tweets
        - Generate tweet_sentiment.csv (1.6GB file)
    """
    
    # Check if tweet sentiment file already exists
    full_file_exists = OUT_TWEET_SENTIMENT.exists()

    # Load raw data and merge with ticker mapping
    df = load_and_merge()

    if full_file_exists:
        print("\n✓ Detected existing tweet_sentiment.csv -> running SAMPLE FinBERT test.\n")

        # Keep only a sample of tweets (EARLY sampling to avoid loading everything)
        df = df.sample(n=min(len(df), SAMPLE_SIZE), random_state=42).reset_index(drop=True)

        # Apply text cleaning and filter out any resulting empty strings
        df["clean_text"] = df["body"].apply(clean_text)
        # FinBERT tokenizer fails on empty strings - filter them before inference
        df = df[df["clean_text"] != ""]

        # Convert UNIX timestamp to date for alignment with daily stock OHLCV data
        df["date"] = pd.to_datetime(df["post_date"], unit="s", utc=True, errors="coerce").dt.date
        df = df.dropna(subset=["date"])

        # SAMPLE FinBERT inference
        device = choose_device()
        tokenizer, model = load_finbert(device)
        preds = predict_sentiment(df["clean_text"].tolist(), tokenizer, model, device)

        preds = preds.reset_index(drop=True)
        df = df.reset_index(drop=True)
        assert len(df) == len(preds)

        df = pd.concat([df, preds], axis=1)
        
        # Polarity labels simplify downstream aggregation and feature engineering
        df["sentiment"] = df.apply(determine_sentiment_label, axis=1)
        df["polarity"] = df.apply(determine_polarity, axis=1)

        # SAMPLE OUTPUT
        df.to_csv(OUT_TWEET_SENTIMENT_SAMPLE, index=False)

        print("\n✓ Sample FinBERT output generated:")
        print(f"  - {OUT_TWEET_SENTIMENT_SAMPLE.name}")
        print("\n✓ Using existing FULL tweet_sentiment.csv for the rest of the pipeline.\n")
        return

    print("\n⚠️  No tweet_sentiment.csv found -> running FULL FinBERT inference.\n")
    print("⚠️  This will take several hours for all tweets...\n")

    # Apply text cleaning
    df["clean_text"] = df["body"].apply(clean_text)
    df = df[df["clean_text"] != ""]

    # Convert UNIX timestamp to date
    df["date"] = pd.to_datetime(df["post_date"], unit="s", utc=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # FULL FinBERT inference
    device = choose_device()
    tokenizer, model = load_finbert(device)
    preds = predict_sentiment(df["clean_text"].tolist(), tokenizer, model, device)

    preds = preds.reset_index(drop=True)
    df = df.reset_index(drop=True)
    assert len(df) == len(preds)

    df = pd.concat([df, preds], axis=1)
    
    # Add sentiment label and polarity
    df["sentiment"] = df.apply(determine_sentiment_label, axis=1)
    df["polarity"] = df.apply(determine_polarity, axis=1)

    # Save tweet-level sentiment
    df.to_csv(OUT_TWEET_SENTIMENT, index=False)

    print("\n✓ Full FinBERT inference complete.")
    print(f"  - {OUT_TWEET_SENTIMENT.name} ({len(df):,} tweets)\n")


if __name__ == "__main__":
    main()
